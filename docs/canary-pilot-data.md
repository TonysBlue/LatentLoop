# Canary / Pilot 数据流水线

这条流水线先构造 1 小时 Canary，再构造独立的 10 小时 Pilot。Canary 的 source
utterance、plan、speaker、session 和 scenario 都会写入选择账本；Pilot 构建前会读取
Canary 清单并拒绝任何 source 或 plan 复用。episode 时长是完整时间线，包含 lead silence、
回复等待和 tail silence，不是只统计有效语音。

## 固定配方

| bucket | Pilot 时长 | 中文/英文 |
|---|---:|---:|
| public real speech | 3.0 h | 2.4 h / 0.6 h |
| synthetic computer dialogue | 5.0 h | 4.0 h / 1.0 h |
| adjacent conversational turns | 1.5 h | 1.2 h / 0.3 h |
| screen-conditioned tasks | 0.5 h | 0.4 h / 0.1 h |

每个 bucket 再按 80/10/10 切 train/validation/test。中文占 80%，英文占 20%。
短/中/长 episode 为 60%（4–16 s）、25%（16–32 s）、15%（32–60 s）。Pilot 的
电脑助手时间线约 55%，因为公开真实语音和相邻轮次合计 45%；不把这组配方宣称成 70%。

当前数据配方不加入播放回流、显式噪声增强、重叠说话、打断和反馈环路；这些由后续声学环境闭环覆盖。每个 episode
只有一路 `mic_audio`，助手目标只出现在 `target_speech`，运行时不经过 TTS。

## 产物树

默认根目录为 `~/latentloop-data/datasets`，也可以给每个命令传 `--root`：

```text
registry/{source-lock.json,licenses/,normalized/,voices/,reports/}
{canary,pilot}/{text/,synthesized/,normalized/,manifests/,shards/,reports/}
```

运行期音频、shard、模型和审计报告都在该根目录，不进入 Git。

## Fixture 闭环

fixture 不下载公开语料、不冒充 CosyVoice，也不代表模型质量；它只用于验证所有阶段、哈希
和门禁：

```bash
ROOT="$HOME/latentloop-data/datasets-fixture"
CFG=configs/local-dev.yaml

uv run data fetch-pilot-data --config "$CFG" --root "$ROOT" --fixture
uv run data select-pilot-voices --config "$CFG" --root "$ROOT" --fixture

for DATASET in canary pilot; do
  uv run data build-pilot-text --config "$CFG" --root "$ROOT" \
    --dataset "$DATASET" --fixture
  uv run data synthesize-pilot --config "$CFG" --root "$ROOT" \
    --dataset "$DATASET" --fixture
  uv run data build-pilot-manifest --config "$CFG" --root "$ROOT" \
    --dataset "$DATASET" --fixture
  uv run data audit-pilot-data --config "$CFG" --root "$ROOT" \
    --dataset "$DATASET" --fixture
done
```

审计输出 `data-card.md`、`license-report.json`、`quota-report.json`、
`quality-report.json` 和 manifest SHA-256。质量审计完全由哈希、格式、时间轴、配额、
许可证、ASR、响度和 Mimi decode-check 自动完成，不读取人工 review ledger。Pilot 构建前
必须已有通过审计的 Canary。

## 生产适配器

生产模式不会猜测缺失依赖。每个外部适配器都通过 JSON request 和 `--output` 接口调用：

- `fetch-pilot-data --lock lock.json --download --extract`：只接受锁定 URL、版本、许可证和 archive SHA-256。DailyTalk 没有可验证的匿名下载地址时直接阻断 Canary。
- `build-pilot-text`：生成 1,200 条计划（960 中文、240 英文）。计划由固定模板和哈希确定，自动进入合成；不依赖人工审核。
- `select-pilot-voices --library voices.json`：要求恰好一个获得授权的固定助手声线，以及按 split 隔离的中英文用户声线。声线 prompt 和授权记录必须有 SHA-256。
- `synthesize-pilot --synth-command CMD --asr-command CMD --model-sha256 HASH`：调用 CosyVoice 和 ASR。ASR 中文 CER、英文 WER 单条超过 20% 时重试一次后剔除；聚合门禁为 8%。合成适配器还必须写同名 `.metrics.json`，其中 `integrated_lufs` 在 `-23 +/- 1 LUFS` 内。
- `build-pilot-manifest --normalize-command CMD --screen-command CMD`：normalizer 负责 24 kHz mono PCM16 FLAC、峰值、loudness 和 source inventory；screen adapter 只允许隔离 sandbox 的稀疏屏幕帧。

源清单扩展字段包括 `source_version`、`source_url`、`source_utterance_ids`、
`template_id`、`intent`、`user_voice_id`、`assistant_voice_id`、`turns`、
`target_segments`、`recipe_sha256` 和 `license_sha256`。

## 标签语义

`target_segments` 的 `start_sample` 必须落在 80 ms tick。包含最后助手采样的帧是
assistant segment 内的 unit 标记为 `speech_mode=SPEECH`，其余 unit 标记为
`speech_mode=SILENCE`；codec mask 只在 SPEECH unit 有效。
其他帧为 false。公共真实输入 episode 的 target 全静音、没有 codec loss。Action/Memory
mask 在 Pilot 保持 false，`PAUSE` 不使用。

## 审计和训练导入

```bash
uv run data audit-pilot-data --config "$CFG" --root "$ROOT" --dataset pilot
uv run data import-speech --config "$CFG" \
  --manifest "$ROOT/pilot/manifests/train.jsonl" \
  --output "$ROOT/pilot/shards/staging/train/train-%06d.tar"
```

审计拒绝重复 ID、跨 split speaker/session/template/scenario、时间戳和音频错误、错误的
codec identity、缺失许可证哈希、Canary/Pilot 复用、配额超差、ASR 超标以及缺少 100 段
Mimi decode-check 的正式 Pilot。通过后再使用既有 `encode-speech` 生成 processed shard。

不需要人工 review ledger。可以用编排命令自动完成所有数据准备阶段：

```bash
ROOT="$HOME/latentloop-data/datasets"
CFG=configs/pilot.yaml

uv run data prepare-pilot-data --config "$CFG" --root "$ROOT" \
  --lock "$ROOT/registry/source-lock.json" --download --extract \
  --library "$ROOT/registry/voices/voice-library.json" \
  --synth-command 'path/to/cosyvoice-adapter' \
  --asr-command 'path/to/asr-adapter' \
  --model-sha256 '<64-char-cosyvoice-model-sha256>' \
  --normalize-command 'path/to/normalizer-adapter' \
  --screen-command 'path/to/screen-adapter' \
  --socket "$HOME/latentloop-data/runtime/sockets/mimi.sock" --encode
```

`prepare-pilot-data` 会先完成 Canary 并通过自动审计，再构建排除 Canary 的 Pilot；`--encode`
会为 train、validation、test 生成 staging 和 processed shards。外部数据、CosyVoice、ASR、
屏幕采集和 Mimi 仍必须提供真实适配器和锁定哈希，命令不会用 fixture 冒充生产产物。

训练前最后执行 fail-closed 检查：

```bash
uv run data check-readiness --config configs/pilot.yaml --root "$ROOT" \
  --checkpoint "$HOME/latentloop-data/checkpoints/base/state-loop.pt"
```

该检查确认 audit、三个 split 的 manifest/shard、编码状态、Mimi 报告、初始 checkpoint 和磁盘
空间；失败时不会进入训练。

编码完成后，Pretrain 和 SFT 使用各自锁定的监督 manifest；Online RL 使用真实
隔离环境的 session manifest。三者由同一个 Pilot recipe 按
`Pretrain -> SFT -> Online RL` 顺序编排；Online RL 的算法为 Online Recurrent PPO：

```bash
scripts/run-training.sh --recipe configs/recipes/pilot.yaml
```
