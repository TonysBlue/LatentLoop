# E2 直接流式语音实施说明

## 1. 完成边界

E2 将模型时钟固定为 80 ms。每个 tick 接收一路 24 kHz、1920 样本的混合麦克风输入，模型生成一个 Mimi 帧，冻结的因果 decoder 将其转换成 1920 个波形采样。运行路径不经过文本或 TTS。

E2 训练受语音与多模态上下文条件化的直接语音回复。扬声器回流、房间响应、重叠说话和用户打断仍由 E4 实现。输出 codec token 不作为额外输入回灌主干。

## 2. 固定 Codec

| 字段 | 值 |
|---|---|
| Codec | Mimi |
| 采样率 | 24 kHz mono |
| 帧率 | 12.5 Hz |
| 帧长 | 80 ms / 1920 samples |
| Codebook | 8 |
| Vocab | 2048 |
| 上游 revision | `a49141e28b3d9c947cf9aa5314431e1b11cbd2f5` |
| 权重 SHA-256 | `09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50` |

主训练环境继续使用项目锁定的 PyTorch。官方 Moshi 运行于 `codec/` 的独立 uv 环境，通过权限为 `0600` 的 Unix socket 提供 `health/reset/encode_step/decode_step`。所有请求校验 protocol、session、shape、dtype 和 codec identity。

```bash
scripts/bootstrap-codec.sh
scripts/download-mimi.sh
scripts/codec-worker.sh

uv run latentloop benchmark-codec \
  --config configs/local-25m.yaml \
  --socket ~/latentloop-data/run/mimi.sock

uv run latentloop benchmark-stream \
  --config configs/local-25m.yaml \
  --socket ~/latentloop-data/run/mimi.sock
```

worker 重启后，runtime 用最近最多 250 个输出帧执行 reset 和重放，恢复 decoder 的局部连续状态。CPU `rustymimi` 可用于离线检查，但本机测得 `RTF=1.53`，不作为实时后端。

## 3. Speech Head

Speech Head 包含两级因果结构：

1. Temporal GRU 使用 `q_t`、pooled `Z_t`、上一帧 codec embedding 和局部声学状态更新跨帧状态。
2. 两层 Depth Transformer 在单帧内按 `q0 -> ... -> q7` 预测 RVQ codebook，任何位置不能读取当前或未来目标 token。

训练使用 teacher forcing，推理逐 codebook 采样。测试与评估使用 greedy；交互默认 `temperature=0.8, top_k=250`。SpeechControl 决定 `SILENT/START/CONTINUE/PAUSE/STOP`，且执行合法状态转移约束。Mimi vocab 不增加 PAD、BOS 或 EOS；Speech Head 使用内部 BOS embedding，语句边界由 SpeechControl 表达。

`RecurrentState.speech_local` 保存 temporal hidden、上一帧 codec tokens、控制状态和 utterance active 标记，并进入 checkpoint/TBPTT detach。它只负责局部声学连续性，不是认知记忆。

## 4. 数据契约

Schema v2 episode 包含：

```text
<episode>.meta.json
<episode>.mic.flac
<episode>.target_speech.flac
<episode>.speech_codes.npy
<episode>.timeline.npz
<episode>.screen.npz
<episode>.actions.json
<episode>.turns.json
<episode>.controls.npy
```

- `mic.flac` 是模型唯一音频输入。
- `target_speech.flac` 只用于离线编码、审计和波形评测，不进入模型输入。
- `speech_codes.npy` 是 `[ticks, 8] uint16`。
- `timeline.npz` 保存时间戳、各任务 mask 和稀疏屏幕索引。
- `screen.npz` 只保存发生变化的屏幕帧；纯语音 episode 可以没有有效屏幕帧。
- `controls.npy` 保存 Speech/Action/Cognitive control target，`turns.json` 保存可审计的轮次标注。
- 非合成数据必须在 manifest 中记录 `source_license` 和 `redistribution_allowed`；原始研究数据不随 shard 对外重新分发。

导入器接收逐行 JSON 对象。音频必须预先转换为 24 kHz mono；相对路径以 JSONL
所在目录为基准。`screens` 可省略，存在时为包含 `ticks: [N]` 和
`frames: [N, 3, H, W]` 的 NPZ。一个最小记录如下：

```json
{"episode_id":"e2-000001","mic_audio":"audio/e2-000001-mic.wav","target_speech":"audio/e2-000001-target.wav","source":"internal-dialogue-v1","source_license":"internal-research","redistribution_allowed":false,"language":"zh-CN","split":"train","session_id_hash":"sha256:...","scenario":"spoken-response","turns":[]}
```

`source_license` 必须是非空字符串，`redistribution_allowed` 必须是真正的 JSON
布尔值。`false` 表示产物只允许在内部训练存储中使用，不能发布 shard。数据按
`device_id_hash + session_id_hash` 分组切分，同一会话不得跨 train/validation/test。

完整离线流水线分为三步。第一步只建立 staging shard，其中 codec token 尚未编码；
训练和常规校验会明确拒绝这种 shard：

```bash
uv run latentloop import-speech \
  --config configs/local-25m.yaml \
  --manifest '~/latentloop-data/sources/e2-10h.jsonl' \
  --output '~/latentloop-data/staging/train-%06d.tar'

uv run latentloop encode-speech \
  --config configs/local-25m.yaml \
  --shards '~/latentloop-data/staging/train-*.tar' \
  --output '~/latentloop-data/processed/train-%06d.tar' \
  --socket ~/latentloop-data/run/mimi.sock

uv run latentloop validate-data \
  --config configs/local-25m.yaml \
  --shards '~/latentloop-data/processed/train-*.tar'
```

编码完成后，把生成的 `train-manifest.jsonl` 配置为 `data.manifest`，并把
`data.source` 改为 `webdataset`、`data.shards` 指向 processed shard。训练读取时会
同时验证 schema、codec revision、权重 hash、样本内容 hash 和会话切分。

首轮构造 10 小时数据，按 8/1/1 小时划分；通过后扩至 50 小时，按 40/5/5 小时划分。中文占 80%，英文占 20%。数据组成固定为开放转写语音 50%、合成语音对话 30%、真实相邻对话轮次 15%、屏幕条件任务 5%。

中文优先 WenetSpeech、AISHELL-3、KeSpeech、AISHELL-4 和 AliMeeting；英文优先 LibriTTS-R、Common Voice、DailyTalk 和 SpokenWOZ。用户侧保留真实或多音色语音；助手目标由离线 CosyVoice2 统一生成为一个获得授权的固定音色。TTS 只制造训练标签，不进入运行时。

## 5. 训练课程

1. Codec gate：流式/离线帧对齐、连续解码、权重 identity 与本机 RTF 全部通过。
2. 32 轨迹 overfit：冻结主干，仅训练 Speech Head 和 SpeechControl，所有 codebook accuracy 超过 90%。
3. 10 小时 pilot：Head 训练 3 epoch，再解冻 audio encoder、latent updater、final norm 和顶层 25% Transformer 训练 2 epoch。
4. 50 小时扩展：Head 训练 1 epoch，选择性联合训练 1 epoch；每 10 batch 插入一个 E1 replay batch。
5. 最后 30% 更新把 scheduled sampling 线性提高至 25%。

使用 `--init-from` 从 E1 checkpoint 加载形状兼容的主干权重；它与严格恢复训练的 `--resume` 互斥。Head LR 为 `1e-4`，解冻主干 LR 为 `3e-5`，3% warmup 后 cosine decay，gradient clipping 为 1.0。

## 6. 验收

- 32 条固定轨迹上 8 个 codebook 分别超过 90% accuracy。
- SpeechControl macro-F1 不低于 0.90，边界误差不超过 80 ms。
- 受控回复准确率至少 80%，中文 CER 不高于 25%，英文 WER 不高于 30%。
- 连续 10 分钟无帧漂移、NaN、削波或队列增长；边界突变 p95 不超过内部差分基线 6 dB。
- 0.0496B 本地主模型加 CUDA codec worker 峰值显存低于 7.5 GiB，codec RTF 小于 1，单帧 p95 小于 80 ms。
- 推理依赖图中不存在文本生成、TTS、额外音频通道、SelfSpeechTrace 或 codec token 回灌。

这些是 E2 的完成门槛。通用开放域对话能力依赖 E5 的 MiniCPM 迁移，不以本机从零训练模型冒充。
