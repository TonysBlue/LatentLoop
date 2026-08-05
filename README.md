# LatentLoop

实时流多模态 LatentLoop 的研究实现。当前基础设施已经包含有界逐层 KV、固定容量
`Z_t`、单路混合麦克风、屏幕输入、直接 codec token Speech Head、独立 Action Head、
TBPTT、精确断点恢复、WebDataset、CPU-only Ray 任务和 W&B Local。

完整架构见 `docs/realtime-multimodal-latent-loop.md`，本地训练设计见
`docs/local-training-platform.md`。直接流式语音的实现与运行命令见
`docs/direct-speech.md`。

## 环境初始化

```bash
scripts/bootstrap.sh
uv run latentloop doctor --config configs/smoke.yaml
uv run pytest
```

`doctor` 会在 GPU 上执行 FP16 前向、反向和 `optimizer.step`。三个配置档位的参数量可用
以下命令确认：

```bash
uv run latentloop inspect-model --config configs/smoke.yaml
uv run latentloop inspect-model --config configs/local-25m.yaml
uv run latentloop inspect-model --config configs/research-0.2b.yaml
```

当前档位参数量为 0.0003B、0.0496B 和 0.2391B。0.0496B 与 0.2391B 档位均已在
RTX 2080 SUPER 8GB 上完成 FP16 反向和 AdamW 更新；实际长期训练仍应以运行时记录的
`runtime/peak_memory_*` 指标为准。

## 数据与训练

生成并校验合成 WebDataset：

```bash
uv run latentloop generate-data \
  --config configs/smoke.yaml \
  --output "$HOME/latentloop-data/processed/train-%06d.tar"

uv run latentloop validate-data \
  --config configs/smoke.yaml \
  --shards "$HOME/latentloop-data/processed/train-*.tar"
```

32 轨迹真实 Mimi 过拟合门禁使用 `configs/direct-speech-overfit.yaml`，完整命令和本机
验收结果见 [`docs/direct-speech.md`](docs/direct-speech.md)。

使用 Ray 的 CPU worker 生成数据：

```bash
uv run latentloop generate-data \
  --config configs/smoke.yaml \
  --output "$HOME/latentloop-data/processed/train-%06d.tar" \
  --ray
```

运行 smoke 训练或恢复 checkpoint：

```bash
uv run latentloop train --config configs/smoke.yaml
uv run latentloop train \
  --config configs/smoke.yaml \
  --resume "$HOME/latentloop-data/checkpoints/step-00000002.pt"
```

训练数据、checkpoint、W&B 离线 run 和 Ray 报告默认写入 `~/latentloop-data`，不进入 Git。

## W&B Local

```bash
scripts/wandb-local.sh up
scripts/wandb-local.sh status
```

服务只监听 `http://127.0.0.1:8080`。首次启动需要完成数据库迁移；脚本以 `/ready`
为就绪条件。打开 UI 获取本地 API key 后执行：

```bash
uv run wandb login --host http://127.0.0.1:8080
uv run latentloop train \
  --config configs/smoke.yaml \
  --set tracking.enabled=true \
  --set tracking.mode=online
```

服务不可用或未登录时，Tracker 自动写入本地 offline run。服务管理与备份命令：

```bash
scripts/wandb-local.sh logs
scripts/backup-wandb.sh
scripts/wandb-local.sh down
scripts/restore-wandb.sh ~/latentloop-data/backups/wandb-local-<timestamp>.tar.gz
```
直接流式语音使用 80 ms 主时钟与 Mimi 24 kHz codec。实现和运行命令见
[`docs/direct-speech.md`](docs/direct-speech.md)。

Canary/Pilot 数据准备、外部语料许可证锁、CosyVoice/ASR/屏幕适配器以及自动审计门禁见
[`docs/pilot-data.md`](docs/pilot-data.md)。本地 fixture 可以在不下载语料和模型的
情况下跑通六个阶段：

真实一小时 Canary 的固定数据下载、TTS/ASR、Mimi 编码、训练和 validation/test 评测
使用统一入口，完整说明见 [`docs/canary-runbook.md`](docs/canary-runbook.md)：

```bash
CANARY_MAX_UPDATES=5 CANARY_TRACKING_MODE=offline scripts/run-canary.sh all
```

```bash
uv run latentloop fetch-pilot-data --config configs/local-25m.yaml --fixture
uv run latentloop select-pilot-voices --config configs/local-25m.yaml --fixture
uv run latentloop build-pilot-text --config configs/local-25m.yaml --dataset canary --fixture
uv run latentloop synthesize-pilot --config configs/local-25m.yaml --dataset canary --fixture
uv run latentloop build-pilot-manifest --config configs/local-25m.yaml --dataset canary --fixture
uv run latentloop audit-pilot-data --config configs/local-25m.yaml --dataset canary --fixture
```
