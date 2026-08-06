# 真实 Canary 运行手册

本流程在单张 8 GB NVIDIA GPU 上依次运行 CosyVoice2、Mimi 和主模型，避免三个模型
同时占用显存。数据是固定版本的 AISHELL-1、LibriSpeech、AISHELL-4 和 DailyTalk；
所有 archive、许可证、TTS、ASR 和 Mimi 权重都通过 SHA-256 或固定 revision 校验。
流程不使用 fixture，也不依赖人工审核。

## 前置条件

- WSL2/Linux、NVIDIA GPU 和可用的 `nvidia-smi`
- `uv`、Git、curl，以及至少 20 GB 可用磁盘
- 可访问 GitHub 和 Hugging Face
- 可选：W&B Local 已在 `http://127.0.0.1:8080` 运行并完成 CLI 登录

默认产物位于：

```text
~/latentloop-data/canary-sources/  固定公开数据下载缓存
~/latentloop-data/models/          CosyVoice2、SenseVoice、Mimi 权重
~/latentloop-data/vendor/          固定 revision 的 CosyVoice 源码
~/latentloop-data/pilot-data/      Canary 中间产物、manifest 和 shard
~/latentloop-data/canary-run/      checkpoint、评测、日志和 W&B run
```

## 一键执行

首次运行先用 5 个 update 验证完整闭环。离线 tracking 不要求 W&B 登录：

```bash
cd ~/LatentLoop
CANARY_MAX_UPDATES=5 CANARY_TRACKING_MODE=offline scripts/run-canary.sh all
```

每个阶段都有内容哈希和收据，可以安全重跑。中断后执行同一命令会复用已验证的下载、
规范化音频、TTS 缓存和 episode。脚本会按阶段启动和关闭 worker，失败时也会清理当前
worker。

终端只显示阶段任务、运行时间和关键指标。各子命令的完整 stdout/stderr 保存到本次运行目录：

```text
~/latentloop-data/canary-run/logs/<UTC 时间>-<PID>/
  prepare.log
  encode.log
  train.log
  evaluate.log
```

长阶段每 30 秒打印一次 elapsed time。阶段失败时，终端会自动打印对应日志的最后 60 行；
无需重新运行即可定位失败。需要固定日志路径时设置 `CANARY_LOG_DIR=/path/to/logs`。

## 分阶段执行

需要定位问题或观察产物时使用：

```bash
# 1. 下载并校验真实语料、CosyVoice2 和 SenseVoice
scripts/run-canary.sh bootstrap

# 2. 规范化、生成计划、CosyVoice2 合成、SenseVoice CER/WER 门禁、构造 manifest
scripts/run-canary.sh prepare

# 3. Mimi decode 门禁、正式审计和三个 split 的 codec 编码
scripts/run-canary.sh encode

# 4. 训练 5 update 并评测 validation/test
CANARY_MAX_UPDATES=5 CANARY_TRACKING_MODE=offline scripts/run-canary.sh train
```

W&B Local 已登录时可把最后一条的 tracking 切成在线：

```bash
CANARY_MAX_UPDATES=5 CANARY_TRACKING_MODE=online scripts/run-canary.sh train
```

## 正式 Canary 训练

短闭环成功后运行配置中的 2000 updates：

```bash
CANARY_RUN_ROOT="$HOME/latentloop-data/canary-run-2000" \
CANARY_MAX_UPDATES=2000 \
CANARY_TRACKING_MODE=online \
scripts/run-canary.sh train
```

没有兼容的状态闭环基础 checkpoint 时，这会从随机初始化训练全部主干，用于验证训练系统和观察
Canary 拟合曲线；有兼容 checkpoint 时设置：

```bash
CANARY_INIT_CHECKPOINT="$HOME/latentloop-data/checkpoints/state-loop.pt" \
CANARY_MAX_UPDATES=2000 scripts/run-canary.sh train
```

## 验收产物

成功的一键短闭环至少包含：

```text
~/latentloop-data/pilot-data/reports/canary-audit.json
~/latentloop-data/pilot-data/reports/canary-codec-benchmark.json
~/latentloop-data/pilot-data/reports/canary-readiness.json
~/latentloop-data/pilot-data/processed/canary/{train,validation,test}/*.tar
~/latentloop-data/canary-run/checkpoints/step-00000005.pt
~/latentloop-data/canary-run/runs/training.json
~/latentloop-data/canary-run/runs/validation-evaluation.json
~/latentloop-data/canary-run/runs/test-evaluation.json
```

`canary-audit.json` 必须为 `passed: true`，readiness 必须没有失败项，validation/test
评测必须读取到非零 episode 和 speech frame。`training.json` 记录训练耗时、吞吐、显存、
最后一次训练指标、W&B 实际模式和 run URL。5 update 只证明真实数据、codec、训练、
checkpoint 和评测链路可运行；此时 codec accuracy、macro-F1 等质量指标不用于判断收敛。

## 自定义目录

所有运行目录都可以通过环境变量覆盖：

```bash
PILOT_DATA_ROOT=/data/pilot-data \
CANARY_SOURCE_CACHE=/data/canary-sources \
LATENTLOOP_MODEL_ROOT=/data/models \
LATENTLOOP_VENDOR_ROOT=/data/vendor \
CANARY_RUN_ROOT=/data/canary-run \
CANARY_MAX_UPDATES=5 \
CANARY_TRACKING_MODE=offline \
scripts/run-canary.sh all
```
