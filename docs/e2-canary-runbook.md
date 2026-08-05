# E2 真实 Canary 运行手册

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
~/latentloop-data/e2-canary-sources/  固定公开数据下载缓存
~/latentloop-data/models/             CosyVoice2、SenseVoice、Mimi 权重
~/latentloop-data/vendor/             固定 revision 的 CosyVoice 源码
~/latentloop-data/e2-pilot/           Canary 中间产物、manifest 和 shard
~/latentloop-data/e2-canary-run/      checkpoint、评测和 W&B run
```

## 一键执行

首次运行先用 5 个 update 验证完整闭环。离线 tracking 不要求 W&B 登录：

```bash
cd ~/LatentLoop
E2_CANARY_MAX_UPDATES=5 E2_TRACKING_MODE=offline scripts/run-canary.sh all
```

每个阶段都有内容哈希和收据，可以安全重跑。中断后执行同一命令会复用已验证的下载、
规范化音频、TTS 缓存和 episode。脚本会按阶段启动和关闭 worker，失败时也会清理当前
worker。

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
E2_CANARY_MAX_UPDATES=5 E2_TRACKING_MODE=offline scripts/run-canary.sh train
```

W&B Local 已登录时可把最后一条的 tracking 切成在线：

```bash
E2_CANARY_MAX_UPDATES=5 E2_TRACKING_MODE=online scripts/run-canary.sh train
```

## 正式 Canary 训练

短闭环成功后运行配置中的 2000 updates：

```bash
E2_CANARY_RUN_ROOT="$HOME/latentloop-data/e2-canary-run-2000" \
E2_CANARY_MAX_UPDATES=2000 \
E2_TRACKING_MODE=online \
scripts/run-canary.sh train
```

没有兼容 E1 checkpoint 时，这会从随机初始化训练全部主干，用于验证训练系统和观察
Canary 拟合曲线；有兼容 checkpoint 时设置：

```bash
E2_INIT_CHECKPOINT="$HOME/latentloop-data/checkpoints/e1.pt" \
E2_CANARY_MAX_UPDATES=2000 scripts/run-canary.sh train
```

## 验收产物

成功的一键短闭环至少包含：

```text
~/latentloop-data/e2-pilot/reports/canary-audit.json
~/latentloop-data/e2-pilot/reports/canary-readiness.json
~/latentloop-data/e2-pilot/processed/canary/{train,validation,test}/*.tar
~/latentloop-data/e2-canary-run/checkpoints/step-00000005.pt
~/latentloop-data/e2-canary-run/runs/validation-evaluation.json
~/latentloop-data/e2-canary-run/runs/test-evaluation.json
```

`canary-audit.json` 必须为 `passed: true`，readiness 必须没有失败项，validation/test
评测必须读取到非零 episode 和 speech frame。5 update 只证明真实数据、codec、训练、
checkpoint 和评测链路可运行，不代表模型质量收敛。

## 自定义目录

所有运行目录都可以通过环境变量覆盖：

```bash
E2_ROOT=/data/e2-pilot \
E2_CANARY_SOURCE_CACHE=/data/e2-canary-sources \
E2_MODEL_ROOT=/data/models \
E2_SOURCE_ROOT=/data/vendor \
E2_CANARY_RUN_ROOT=/data/e2-canary-run \
E2_CANARY_MAX_UPDATES=5 \
E2_TRACKING_MODE=offline \
scripts/run-canary.sh all
```
