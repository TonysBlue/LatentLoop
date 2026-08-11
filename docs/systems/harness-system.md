# Harness System

Harness 是模型服务的物理对端，负责麦克风、屏幕、时钟、扬声器、隔离电脑生命周期、
ControlSignal 安全校验、输入执行、receipt、任务评估和 reward。模型服务只看见
`ObservationSignal`（混合麦克风 PCM、屏幕像素/版本、时间），并返回
`ActuationSignal`（80 ms speech PCM 和已解码的统一 `ControlSignal`）；raw token、DOM、
可访问性树、receipt 和 reward 永不跨物理输入边界。

## 生产部署模块

正式启动固定加载仓库内的 `harness.deployment.qemu`：

```text
systems/harness/src/harness/deployment/qemu.py
```

该模块为每个 session 创建 QEMU/KVM overlay，并显式连接真实 screen capture、microphone
capture、speech playback、QMP input injector 和外部 task evaluator。screen 使用 SPICE
display channel（QMP `screendump` 只用于健康检查），audio 使用 SPICE/虚拟音频 channel，
键鼠使用 QMP `input-send-event`，evaluator 使用 task manifest 指定的 command 或 Unix
socket。任何设备、endpoint、codec identity 或 evaluator 健康检查失败都必须在 serve 前
fail-closed；运行中断返回 `infrastructure_failure`，不得进入策略更新。

配置必须声明 base qcow2、QMP/SPICE/audio endpoint、input backend、evaluator endpoint、
task/reward spec 和资源上限。不存在静音、空画面、NOOP executor、固定 reward 或 fixture
fallback。每次 reset、崩溃和服务退出都回收 QEMU 进程、所有 socket、临时屏幕文件和 overlay，
并报告 `environment_id/version/protocol/action_schema_id` 供 control server 比对。

进程内 fake backend 只用于显式标记的契约/集成测试，正式 Canary、Pilot、Production 配置
没有 fake fallback。
