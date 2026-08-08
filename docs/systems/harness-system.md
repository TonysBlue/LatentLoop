# Harness System

Harness 负责物理设备和隔离电脑环境：麦克风、屏幕、时钟、扬声器、QEMU/KVM snapshot、
ControlSignal 安全校验、输入执行、receipt、任务评估和 reward。正式 backend 为 QEMU/KVM
per-session overlay；进程内 fake backend 只用于测试。

Harness 接收 speech PCM 和 decoded ControlSignal，不持有 Model Core 的 recurrent state，
也不向模型输入隐藏成功标签、DOM 或 reward。
