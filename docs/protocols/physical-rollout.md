# 物理 Rollout 闭环

Online GRPO 的正式 rollout 使用 Harness 作为唯一环境边界。训练进程在 GPU 上运行
Model Core，Harness 进程负责采集 ObservationSignal、执行 ActuationSignal，并返回
receipt/reward。训练进程不再向环境发送 speech/action token。

```text
Harness.reset -> ObservationSignal
Model Core.forward -> Speech/Action token
Training codec/action decoder -> ActuationSignal
Harness.apply(ActuationSignal) -> ObservationSignal + EnvironmentReceipt
Harness.evaluate -> RewardBreakdown
```

模型内部的 token、old/reference log-prob 和 advantage 只保存在训练 trace；物理 socket
只传输 protobuf ObservationSignal、ActuationSignal 和控制面的 receipt/reward。

正式 Harness control socket 必须校验 environment identity、session、unit、screen revision
和 action vocabulary。连接失败、设备失败、QEMU 崩溃或 receipt 不完整时，rollout 标记为
infrastructure failure，不作为低 reward 样本更新策略。

## 会话和时钟约束

`reset(task_id, seed, session_id)` 为唯一的会话创建操作。同一 `session_id` 再次 reset 时，
Harness 必须先关闭旧 backend 并回收旧 overlay；空 task/session identity、重复或跳跃 unit、
回退的 screen revision、与当前画面 revision 不一致的控制事件都必须 fail-closed。

每次 reset 返回 unit 0；unit N 的 ActuationSignal 只能作用于同一 session 的 unit N
ObservationSignal，并返回 unit N receipt 和 unit N+1 ObservationSignal。80 ms 音频单元固定为
24 kHz 单声道，即 float32 PCM 为 7,680 bytes，int16 PCM 为 3,840 bytes。Harness 不接受
任意长度 PCM 来掩盖音频设备或 codec 错误。

EnvironmentReceipt 和 RewardBreakdown 使用与物理 signal 相同版本的 protobuf schema，外层
control request 只负责 operation、task/session identity 和 length framing。receipt 的
`terminated` 只控制 episode 生命周期；`infrastructure_failure` 表示样本不可用于策略更新，
二者都不进入下一轮 ObservationSignal。

## 生产 backend

正式 backend 是 per-session QEMU/KVM 实例。Harness 为每个 session 创建唯一 qcow2 overlay，
通过 QMP 完成 readiness、健康检查、屏幕采集和受控关机，并在 close、reset 失败或服务退出时
回收进程、socket、临时屏幕文件和 overlay。麦克风采集、语音播放、键鼠注入和任务 evaluator
由部署 adapter 显式提供；adapter 或所需设备不存在时服务拒绝启动，不得回退到静音、空画面、
NOOP executor 或 fixture reward。

## 契约测试设计

契约测试必须覆盖 protobuf round-trip、token 不跨物理边界、80 ms PCM 校验、identity 查询不
泄漏 backend、重复 reset 关闭旧 backend、未知 session、错序 unit、错 session、过期 screen
revision、backend 返回错误 identity、server shutdown 清理全部 session，以及 receipt/reward
protobuf round-trip。QEMU 生命周期测试使用可控假进程和 QMP server 验证唯一 overlay、readiness、
崩溃诊断和资源回收；端到端 Online GRPO 测试使用 fake codec 与实现同一严格协议的 Harness，
验证环境只收到 ActuationSignal，reward/receipt 只进入训练 trace。
