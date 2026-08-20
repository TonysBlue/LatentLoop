# 物理 Rollout 闭环

Online Recurrent PPO 的正式 rollout 使用 Harness 作为唯一环境边界。训练进程在 GPU 上运行
Model Core，Harness 进程负责采集 ObservationSignal、执行 ActuationSignal，并返回
receipt；独立 Reward Judge 产生只基于观察的 Reward Event。训练进程不再向环境发送 speech
token 或 raw ActionFrame tensor。

```text
Harness.start_lifetime_session -> ObservationSignal
Model Core.forward -> Speech token + Structured ActionFrame
Training codec/frame decoder -> ActuationSignal
Harness.apply(ActuationSignal) -> ObservationSignal + EnvironmentReceipt
Observation timeline -> frozen perceptual Reward Judge -> RewardEvent
```

模型内部的 speech token、ActionFrame、old/reference log-prob 和 advantage 只保存在训练 trace；物理 socket
只传输 protobuf ObservationSignal、ActuationSignal 和控制面的 receipt。Reward Judge socket
只接收 canonical ObservationSignal bytes 与其 hash-chain identity，不接收 receipt、action、
task manifest、DOM 或隐藏 evaluator 字段。

正式 Harness control socket 必须校验 environment identity、session、unit
和 action schema。连接失败、设备失败、QEMU 崩溃或 receipt 不完整时，rollout 标记为
infrastructure failure，不作为低 reward 样本更新策略。

## 会话和时钟约束

`start_lifetime_session(session_id, snapshot)` 为生命期会话创建操作；普通任务完成不会 reset。
训练暂停后的 `resume_lifetime_session(session_id, expected_next_unit)` 只能返回该 session 当前
未消费 observation，不创建 backend、不恢复 snapshot，也不改变 unit cursor；session 不存在或
cursor 不一致时 fail closed。
空 task/session identity、重复或跳跃 unit、
错 session、重复或跳跃 unit 必须 fail-closed。动态画面的动作时延由策略学习处理，控制事件
不绑定屏幕 revision。

生命期创建返回 unit 0；unit N 的 ActuationSignal 只能作用于同一 session 的 unit N
ObservationSignal，并返回 unit N receipt 和 unit N+1 ObservationSignal。80 ms 音频单元固定为
24 kHz 单声道，即 float32 PCM 为 7,680 bytes，int16 PCM 为 3,840 bytes。Harness 不接受
任意长度 PCM 来掩盖音频设备或 codec 错误。

EnvironmentReceipt 使用与物理 signal 相同版本的 protobuf schema，外层 control request
只负责 operation、session identity 和 length framing。普通任务完成不终止 lifetime session；
`infrastructure_failure` 表示样本不可用于策略更新，且不进入下一轮 ObservationSignal。

## 生产 backend

正式 backend 是 per-session QEMU/KVM 实例。Harness 为每个 session 创建唯一 qcow2 overlay，
通过 QMP 完成 readiness、健康检查、屏幕采集和受控关机，并在 close、reset 失败或服务退出时
回收进程、socket、临时屏幕文件和 overlay。麦克风采集、语音播放和键鼠注入由部署 adapter
显式提供；adapter 或所需设备不存在时服务拒绝启动，不得回退到静音、空画面、NOOP executor
或 fixture reward。

## 契约测试设计

契约测试必须覆盖 protobuf round-trip、模型输出张量不跨物理边界、80 ms PCM 校验、identity 查询不
泄漏 backend、重复 reset 关闭旧 backend、未知 session、错序 unit、错 session、
backend 返回错误 identity、server shutdown 清理全部 session，以及 receipt protobuf
round-trip。QEMU 生命周期测试使用可控假进程和 QMP server 验证唯一 overlay、readiness、
崩溃诊断和资源回收。

端到端 Online Recurrent PPO 测试使用 fake codec worker 与实现同一严格协议的进程内 Harness，
调用正式 `train_online_ppo` 和 `PhysicalRolloutClient`。测试使用一个 lifetime session、多个
连续 RewardEvent，并至少完成一次 optimizer update。它必须断言：

1. speech token 经正式 worker 解码为恰好一个 80 ms PCM，每 unit ActionFrame 解码为
   有序 `ControlSignal` 并立即执行，Harness 收不到 raw frame tensor。
2. reset、apply、receipt 和下一轮 observation 的 session/unit 严格递增，
   speech PCM 与 action 均经过 protobuf round-trip。
3. receipt、任务成功、DOM 和 evaluator 字段不能发送给 Reward Judge，也不能出现在下一轮
   `ObservationSignal` 或模型输入张量；Reward Event 单独保存在训练元数据。
4. `infrastructure_failure`、设备断连、错序 unit 或 safety reject 不产生 advantage，也不
   参加策略梯度；只有 finalization watermark 覆盖且 identity 正确的 sealed window 才更新。
5. 更新后的 checkpoint 记录 schema/codec/environment/reward identity、reward 分量、
   有效 unit 数和实际 optimizer update。

真实部署验收还要在 QEMU/KVM、SPICE audio/display、QMP input 和独立 Reward Judge endpoint
上运行同一矩阵；本机缺少外部设备时必须明确标记未执行，不得用 fixture 结果声称闭环通过。
