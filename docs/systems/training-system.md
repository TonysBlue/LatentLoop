# Training System

Training System 统一执行 `Pretrain -> SFT -> Online GRPO`。Pretrain/SFT 通过 Data replay
训练 Model Core；Online GRPO 由 Training Policy Runner 通过 Harness physical rollout
client 获取 ObservationSignal，并提交由 token decoder 还原的 ActuationSignal。训练进程
重新计算 log-prob 并更新完整模型。

Reward、receipt、old/reference log-prob 和 task evaluator 信息只存在训练 trace，不进入
ObservationSignal。

训练系统只依赖正式的 `model`、`data`、`media`、`runtime`、`contracts` 包以及 Harness
control client。模型、数据、checkpoint、GRPO、evaluation 和 tracking 的实现分别由这些正式
workspace 边界提供；旧单体包不再作为 Training System 的实现来源或反向依赖。

Online GRPO 遇到 Harness/QEMU/codec 的基础设施失败时终止该 rollout 并记录失败类别，但不把
失败伪装成低 task reward。只有 identity、session、unit 和 receipt 均通过校验的完整 rollout
才能组成 group 并进入策略更新。
