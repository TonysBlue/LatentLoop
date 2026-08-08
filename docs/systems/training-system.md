# Training System

Training System 统一执行 `Pretrain -> SFT -> Online GRPO`。Pretrain/SFT 通过 Data replay
训练 Model Core；Online GRPO 由 Training Policy Runner 编排 Harness 的真实环境 rollout，
训练进程重新计算 log-prob 并更新完整模型。

Reward、receipt、old/reference log-prob 和 task evaluator 信息只存在训练 trace，不进入
ObservationSignal。
