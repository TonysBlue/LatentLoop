# Data System

Data 是跨 Capture、Replay、Pretrain、SFT、Online Rollout 和 Evaluation 的共享持久化领域。
它管理 schema v5、episode/trajectory/rollout、WebDataset、manifest/hash、Mimi target 编码、
action target、审计和 readiness。数据目录位于仓库外的 `~/latentloop-data/datasets`。
