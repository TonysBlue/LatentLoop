# 轨迹 Schema v5

监督 episode、SFT 样本和 Online GRPO rollout 统一使用 schema v5。schema 是跨数据准备、训练、
评估和 checkpoint lineage 的稳定身份，不因 Canary/Pilot/Production 规模变化。

每个 `meta.json` 和 manifest entry 必须包含 `schema_version=5`、`dataset_scale`、
`sample_kind`、`stage`、`supervision_kind`、`task_id`、`environment_id/version`、
`protocol_version`、`action_vocabulary_id`、`codec_id/revision/weight_hash` 以及完整的
`runtime_identity`。每个 unit 固定 80 ms、24 kHz 单声道、每 unit 一帧 Mimi，并记录
`timestamp_ms/delta_ms`、混合 `mic`、screen frame/index/revision、speech mode/mask、
speech codes/mask、action tokens/mask。

decoded `ControlSignal`、`EnvironmentReceipt`、`RewardBreakdown` 仅作为审计和 rollout
lineage 的 control-plane metadata，不能被 reader 拼入模型输入。processed sample 必须
记录 `speech_codes_encoded=true`、unit 数、持续时间和 content SHA-256。

v4 不得被 reader 静默读取。正式迁移必须从 source manifest 重新构建 staging shards，经
正式 Mimi worker 重编码和 decode/length/hash audit 后，原子写入新的
`shards/processed/<split>/`；旧目录只读保留在 archive，验证通过后才切换 manifest。
