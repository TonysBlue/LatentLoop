# 轨迹 Schema v5

监督 episode 和 online rollout 统一记录 schema v5 identity、unit 时钟、混合 mic、屏幕
revision、speech/action mask、decoded ControlSignal、receipt 和 lineage。v4 不得静默读取，
必须显式迁移或重新生成。
