# 轨迹 Schema v6

监督 episode、SFT 样本和 Online GRPO rollout 统一使用 schema v6。schema 是数据准备、
训练、评估和 checkpoint lineage 的稳定身份，不因 Canary/Pilot/Production 规模变化。

每个 `meta.json` 和 manifest entry 必须包含 `schema_version=6`、`dataset_scale`、
`sample_kind`、`stage`、`supervision_kind`、`task_id`、`environment_id/version`、
`protocol_version`、`action_schema_id=structured-action-v1`、codec identity 和完整
`runtime_identity`。每个 unit 固定 80 ms、24 kHz 单声道、每 unit 一帧 Mimi。

timeline 按 unit 保存：

```text
timestamp_ms, delta_ms, screen frame/index/revision
speech_mode, speech_mode_mask, speech_codes, speech_codec_mask
action_kind, action_supervision_mask
action_coordinate_cell, action_coordinate_residual
action_button, action_button_phase, action_scroll_delta
action_text_bytes[16], action_text_length
action_hotkey_keys[8], action_hotkey_length
```

`NO_ACTION` 是有监督决策；缺失专家 action 标签必须使用
`action_supervision_mask=false`。参数 mask 由 kind 和 length 确定，不在数据中保存重复
mask。TYPE continuation 属于按时间线重放得到的 Action local state，不增加公开 frame 字段。

decoded `ControlSignal`、`EnvironmentReceipt`、`RewardBreakdown` 仅作为审计和 rollout
lineage 的 control-plane metadata，reader 不得把它们拼入模型输入。processed sample 必须
记录 `speech_codes_encoded=true`、unit 数、持续时间和 content SHA-256。

旧 schema 与 `action_tokens/action_token_mask` 不得被 reader 静默读取。正式迁移必须从
source manifest 重新构建 staging shards，经 Mimi worker 和 structured-action audit 后原子
切换 processed manifest；不提供 flat-token 转换兼容分支。

```text
unit_ms            = 80
audio_sample_rate  = 24000
audio_samples      = 1920
codec_frame_rate   = 12.5
codec_frames/unit  = 1
```

## 1. StreamUnit

```python
@dataclass
class StreamUnit:
    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    screen: Tensor
    screen_valid: Tensor
    screen_revision: Tensor
    speech_mode: Tensor
    speech_mode_mask: Tensor
    speech_codes: Tensor
    speech_codec_mask: Tensor
    action: ActionFrame
    action_supervision_mask: Tensor
```

`delta_ms` 为正，时间戳严格递增；每个 target 都有明确监督 mask。reader 必须校验 kind
条件参数、TYPE/HOTKEY length、数值边界、screen revision 和 episode 时间顺序。

## 2. Online rollout

每个 rollout unit 记录 sampled ActionFrame、frame joint old/reference log-prob、解码后的
有序 controls、receipt 和 reward lineage。receipt/reward 不进入 ObservationSignal。
GRPO ratio 以 frame joint probability 为 action 单位；kind-conditioned 参数是同一动作的
概率分解，不是额外环境 step。

## 3. Checkpoint identity

checkpoint format v6 锁定 schema version、action schema ID、32x32 coordinate grid、每 unit
16 TYPE bytes、最多 8 HOTKEY keys 和 key table identity。旧 flat Action Head checkpoint 直接
拒绝 resume/warm-start。

## 4. 验证设计

- v6 episode 写入/读取逐字段往返；
- 缺失监督与有监督 NO_ACTION 明确区分；
- kind-conditioned 参数边界及非法冗余参数拒绝；
- TYPE 跨 unit UTF-8 pending 重放，非 TYPE 切换时不完整序列拒绝；
- flat arrays、错误 action schema ID 和任何旧 schema fail closed；
- audit metadata 不进入模型输入；
- checkpoint/data/runtime identity 完全一致。
