# 当前轨迹契约

监督 episode、SFT 样本和 Online GRPO rollout 统一使用当前轨迹契约。项目只维护这一份
数据契约，不维护 schema 编号或历史迁移路径。

每个 `meta.json` 和 manifest entry 必须包含 `dataset_scale`、
`sample_kind`、`stage`、`supervision_kind`、`task_id`、`environment_id/version`、
`protocol_version`、`action_schema_id=structured-action-v1`、codec identity 和完整
`runtime_identity`。每个 unit 固定 80 ms、24 kHz 单声道、每 unit 一帧 Mimi。

timeline 按 unit 保存：

```text
timestamp_ms, delta_ms, dense screen frame
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

数据准备从 source manifest 直接构建当前 staging/processed shards，经 Mimi worker 和
structured-action audit 后生成当前 manifest；不提供历史 schema 或 flat-token 转换兼容分支。

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
    speech_mode: Tensor
    speech_mode_mask: Tensor
    speech_codes: Tensor
    speech_codec_mask: Tensor
    action: ActionFrame
    action_supervision_mask: Tensor
```

`delta_ms` 为正，时间戳严格递增；每个 unit 必须有 `[3,224,224]` 屏幕张量。采集缺失时
输入适配器提供全黑帧并记录控制面统计；模型输入不包含 revision 或 valid 标记。reader 必须
校验 kind 条件参数、TYPE/HOTKEY length、数值边界和 episode 时间顺序。

## 2. Online rollout

每个 rollout unit 记录 sampled ActionFrame、frame joint old/reference log-prob、解码后的
有序 controls、receipt 和 reward lineage。receipt/reward 不进入 ObservationSignal。
GRPO ratio 以 frame joint probability 为 action 单位；kind-conditioned 参数是同一动作的
概率分解，不是额外环境 step。

## 3. Checkpoint identity

checkpoint 只保存当前模型、数据、codec、action 和阶段谱系身份，不写入 format/schema 编号。
32x32 coordinate grid、每 unit 16 TYPE bytes、最多 8 HOTKEY keys 和 key table identity 仍是
当前 Action 契约常量。

## 4. 验证设计

- 当前 episode 写入/读取逐字段往返；
- 缺失监督与有监督 NO_ACTION 明确区分；
- kind-conditioned 参数边界及非法冗余参数拒绝；
- TYPE 跨 unit UTF-8 pending 重放，非 TYPE 切换时不完整序列拒绝；
- flat arrays、错误 action schema ID 和不完整当前 metadata fail closed；
- audit metadata 不进入模型输入；
- checkpoint/data/runtime identity 完全一致。
