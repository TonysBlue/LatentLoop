# 统一三阶段训练架构

> 状态：最终目标训练契约
> 日期：2026-08-08
> 关联文档：[实时流多模态 LatentLoop](realtime-multimodal-latent-loop.md) · [在线 GRPO 与隔离环境](online-grpo-training.md)

## 1. 总体定义

LatentLoop 的正式训练顺序固定为：

```text
Pretrain -> SFT -> Online GRPO
```

Canary、Pilot、Production 都完整执行相同的三个阶段。它们使用同一模型语义、
同一 `training.py::train`、同一 `recipe.py::run_recipe`、同一环境协议、同一
reward 公式和同一 checkpoint 格式。规模之间只改变数据数量、更新数、GRPO
group size、环境 worker 数和计算资源，不改变训练目标。

模型始终只有两个输出头：Speech Head 直接输出 speech mode 与 Mimi codec token；
Unified Action Head 在一个统一 token vocabulary 中输出全部电脑操作。三个阶段均不
增加文本 head、value head、memory head 或阶段专用 head。

## 2. 三类独立数据

三个阶段的数据不能混同：

| 阶段 | 数据来源 | 主要用途 |
|---|---|---|
| Pretrain | 广覆盖多模态监督 episode | 学习音频、屏幕、时间、语言、动作和状态转移的通用规律 |
| SFT | 独立的高质量专家轨迹 | 学习可靠交互、语音响应和电脑操作策略 |
| Online GRPO | 当前 policy 在真实隔离环境中的在线 rollout | 按任务结果、交互质量、延迟、效率和安全反馈优化策略 |

Pretrain/SFT episode 按 80 ms unit 保存。没有专家 action 的 unit 必须设置
`action_token_mask=false`；不得把“无 action 标签”伪造成 `NOOP + END_ACTION`。
同理，缺少 speech 标签时必须关闭相应 speech mask。mask 表示监督是否存在，而非
模型在该时刻应该做什么。

## 3. Pretrain

Pretrain 使用 teacher forcing 的 masked token loss：

```text
L_pretrain = speech_weight * (L_speech_mode + L_speech_codec)
           + action_weight * L_action_token
```

各项只在自己的有效 mask 上归一化。Speech Head、Unified Action Head、
InputEncoder、Backbone 和 MemoryUpdater 全部更新。MemoryUpdater 没有独立 target；
未来 speech/action loss 通过 `Z_t -> Backbone -> H_t -> heads` 反向监督记忆更新。

## 4. SFT

SFT 与 Pretrain 使用相同的模型 forward 和损失形式，但输入是独立、审核过的专家
交互轨迹。SFT 不是“只训练 head”的适配阶段，全模型继续更新。SFT 最终 checkpoint
同时成为 Online GRPO 的初始 policy 与冻结 reference policy。

Pretrain checkpoint 只能作为 SFT 的父 checkpoint；SFT 数据身份、stage、objective
和父 checkpoint hash 都进入谱系。SFT 不读取 Pretrain 数据作为隐式回退。

## 5. Online GRPO

Online GRPO 从同一个任务初始状态和 seed 采样 G 条独立 rollout。环境观察只包含
模型运行时真实可见的混合麦克风、屏幕、screen revision 和时间，不包含任务成功、
隐藏执行结果或 reward 字段。环境在 rollout 结束后单独给出 reward breakdown。

组内 reward 标准化为 advantage，零方差组不产生 optimizer update。训练采用 clipped
policy ratio，并相对冻结的 SFT reference policy 计算 sampled-token KL。没有 critic
或 value head。详细数学定义和环境接口见 `online-grpo-training.md`。

## 6. Schema v5

监督 episode 与在线 rollout 统一使用 schema v5 identity。监督样本 metadata 至少包含：

```text
schema_version = 5
stage
dataset_scale
sample_kind
supervision_kind
action_source
task_id
environment_id
environment_version
protocol_version
action_vocabulary_id
runtime_identity
decoded_controls
receipts
```

在线 rollout 还必须记录 group/rollout ID、policy/reference hash、采样 token、old
log-prob、reward components、环境 event/receipt、task ID、seed 和 termination reason。
schema v3 不得静默兼容；旧资产必须显式重新生成。

## 7. Checkpoint v5 与阶段谱系

正式 checkpoint 使用 format v5，metadata 至少包含：

```text
schema_version
stage
objective
data_identity
codec identity
action_vocabulary_id
parent_sha256
reference_checkpoint_sha256
environment_id
task_manifest_sha256
reward_spec_id
```

Pretrain checkpoint 的 parent 可以为空；SFT 的 parent 必须是 Pretrain；GRPO 的 parent
必须沿训练更新链前进，同时 reference hash 始终指向冻结的 SFT checkpoint。format v4
只允许显式 warm-start 兼容权重，不能作为 v5 训练的 resume checkpoint。

## 8. 三种规模配置

| 参数 | Canary | Pilot | Production |
|---|---:|---:|---:|
| Pretrain updates | 1,000 | 50,000 | 100,000 |
| SFT updates | 600 | 30,000 | 60,000 |
| GRPO updates | 400 | 20,000 | 40,000 |
| GRPO group size | 4 | 8 | 16 |
| 环境并发 | 1 | 4 | 8 |
| memory / rollout horizon | 750 units | 750 units | 750 units |

Canary 是完整训练链的小规模证明，不是删减版算法。它同样使用真实隔离电脑环境和
在线 rollout；只减少任务数、group size、worker 和 update 数。

## 9. 测试契约

实现必须由以下测试保护：

- 配置拒绝错误 stage/objective、正式 RL 缺失环境 identity/socket、非法 GRPO 参数；
- 三个正式 recipe 都严格包含 `pretrain -> sft -> rl`，且全部 `backbone_train_mode=all`；
- schema v5 往返保存 mask、runtime identity、decoded controls、receipts 与 metadata，明确拒绝旧 schema；
- speech-only 导入保持 action mask 全 false，显式专家动作可正确编码；
- 环境客户端校验 identity，并保证 observation 不携带 reward/隐藏状态；
- rollout 的同组成员使用相同 task/seed 初始状态并记录 old/reference log-prob；
- GRPO advantage、clipping、KL、零方差跳过和全模型梯度可验证；
- format v5 resume 校验完整谱系，format v4 只能 warm-start；
- Canary、Pilot、Production 通过同一 recipe 和 train 分派路径。
