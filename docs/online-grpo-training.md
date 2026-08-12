# Online GRPO 与真实隔离电脑环境

> 状态：最终目标强化学习与环境协议
> 日期：2026-08-08
> 关联文档：[统一三阶段训练架构](three-stage-training.md) · [统一电脑动作输出协议](unified-action.md) · [物理 Rollout 闭环](protocols/physical-rollout.md)

## 1. 环境选择

Canary、Pilot、Production 全部使用同一种真实隔离电脑环境。环境可以由容器、虚拟机
或受控远程桌面承载，但必须实现同一版本化协议、从可复现 snapshot reset、执行真实
动作并返回真实屏幕/混合音频观察。Canary 不使用 fixture 或离线 rollout 替代环境，
只把并发降为 1、减少任务和 rollout 数。

进程内 deterministic environment 只用于单元测试协议和梯度，不允许被正式配置选择，
也不能在真实环境连接失败时自动回退。

## 2. 环境协议

Harness control client 提供以下语义：

```text
identity() -> EnvironmentIdentity
reset(task_id, seed, session_id) -> ObservationSignal
apply(session_id, unit_index, ActuationSignal)
    -> ObservationSignal + EnvironmentReceipt
evaluate(task_id, session_id) -> RewardBreakdown
close(session_id)
```

`EnvironmentIdentity` 包含 environment ID、version、protocol version 和 action schema
identity。连接、reset 和每次 apply 都必须校验 session/task/unit 顺序。

模型可见的 `ObservationSignal` 仅包含：

```text
timestamp_ms
delta_ms
mixed_microphone[1920]
screen[3,224,224]
```

每个 unit 都携带一帧完整屏幕；采集缺帧表示为黑帧，不使用 valid 或 revision 分支。

是否继续 rollout 由 Harness receipt 的 `terminated` 控制字段表示；它不属于模型物理
输入，也不携带任务成功原因。

`terminated` 只表示 episode 是否结束，不透露成功原因。task success、评分器内部状态、
权限判断和隐藏 UI 元数据只能进入训练侧 receipt/reward 记录，不能拼进下一 unit 输入。

## 3. Rollout 组

每个 GRPO group 选择一个 task ID 和 seed。环境必须为组内 G 个成员分别恢复完全相同的
初始 snapshot。每个成员由当前 policy 独立采样 Speech Head 和 Unified Action Head，
直到 environment terminated 或达到 `rollout_horizon_units`。

每个 unit 的 sampled 输出保存：

```text
speech token/mask/log-prob
structured action frame
action supervision mask
action frame joint old/reference log-prob
unit_index
probability component audit
```

Speech mode、speech codec 和 action frame 分别按有效 mask 统计，再组合成策略目标。
SILENCE unit 只有 speech mode log-prob，不虚构 codec token。每个 action frame 是一个
环境 step 的联合概率单位；TYPE bytes 仍属于该 frame 参数，不能重复计成多个环境 step。

## 4. Reward

固定 reward spec 为：

```text
R_total = R_task + 0.2 * R_interaction + R_safety

R_interaction = 0.4 * speech_quality
              + 0.3 * latency_quality
              + 0.3 * action_efficiency
```

- `R_task`：由任务验证器基于最终环境状态计算；
- `speech_quality`：可懂度、内容正确性、打断处理和不必要发言；
- `latency_quality`：从观察到首个有效响应/动作的时延；
- `action_efficiency`：完成任务所需动作数、无效重复和路径长度；
- `R_safety`：非法 grammar、越权、危险操作和安全策略拒绝的惩罚。

reward breakdown 必须保存所有分量和 `reward_spec_id`，不能只保存总分。

## 5. GRPO 数学目标

对同组总 reward `R_i`：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

若组内 reward 方差低于 `advantage_epsilon`，整组跳过，不用数值噪声制造梯度。

对 rollout 中 sampled speech item 或 action frame t：

```text
rho_i,t = exp(log pi_theta(a_i,t|s_i,t) - log pi_old(a_i,t|s_i,t))

L_policy = -mean(min(
    rho_i,t * A_i,
    clip(rho_i,t, 1-epsilon, 1+epsilon) * A_i
))
```

冻结 SFT reference policy 的 sampled-token KL 使用稳定的非负近似：

```text
log_ratio_ref = log pi_ref - log pi_theta
KL_sample = exp(log_ratio_ref) - log_ratio_ref - 1

L_grpo = L_policy + beta * mean(KL_sample)
```

策略比率使用 rollout 时保存的 old log-prob；reference policy 在整个 RL stage 固定。
Speech mode、codec、action frame 三类概率的 loss 分别 mask-normalize 后等权组合，避免
TYPE byte 数量较多时自动支配梯度。梯度穿过两个 head、Backbone、InputEncoder 和
MemoryUpdater，不存在 value loss 或单独 memory loss。

## 6. 在线性与安全边界

- rollout 必须由当前 policy 在线产生，训练不得从静态文件冒充 on-policy 数据；
- policy update 后未消费的旧 rollout 必须丢弃；
- 每个 action frame 在当前 unit 解码后由 Harness 验证 schema、权限和危险操作；
- 环境超时、崩溃、身份变化或 receipt 不完整时，该 rollout 标为 infrastructure failure，
  不当作低 reward 样本训练；
- 环境 socket 和 snapshot 位于仓库外，连接失败必须 fail closed；
- policy、reference、task manifest、环境和 reward spec identity 全部进入 checkpoint。

## 7. 规模参数

三种规模使用同一协议：Canary `G=4/workers=1`，Pilot `G=8/workers=4`，Production
`G=16/workers=8`。每条 rollout horizon 都是 750 units。worker 并发只影响吞吐，
不得改变同组初始状态一致性、reward、advantage 或优化目标。
