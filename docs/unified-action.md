# 统一电脑动作输出协议

> 状态：最终目标 Unified Action Head 协议
> 日期：2026-08-08
> 关联顶层架构：[实时流多模态 LatentLoop 完整方案](realtime-multimodal-latent-loop.md)
> 对称语音协议：[直接流式语音实施说明](direct-speech.md)

## 1. 完成边界

Action token 是模型内部和训练数据中的统一离散表示。Model Service 在输出边界将
token 解码为 `ControlSignal`（鼠标、键盘、文本、滚动、等待、取消等物理控制事件）；
Harness 只接收 `ControlSignal`，然后执行参数、screen revision、安全策略和权限校验。
Harness 不依赖 Model Core，也不接收 raw action token 作为执行接口。

Unified Action Head 是实时多模态模型的独立电脑操控输出头。它与 Speech Head 共享 `H_t`，但使用独立 vocabulary、局部 decoder state 和 loss：

~~~text
E_t       = InputEncoder(U_t)
Z_t       = MemoryUpdater(Z_(t-1), H_(t-1))
H_t, KV_t = Backbone(E_t, KV_(t-1), Z_t)
action_t  = ActionHead(H_t, action_local_(t-1))
~~~

“统一空间”只指所有电脑操控 action 共用一个离散 token 序列空间，不表示语音与 action 共用输出空间。项目运行时恰好有两个模型输出头：

~~~text
Speech Head -> speech mode + Mimi codec tokens
Action Head -> unified computer-action tokens
~~~

Action Head 只产生动作意图序列，不直接调用操作系统。ActionTokenizer 负责可逆编码和 grammar；外部 Harness 负责权限、安全、screen revision、调度和真实执行。

## 2. 统一词表

### 2.1 Token 分区

当前协议使用一个版本化词表：

| 分区 | 数量 | 语义 |
|---|---:|---|
| 特殊/动作类型 token | 12 | `PAD`、`END_ACTION` 和 10 种 action kind |
| coordinate bins | 256 | 归一化屏幕坐标 `[0, 1]` |
| scroll bins | 256 | 有符号滚动量 `[-1, 1]` |
| duration bins | 128 | `WAIT` 时长 `[0, max_action_duration_ms]` |
| UTF-8 byte tokens | 256 | `TYPE` 的原始字节 |
| key tokens | 32 | `HOTKEY` 的版本化按键表 |

词表布局为：

~~~text
0   PAD
1   END_ACTION
2   NOOP
3   CLICK
4   DOUBLE_CLICK
5   RIGHT_CLICK
6   DRAG
7   SCROLL
8   TYPE
9   HOTKEY
10  WAIT
11  CANCEL

12..267    coordinate bins
268..523   scroll bins
524..651   duration bins
652..907   UTF-8 byte tokens
908..939   key tokens
~~~

因此当前 `vocab_size = 940`。action vocabulary 的版本、bin 数量、key table、最大时长和 burst 长度必须写入配置、数据 manifest 与 checkpoint identity；任何改变都属于协议版本变更。

### 2.2 为什么使用一个序列空间

action kind 与参数都离散化成 token 后，点击、拖拽、滚动、输入和快捷键可以使用同一个自回归 decoder 与一个 masked token CE。这样不需要为 type、coordinate、duration、text、confidence 分别建立 head，也避免不同结构化 head 之间的组合和 mask 语义分裂。

统一不等于无 grammar。每种 kind 允许的后续 token 类型和长度仍由确定性状态机约束；统一空间提供共同的生成接口，grammar 提供合法序列边界。

## 3. Action grammar

每个完整事件以 action kind 开始，以 `END_ACTION` 结束：

~~~text
NOOP         END_ACTION
CANCEL       END_ACTION
CLICK        X Y END_ACTION
DOUBLE_CLICK X Y END_ACTION
RIGHT_CLICK  X Y END_ACTION
DRAG         X1 Y1 X2 Y2 END_ACTION
SCROLL       DX DY END_ACTION
WAIT         DURATION END_ACTION
TYPE         BYTE* END_ACTION
HOTKEY       KEY+ END_ACTION
~~~

约束如下：

- `X/Y/X1/Y1/X2/Y2` 必须来自 coordinate 分区；
- `DX/DY` 必须来自 scroll 分区；
- `DURATION` 必须来自 duration 分区；
- `BYTE*` 只允许 UTF-8 byte token，空文本由任务协议决定是否允许；
- `KEY+` 至少包含一个 key token；
- `NOOP` 与 `CANCEL` 不接受参数；
- 一个 event 只能有一个终止 `END_ACTION`；
- `PAD` 不是动作内容，只用于定长 burst 的无效尾部。

训练导入、推理 decode 和 Harness 执行前都必须使用同一 grammar 校验器，不能分别实现含义不同的宽松解析。

## 4. 参数量化与还原

### 4.1 坐标

屏幕坐标先归一化到 `[0, 1]`，再映射到 256 个 bin：

~~~text
coord_bin = round(value * 255)
value_hat = coord_bin / 255
~~~

坐标 token 不绑定某个物理分辨率。执行时 Harness 使用事件关联的目标 screen revision 和当前显示区域恢复像素位置；过期 revision 不允许直接套用到新屏幕。

### 4.2 滚动

滚动分量限制在 `[-1, 1]`：

~~~text
scroll_bin = round((value + 1) * 0.5 * 255)
value_hat  = scroll_bin / 127.5 - 1
~~~

具体滚轮 tick、触控板距离或页面位移由 Harness 的版本化执行适配器映射，模型协议只表达归一化意图。

### 4.3 时长

`WAIT` 在 `[0, max_action_duration_ms]` 内量化到 128 bins：

~~~text
duration_bin = round(duration_ms / max_duration_ms * 127)
duration_hat = duration_bin / 127 * max_duration_ms
~~~

超出配置范围的输入在编码时拒绝，推理时也不能通过 clamp 静默改变意图。

### 4.4 文本与快捷键

`TYPE` 使用 UTF-8 bytes，因而不依赖自然语言 tokenizer，能够表示任意合法 Unicode 文本。decode 必须验证字节序列可构成 UTF-8；无效序列不得传入操作系统。

`HOTKEY` 使用固定 32-key table。key token 只表达协议中的逻辑键，不直接等于平台 scan code；平台映射由 Harness 管理并纳入版本校验。

## 5. Action Head

### 5.1 Decoder 输入输出

Action Head 读取 `H_t` 的当前输出位置和上一 action local state：

~~~text
context_t = Linear(last_position(H_t))

repeat action_burst_tokens times:
    decoder_hidden = Decoder(previous_token, context_t, decoder_hidden)
    action_logits  = VocabularyProjection(decoder_hidden)
    token          = teacher target or sampled token
~~~

当前实现使用 token embedding、context projection、GRUCell 和词表线性层。这里的 decoder local state 只负责一个未结束 action 的短期序列连续性，不承担任务规划或长期记忆。

### 5.2 独立输出空间

Action Head 不读取语音 token，Speech Head 也不读取 action token。二者通过共享 Backbone 间接协同：

~~~text
L_speech -> Backbone parameters
L_action -> Backbone parameters

speech local 与 action local 彼此独立
speech vocabulary 与 action vocabulary 彼此独立
~~~

一个 80 ms unit 可以同时说话和产生电脑动作，两条输出不需要互斥 control head。

### 5.3 NOOP、WAIT 与静默

- 没有需要执行的电脑动作时，数据可以用无效 action mask 表达“本 unit 无 action target”；
- `NOOP` 是显式动作事件，适合任务确实要求模型确认不操作的场景；
- `WAIT` 是带时长的动作意图，用于序列中明确需要等待的任务；
- 语音静默由 Speech Head 的 `SILENCE` 表达，与 Action Head 无关。

不能用 action `NOOP` 代替语音静默，也不能用 Speech `SILENCE` 代替电脑动作缺省。

## 6. 跨 unit continuation

每个 80 ms unit 最多生成 `action_burst_tokens` 个 token，当前默认值为 16。若完整事件超过一个 burst，`action_local` 把未结束事件延续到下一 unit：

~~~text
ActionLocalState = {
    hidden,
    previous_token,
    active,
    event_type,
    burst_tokens,
}
~~~

- `hidden` 保存 decoder 的连续状态；
- `previous_token` 是下一位置的自回归输入；
- `active` 表示事件尚未以 `END_ACTION` 结束；
- `event_type` 保存当前 kind，供 continuation 与校验使用；
- `burst_tokens` 记录当前事件已经生成的 token 数。

事件跨 unit 时不能重复输出 kind，也不能在 unit 边界自动补 `END_ACTION`。收到 `END_ACTION` 后 local event 结束，固定 burst 的剩余位置为 `PAD` 且 loss mask 为 false。episode/session reset、明确 `CANCEL` 或运行时恢复策略才能终止未完成事件。

长 `TYPE` 文本天然可能跨越多个 unit。Harness 必须等到完整 grammar 事件结束并通过安全校验后再原子执行，不能逐 byte 边生成边注入键盘。

## 7. 数据契约

### 7.1 Unit targets

Schema v3 在每个 unit 保存：

~~~text
action_tokens      [B, action_burst_tokens]
action_token_mask  [B, action_burst_tokens]
screen_revision    [B]
~~~

`action_token_mask=true` 的位置参与 loss；`PAD` 和不存在 action target 的位置为 false。跨 unit 事件必须在完整 episode 时间线上连续编码，并保留同一事件与 screen revision 的关联。

### 7.2 轨迹来源

训练 action 轨迹应来自可回放的真实或 sandbox 环境任务，至少记录：

~~~text
timestamp
screen revision before action
action event and token sequence
execution decision
environment observation after action
episode/session identity
~~~

模型输入只能看到当时可观测的屏幕、混合音频和时间；未来结果、执行成功标签、隐藏 DOM、特权坐标或教师内部计划不能作为输入旁路。执行后的真实屏幕和声音在后续 unit 进入模型，由后续 action/speech loss 教会模型利用结果。

### 7.3 数据校验

导入训练前拒绝：

- 非法 kind/参数组合或缺少 `END_ACTION`；
- 超出 token 分区的参数；
- 无效 UTF-8 或未知 key token；
- 坐标、滚动、时长超出协议范围；
- continuation 在 unit 边界丢失或重复 kind；
- `PAD` 位置被标为有效 target；
- screen revision 缺失、倒退或与事件不一致；
- 同一 session 跨 train/validation/test 泄漏；
- action vocabulary identity 与配置不一致。

## 8. 训练目标与梯度

Action Head 只有一个 masked token cross entropy：

~~~text
L_action = masked_CE(action_logits, action_tokens)
~~~

action kind、坐标、滚动、时长、UTF-8 byte、key 和 `END_ACTION` 都在同一个 loss 中按有效 token 归一化。没有独立 type loss、coordinate regression、duration regression、confidence、action-control 或 grammar auxiliary loss。

项目总目标为：

~~~text
L_total = speech_loss_weight * L_speech
        + action_loss_weight * L_action
~~~

梯度影响：

| 模块 | Action loss 的影响 |
|---|---|
| Action Head | 直接学习 token grammar、参数与 continuation |
| action local decoder | 直接学习跨位置和跨 unit 的事件连续性 |
| Backbone | 学习产生支持电脑操控的共享多模态表示 |
| InputEncoder | 学习从屏幕、混合音频和时间提取动作所需条件 |
| MemoryUpdater | 通过未来 action loss 学习保留长期目标、约束和早期执行结果 |
| Speech Head | 不接收 Action loss 的直接参数梯度；只通过共享 Backbone 的联合训练间接受影响 |

长期监督路径为：

~~~text
future action token loss
 -> future Action Head
 -> future H
 -> future Z
 -> earlier MemoryUpdater
~~~

真正的长任务、动作后果利用和错误恢复应由轨迹数据与后续 action token loss 提供，不增加 memory probe 或 action-success head。

## 9. 推理与 Harness 安全边界

模型输出不是执行授权。Harness 在提交操作系统之前必须执行：

1. 按 vocabulary identity 解码并验证完整 grammar；
2. 检查 event 是否已完整结束，拒绝半个 continuation；
3. 验证 screen revision 新鲜度和目标屏幕/窗口一致性；
4. 检查坐标、滚动、时长、文本长度和速率上限；
5. 应用程序、窗口、区域和动作类型白名单；
6. 对删除、支付、发送、安装、权限修改等高风险操作请求批准；
7. 支持队列超时、`CANCEL`、session reset 和全局紧急停止；
8. 记录原始 token、解码事件、审批、执行结果和时间戳。

Harness 拒绝或执行 action 后，不向模型注入隐藏成功标志。真实环境变化通过后续屏幕、音频和时间输入返回主干，保证训练与推理都依赖可观察闭环。

## 10. Checkpoint 与恢复

Checkpoint 必须保存：

~~~text
Action Head parameters
action_local state
action vocabulary identity
coordinate/scroll/duration bin counts
key table identity
max_action_duration_ms
action_burst_tokens
unit cursor and session identity
~~~

从 checkpoint 恢复未结束事件后，下一 token logits、mask、local state 和最终解码事件应与不中断连续运行一致。任何 vocabulary、burst、key table 或量化参数不兼容都必须拒绝恢复，不能只加载形状碰巧一致的权重。

## 11. 验证设计

### 11.1 Tokenizer 与 grammar

- 每种 ActionEvent encode/decode 往返；
- 坐标、滚动和时长边界与量化误差；
- 中文、英文、emoji 等 UTF-8 TYPE 往返；
- HOTKEY 空参数、未知 key 和上界拒绝；
- 错误参数分区、缺失 `END_ACTION` 和多余 body 拒绝；
- vocabulary identity 与配置一致。

### 11.2 Head 与 continuation

- teacher forcing 与自由生成 shape/mask；
- `END_ACTION` 后 PAD 被 mask；
- 超过 16 token 的 TYPE 跨 unit 连续；
- checkpoint 中断恢复与连续生成等价；
- session reset 清除未结束 local event；
- Action loss 能到达 Action Head、Backbone 和 MemoryUpdater；
- action mask 全 false 时 loss 有限且不产生虚假监督。

### 11.3 Harness

- 过期 screen revision 被拒绝；
- 越界坐标、超长等待、非法文本和不允许按键被拒绝；
- 高风险操作进入审批而非直接执行；
- 半个 event 不执行，完整 event 原子提交；
- 速率限制、超时、取消和紧急停止有效；
- 执行日志可关联模型 checkpoint、session、unit 和原始 token。

### 11.4 行为评测

- action token accuracy 与完整 event exact match；
- grammar validity 和可解码率；
- 坐标量化后命中率、拖拽/滚动/输入成功率；
- 跨 unit continuation 完成率；
- screen revision 拒绝率和误执行率；
- 长任务中早期目标与动作结果的后续利用；
- Action Head 与 Speech Head 同时输出时的质量和延迟。

## 12. 结论

Unified Action Head 把所有电脑操控表达为一个版本化离散 token 序列空间：

~~~text
H_t + action_local_(t-1)
 -> action token burst
 -> cross-unit continuation
 -> complete grammar event
 -> Harness safety/revision/permission checks
 -> operating-system execution
~~~

它与 Speech Head 是并列而独立的第二个输出头。Action Head 只使用一个 masked token loss；类型、参数和终止符通过共同 vocabulary 与 grammar 学习。长期任务信息不由额外 control 或 memory loss 监督，而由未来真实 action token loss 通过 `Z_t` 反向塑形。模型负责提出动作，Harness 始终负责是否允许及如何安全执行。
