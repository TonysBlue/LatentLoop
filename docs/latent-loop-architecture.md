# LatentLoop：双通道连续状态自回归语言模型方案

> 状态：最终目标纯文本输出方案
> 日期：2026-08-08
> 适用范围：以文本 token 为输入和输出的 LatentLoop 模型思想、状态转移、训练监督和可验证假设。实时语音与电脑操控项目的实现契约见 [实时流多模态 LatentLoop 完整方案](realtime-multimodal-latent-loop.md)。

## 1. 摘要

LatentLoop 是一种双通道、双时间尺度的连续状态自回归语言模型。模型一方面沿标准 token/KV 通道生成文本，并把实际采用的 token 作为后续输入；另一方面维护固定容量的连续 latent memory `Z_t`，用来压缩可能跨越 KV 窗口仍需使用的目标、约束、计划和较早事实。

核心状态转移为：

~~~text
E_t       = TokenEncoder(X_t)
Z_t       = MemoryUpdater(Z_(t-1), H_(t-1))
H_t, KV_t = Backbone(E_t, KV_(t-1), Z_t)
logits_t  = LMHead(H_t)
y_t       = Decode(logits_t)
~~~

其中 `H_t` 是当前 token block 经过 final normalization 后的完整 hidden 序列，必须暂存到下一步；不存在额外的 `r_t`、query summary 或 control state。`Z_t` 没有独立记忆标签和辅助 loss，只通过当前及未来文本 token 的 next-token loss 学习写入、保持、增强和遗忘。

本文中的“纯文本输出”指模型外部输出空间只有文本 token。本文不定义其他模态输出、声学 codec、屏幕输入或电脑执行 Harness；这些属于项目顶层多模态架构。

## 2. 设计目标与非目标

### 2.1 设计目标

1. 保持标准文本自回归语义：未来条件于实际 token，而不是未被采用的概率分支。
2. 用有界 KV 保存近期精确 token 历史，用固定容量 `Z_t` 保存长期压缩状态。
3. 让完整 `H_(t-1)` 驱动下一次记忆更新，不额外构造有损摘要状态。
4. 让长期记忆仅由真实文本输出任务的未来 loss 塑形。
5. 保持训练、验证、推理和状态恢复使用同一时间索引。
6. 使内存开销不随全部会话历史无限增长。

### 2.2 非目标

- `Z_t` 不是可读的自然语言思维链，也不要求 slot 具有人类预设语义。
- `Z_t` 不替代近期 KV，也不承诺无损压缩任意长度历史。
- 不引入 `THINK / SPEAK / STOP` 控制器；文本何时结束由普通终止 token 表达。
- 不引入 future embedding head、memory probe、recall head、write-budget、slot diversity 或 control loss。
- 不在本文描述语音、电脑 action、多模态时间轴、数据 schema 或运行时工程接口。
- 不描述实现阶段、训练阶段、里程碑或过渡方案。

## 3. 核心概念

设第 `t` 个文本计算单元包含以下变量：

| 符号 | 含义 |
|---|---|
| `X_t` | 当前输入的文本 token 或 token block |
| `E_t` | `TokenEncoder(X_t)` 的 token embedding 序列 |
| `y_t` | 当前实际采用的输出 token |
| `H_t` | 主干在当前单元输出的完整 final-normalized hidden 序列 |
| `KV_t` | 主干各层的有界 Key/Value Cache |
| `Z_t` | 固定数量的 latent memory slots |
| `logits_t` | LM Head 对文本词表的预测 logits |

模型内部存在三个不同角色的状态：

~~~text
KV_t：近期文本历史的逐层精确表示
Z_t ：较长期历史和任务状态的固定容量压缩表示
H_t ：当前单元读取 E_t、KV_(t-1)、Z_t 后得到的完整工作状态
~~~

### 3.1 `H_t`：当前完整工作状态

若一个单元包含 `S_t` 个 token，则：

~~~text
H_t.shape = [B, S_t, d_model]
~~~

`H_t` 综合当前 token block、近期 KV 和长期 `Z_t`，并有两个去向：

~~~text
H_t -> LM Head -> 文本 token logits
H_t -> 暂存 -> 下一步 MemoryUpdater
~~~

不能只保存最后一个 query、平均池化向量或另一个 `r_t`。这些摘要会提前限制 MemoryUpdater 可读取的信息，并产生与核心方案不同的状态协议。

### 3.2 `Z_t`：固定容量 latent memory

~~~text
Z_t.shape = [B, M, d_z]
~~~

`M` 是固定 slot 数量，容量不随文本长度增长。多个 slot 允许更新器和主干通过 attention 选择性读写不同连续表示；训练可能形成目标、约束、实体事实和未完成事项等分工，但不显式规定 slot 语义。

`Z_t` 保存的是经过学习压缩、对未来 token 预测有用的信息，而不是历史 token 的精确副本。信息是否值得保留，最终由它对未来文本 loss 的贡献决定。

### 3.3 `KV_t`：近期精确历史

每层 Key/Value Cache 的典型形状为：

~~~text
K_t^(l), V_t^(l): [B, H_kv, S, D]
~~~

KV 保留历史 token 的逐层注意力表示，适合精确措辞、局部语法和近距离指代，但存储量随缓存长度增长。因此 LatentLoop 使用有界滑动 KV，并让窗口外仍有价值的信息只能通过 `Z_t` 继续影响输出。

### 3.4 三种状态的区别

| 状态 | 容量 | 时间尺度 | 主要职责 |
|---|---:|---|---|
| `H_t` | 一个 token block | 当前/跨一步 | 当前完整工作状态和下一次记忆更新来源 |
| `KV_t` | 有界窗口 | 近期 | 精确 token 历史、语法和局部引用 |
| `Z_t` | 固定 `M` slots | 长期 | 目标、约束、计划和窗口外事实的压缩表示 |

简单令 `Z_t = last(H_(t-1))` 并不等价。单个最后位置会随措辞快速变化，也无法在多个 slot 间学习不同信息的保持速度；独立 MemoryUpdater 提供了选择、融合与稳定递归的能力。

## 4. 整体架构

~~~text
当前文本 token X_t
        |
        v
TokenEncoder -> E_t -------------------------+
                                                 |
Z_(t-1) + H_(t-1) -> MemoryUpdater -> Z_t       |
                                      |          |
KV_(t-1) -----------------------------+----------+
                                                 v
                                             Backbone
                                              /      \
                                            H_t      KV_t
                                             |
                                             v
                                          LM Head
                                             |
                                             v
                                    文本词表 logits_t
                                             |
                                             v
                                      实际 token y_t
                                             |
                                      下一步 TokenEncoder
~~~

这里有两条互补反馈通道：

1. token/KV 通道反馈模型实际读到和输出的离散文本，保持生成轨迹与采样结果一致；
2. latent 通道反馈由完整 hidden 压缩出的连续长期状态，避免所有长期信息都必须保存在增长的 KV 中。

`Z_t` 通过主干中的 latent cross-attention 或等价 gated adapter 被读取，但不直接拼入普通 KV 序列。这样 KV 淘汰与 latent 更新保持独立，二者承担不同时间尺度。

## 5. 单步状态转移

### 5.1 初始化

在新 session 边界：

~~~text
Z_0  = zeros + learned slot identity
H_0  = zeros
KV_0 = empty
~~~

learned slot identity 用于打破零初始化 slot 的对称性，不规定 slot 的人类可解释语义。若业务需要跨 session 持久记忆，必须把持久化、授权和清除定义为独立外部策略，不能静默复用上一用户状态。

### 5.2 记忆先于当前主干

第 `t` 步先用上一状态计算 `Z_t`：

~~~text
Z_t = MemoryUpdater(Z_(t-1), H_(t-1))
~~~

再处理当前 token 输入：

~~~text
H_t, KV_t = Backbone(E_t, KV_(t-1), Z_t)
~~~

这个顺序表示 `H_(t-1)` 中形成的信息从第 `t` 步起进入长期记忆。不能先计算 `H_t`，再回头生成同一步已经读取的 `Z_t`，否则会形成时间索引环路。

### 5.3 文本 token 输出与自反馈

LM Head 读取需要预测的位置：

~~~text
logits_t = LMHead(H_t)
y_t      = Decode(logits_t)
~~~

实际采用的 `y_t` 作为下一 token 输入，因而模型仍满足标准自回归条件：

~~~text
p(y_(t+1) | y_(<=t), X_(<=t), Z_t)
~~~

训练时 teacher forcing 使用目标前缀；自由生成时使用模型实际采样结果。无论 token 来自标签、greedy、temperature 还是 top-p，下一步只能读取实际采用的 token，不能沿未采用的 logits 分支更新状态。

### 5.4 当前单元结束

~~~text
state_(t+1) = {
    latent: Z_t,
    hidden: H_t,
    layer_kv: KV_t,
    token_cursor: cursor_t,
}
~~~

下一步使用完整 `H_t` 更新 `Z_(t+1)`。`H_t` 只跨一步暂存，不是第二份长期 memory；窗口内精确历史仍由 KV 保存。

## 6. MemoryUpdater

### 6.1 输入边界

MemoryUpdater 的外部接口只有：

~~~text
MemoryUpdater(Z_(t-1), H_(t-1)) -> Z_t
~~~

它不额外接收当前 `E_t`、实际 token embedding、控制动作或人工 memory label。上一轮实际 token 已经进入主干并体现在 `H_(t-1)` 中，因此再次作为独立输入既非必要，也会让接口产生重复语义。

### 6.2 一种合法参数化

内部可以采用 attention 与门控残差：

~~~text
query     = LatentProjector(Z_(t-1)) + learned_slot_identity
context   = Attention(query, H_(t-1), H_(t-1))
candidate = Candidate(Z_(t-1), context) + learned_slot_identity_latent
gate      = sigmoid(Gate(Z_(t-1), context))
Z_t       = LayerNorm((1 - gate) * Z_(t-1) + gate * candidate)
~~~

候选分支负责提出新的压缩状态，gate 决定各 slot 保留或更新多少，残差路径和 normalization 负责递归稳定。具体内部网络可以变化，但不能改变 `Z_(t-1), H_(t-1) -> Z_t` 的外部时间语义。

### 6.3 写入、增强与自然遗忘

对未来输出重要的信息会反复帮助降低 next-token loss，因此更新器可以学习：

- 在信息不重要时让 gate 接近保持；
- 在出现新目标、约束或关键事实时写入候选；
- 在同一事实被再次确认时增强其对后续主干读取的影响；
- 在事实被更新时用新表示覆盖或修正旧表示；
- 在容量竞争中逐渐降低长期无用信息的相对影响。

“遗忘”不是把某个 slot 按固定时间清零，而是在有限容量和任务梯度下动态重分配表示。自然遗忘是否真的出现必须通过行为评测验证，不能仅凭 gate 数值宣称成立。

### 6.4 数值与递归稳定性

长期递归需要关注 `Z_t` 范数漂移、门饱和、slot 同质化和梯度衰减。结构上使用残差门控、normalization、learned slot identity 和梯度裁剪；训练诊断记录 gate、slot cosine、latent norm 及梯度范数，但这些监控量不是额外训练目标。

## 7. 文本输出空间

模型只有一个外部输出头：

~~~text
LMHead: H_t -> vocabulary logits
~~~

词表包含普通文本 token 与必要的结构/终止 token。所有输出都通过同一 next-token 预测空间生成；不为“回答”“停止”“记忆”“计划”或“思考”再建立独立 head。回答结束由 EOS 或协议中的终止 token 表达。

纯文本 LatentLoop 可以接收带 role/type 标识的用户、assistant、工具结果等文本事件，但这些仍编码为文本 token 序列。工具调用若需要结构化协议，也应作为词表中的序列语法处理；本文不定义真实电脑动作空间。

## 8. 缓存与长期记忆分工

### 8.1 混合记忆

推荐组合为：

| 组件 | 是否随完整历史增长 | 信息粒度 | 用途 |
|---|---:|---|---|
| 当前 token/block | 否 | 当前离散输入 | 推进文本自回归 |
| 有界 KV | 否 | 近期逐 token 精确记录 | 措辞、局部语法和近距离引用 |
| 固定 `Z_t` | 否 | 长期压缩连续状态 | 目标、约束、计划和窗口外事实 |
| 完整 `H_t` | 否 | 当前 block 的工作表示 | 驱动下一次 MemoryUpdater |

KV 必须有界，否则主干可以始终从完整历史直接取回信息，既失去固定内存优势，也使 `Z_t` 更容易被忽略。窗口长度、slot 数和更新频率是模型容量参数，应由质量、显存与吞吐共同决定，而不是在文档中绑定某个项目运行时数值。

### 8.2 跨轮连续性

当用户和 assistant 轮次都属于同一 session 时，`Z`、`H` 和 KV 按 token 时间顺序连续传递。role boundary 是输入序列的一部分，不触发隐式 reset。只有明确 session 边界、隐私清除或恢复策略才允许初始化状态。

### 8.3 固定长度不等于固定内容

`Z_t` 的张量形状固定，但内容每步都可能更新。它类似固定容量的连续 recurrent memory，而不是把原始 KV 压缩后永久封存的静态缓存。是否保留、覆盖或增强由 MemoryUpdater 结合旧 `Z` 和上一完整 `H` 决定。

## 9. 训练目标与监督路径

### 9.1 标准 next-token loss

训练目标只有 masked next-token cross entropy：

~~~text
L_text = - sum_(b,s) mask_(b,s) * log p(y*_(b,s) | prefix, Z) / sum mask
L_total = L_text
~~~

padding、输入提示中不要求预测的位置和 session 外位置由 mask 排除。普通 token、结构 token 和终止 token 使用同一个词表 loss。

### 9.2 模块的监督来源

| 模块 | 监督来源 | 主要作用 |
|---|---|---|
| LM Head | 当前 next-token loss | 文本词表预测 |
| Backbone | 当前及未来 next-token loss | token/KV/latent 的共享表示 |
| MemoryUpdater | 经过未来 `Z -> Backbone -> H -> LMHead` 的 next-token loss | 长期信息选择与更新 |
| `Z_t` | 无独立标签 | 固定容量长期连续状态 |
| KV | 无独立 loss、无可学习状态参数 | 近期精确上下文 |

长期监督路径为：

~~~text
future next-token loss
 -> future LMHead
 -> future H
 -> future Z
 -> earlier MemoryUpdater(Z_(t-1), H_(t-1))
~~~

因此 delayed recall、跨轮约束和事实更新应当作为真正的文本任务样本出现：远期正确答案位置仍计算普通 next-token loss，而不是增加 memory probe 或 recall head。

### 9.3 为什么不使用多个辅助 loss

给一个模块配置多个 loss 并非原则上错误，但只有当每个目标都对应必要且可验证的外部行为时才值得保留。本方案希望 MemoryUpdater 学到“对未来文本输出有用的状态”，因此 next-token loss 已经与最终目标一致。独立的 write budget 或 diversity 可能优化漂亮的内部统计，却不一定改善语言行为，并可能把手工偏好固化进 slot。

若未来实验发现单一 loss 无法形成可用长期记忆，应先检查数据是否真的要求窗口外信息、TBPTT 是否覆盖依赖距离、KV 是否泄漏答案以及梯度是否可达；不默认用辅助 head 掩盖这些问题。

### 9.4 跨时反向传播

训练必须按 session 内 token 顺序传递 `Z`、`H` 和 KV。TBPTT 边界可以 detach 计算图以控制显存，但不能清零数值状态。若需要学习距离为 `D` 的依赖，至少要让有效梯度跨度覆盖该依赖，或者使用能保持相同监督语义的梯度估计；仅前向携带状态而在答案前过早 detach，不能训练早期写入决策。

## 10. 训练与推理一致性

训练和推理共享同一递归式：

~~~text
Z_t       = MemoryUpdater(Z_(t-1), H_(t-1))
H_t, KV_t = Backbone(E_t, KV_(t-1), Z_t)
logits_t  = LMHead(H_t)
~~~

差别只在 token 来源：训练通常读取目标前缀，推理读取实际生成前缀。teacher forcing 不得改变 MemoryUpdater 的输入签名，也不得在训练时提供推理不可见的 memory 标签、未来文本或完整历史旁路。

为了评估 exposure bias，可以在独立实验中使用模型生成前缀或错误恢复样本，但状态仍按实际采用的 token 重新前向计算，不能把 soft logits 当作已经发生的 token。

## 11. 推理算法

~~~text
输入：文本事件 X、状态 (Z, H, KV)、最大输出预算

1. 将 X 编码为 token，并按顺序执行 prefill：
   a. Z_t = MemoryUpdater(Z_(t-1), H_(t-1))
   b. H_t, KV_t = Backbone(E_t, KV_(t-1), Z_t)
2. 从 LM Head 得到下一个 token 分布。
3. 选择实际 token y_t；输出该 token。
4. 将 y_t 作为下一步输入，继续相同状态转移。
5. 生成终止 token或达到外部安全预算时停止。
6. 保存需要延续同一 session 的 Z、H、KV 和 token cursor。
~~~

伪代码：

~~~python
def step(token_ids, state):
    embeddings = token_encoder(token_ids)
    latent = memory_updater(state.latent, state.hidden)
    hidden, layer_kv = backbone(
        embeddings,
        state.layer_kv,
        latent,
    )
    logits = lm_head(hidden)
    return logits, State(latent=latent, hidden=hidden, layer_kv=layer_kv)
~~~

实际系统还需定义 batch 中不同 session 的 reset mask、KV 淘汰、checkpoint identity 和隐私清除，但这些工程契约不改变上述模型语义。

## 12. 防止 latent 分支退化

### 12.1 主要退化模式

1. **被忽略**：KV 覆盖全部历史，latent cross-attention 权重趋近无效。
2. **只复制局部 token**：`Z_t` 只表示最近措辞，没有保存窗口外信息。
3. **slot 同质化**：所有 slots 学到近似相同表示。
4. **过度写入**：每步大幅覆盖，长期状态持续漂移。
5. **过度保持**：gate 长期关闭，无法吸收新事实。
6. **训练不到远期写入**：TBPTT 在 delayed answer 之前截断梯度。
7. **状态错位**：错误地让 `Z_t` 读取当前 `H_t`，造成训练与推理索引不一致。

### 12.2 结构和数据对策

- 使用有界短窗口 KV，并保证关键事实在回答时确实位于窗口外；
- 数据中包含 delayed recall、约束保持、事实覆盖和干扰信息；
- 使用 learned slot identity 打破初始化对称性；
- 使用门控残差、normalization 和梯度裁剪保持递归稳定；
- 让 TBPTT 或等价训练机制覆盖目标依赖跨度；
- 比较 latent on/off、短 KV 与长 KV，而不是只观察训练 loss；
- 将 gate、slot 相似度和 latent norm 作为诊断指标，不把它们变成默认优化目标。

## 13. 与相关范式的区别

| 范式 | 实际 token 自回归 | 连续递归状态 | 固定长期 slots | 有界 KV |
|---|---:|---:|---:|---:|
| 标准 Transformer + KV | 是 | 仅通过 KV | 否 | 可选 |
| RNN/LSTM | 是 | 是 | 通常单一 state | 不适用 |
| Feedback Transformer | 是 | 高层表示反馈 | 通常无独立固定 slots | 可选 |
| Mamba/RWKV | 是 | 是 | 架构内 state | 不同机制 |
| Coconut 类连续思考 | 部分阶段不输出 token | 是 | 通常不是本方案的长期 slots | 依实现而定 |
| LatentLoop | 是 | 是，`H -> Z` | 是 | 是 |

LatentLoop 的关键组合不是简单增加 hidden skip connection，而是：实际 token 保持文本轨迹一致；完整 `H_(t-1)` 驱动固定容量 `Z_t`；有界 KV 保存近期精确历史；所有可学习模块只由文本 next-token loss 监督。

## 14. 验证设计

### 14.1 基线与消融

至少比较：

| 配置 | 目的 |
|---|---|
| 标准 Transformer + 长 KV | 质量上界与高缓存基线 |
| 标准 Transformer + 相同短 KV | 验证短缓存本身的损失 |
| LatentLoop + 短 KV | 验证固定 latent 的增益 |
| LatentLoop 关闭 latent read | 验证模型是否真正使用 `Z_t` |
| LatentLoop 只保存最后 hidden | 验证完整 `H_t` 的价值 |
| LatentLoop 在答案前 detach | 验证远期梯度路径的重要性 |

### 14.2 文本任务

- 标准语言建模 perplexity 与生成质量；
- 关键信息超过 KV 窗口后的 delayed recall；
- 多轮用户约束和输出格式保持；
- 新事实覆盖旧事实后的正确回答；
- 多个干扰事实下的目标持续；
- 早期工具文本结果在远期的正确引用；
- checkpoint 恢复后与连续运行的 token logits 一致性。

delayed recall 是数据任务，不是独立 head。答案 token 仍用普通 next-token CE 训练和评测。

### 14.3 行为与系统指标

- latent on/off 的窗口外任务差值；
- 不同 KV 长度、slot 数和模型参数预算下的质量曲线；
- 普通短文本能力是否相对参数匹配基线退化；
- `Z_t`、gate 和各层 latent attention 的梯度是否非零且有限；
- slot cosine、latent norm、gate 饱和率和状态漂移；
- 训练吞吐、推理延迟、KV 显存与 latent 固定显存；
- 不同采样策略下状态与实际 token 轨迹的一致性。

### 14.4 成功判据

本方案成立至少需要同时满足：

1. 相同短 KV 下，LatentLoop 在窗口外文本任务上稳定优于无 latent 基线；
2. 关闭 latent read 后长时能力明显下降，证明 `Z_t` 未被忽略；
3. 普通语言建模能力不因递归状态显著退化；
4. 状态显存保持固定，KV 显存受窗口上限约束；
5. 保存完整 `H_t` 相比有损摘要具有可测收益；
6. checkpoint 恢复与连续执行具有数值一致性；
7. 增益来自 next-token 任务监督，而非依赖推理时不存在的标签。

## 15. 风险与边界

| 风险 | 影响 | 约束或缓解 |
|---|---|---|
| 顺序递归降低并行度 | 训练吞吐下降 | 以 block 为 unit，评估并行度与记忆粒度 |
| 长链梯度衰减或爆炸 | 早期写入学不到 | 门控残差、归一化、梯度裁剪和足够梯度跨度 |
| latent 被 KV 忽略 | 无实际长期收益 | 有界 KV、窗口外任务和 latent 消融 |
| 固定 slots 容量不足 | 长会话信息竞争 | 评估 slot/宽度曲线；外部检索属于另一层系统能力 |
| 隐状态难解释 | 调试与审计困难 | 行为评测、状态统计、恢复日志和数据溯源 |
| 跨用户状态泄漏 | 隐私风险 | 明确 session reset、持久化授权和删除策略 |
| 单一 loss 学习信号稀疏 | 收敛慢 | 提高真正 delayed task 的数据密度和有效梯度跨度 |

固定 latent memory 不等同于无限记忆。若任务需要精确保存大量外部事实，应使用显式检索或结构化存储；`Z_t` 更适合保存会影响未来生成策略的压缩任务状态。

## 16. 待验证的关键假设

1. 完整 `H_(t-1)` 含有比单个摘要更适合长期更新的信息。
2. 单一 future next-token loss 足以让 MemoryUpdater 学会有用的写入、增强和遗忘。
3. 有界 KV 与固定 `Z_t` 能在有限显存下互补，而不是互相复制。
4. 多个 learned slots 能在没有显式分工 loss 的情况下形成有效容量利用。
5. 文本 delayed task 的行为增益能迁移到自然多轮对话，而非只拟合合成规则。
6. 按 block 更新记忆可以在训练并行度与状态粒度之间取得可接受平衡。

这些是需要实验回答的问题，不应在缺少消融证据时当作既成事实。

## 17. 结论

LatentLoop 的纯文本目标方案是一个单文本输出头、双反馈通道、三类状态的自回归语言模型：

~~~text
离散文本轨迹：实际 token + 有界 KV_t
长期连续状态：固定容量 Z_t
当前完整状态：H_t

Z_t       = MemoryUpdater(Z_(t-1), H_(t-1))
H_t, KV_t = Backbone(TokenEncoder(X_t), KV_(t-1), Z_t)
text_t    = LMHead(H_t)
~~~

模型只有标准 next-token loss。远期文本输出的梯度通过未来 `H` 和 `Z` 训练早期 MemoryUpdater；真正的 delayed recall 和跨轮约束由数据任务提供，而不是由额外 memory head 或正则定义。KV 负责近期精确历史，`Z_t` 负责固定容量长期压缩，`H_t` 负责连接相邻的记忆更新。

## 18. 参考方向

以下工作提供相关背景，但不等同于 LatentLoop 的完整组合：

1. Fan et al., *Addressing Some Limitations of Transformers with Feedback Memory*, 2020. <https://arxiv.org/abs/2002.09402>
2. Hao et al., *Training Large Language Models to Reason in a Continuous Latent Space (Coconut)*, 2024/2025. <https://arxiv.org/abs/2412.06769>
3. Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023. <https://arxiv.org/abs/2312.00752>
4. Peng et al., *RWKV: Reinventing RNNs for the Transformer Era*, 2023. <https://arxiv.org/abs/2305.13048>

LatentLoop 需要用参数量、显存、KV 窗口和训练数据匹配的对照，独立验证“实际文本 token 自回归 + 完整 `H` 驱动固定 latent memory + 有界 KV + 单一 next-token loss”的增量价值。
