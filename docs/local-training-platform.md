# LatentLoop 本地训练平台与工程实施方案

> 状态：工程设计 v0.1  
> 日期：2026-07-31  
> 关联架构：[实时流多模态 LatentLoop 完整方案](realtime-multimodal-latent-loop.md)  
> 目标：在单张 RTX 2080 SUPER 8GB 的 WSL2 环境中，先跑通可扩展、可恢复、可观测的流式多模态训练闭环。

## 1. 设计结论

本地训练平台采用以下固定边界：

```text
Windows 11
└── WSL2 Ubuntu 24.04
    ├── 原生训练进程
    │   ├── uv + Python 3.11
    │   ├── PyTorch + Accelerate
    │   ├── Streaming LatentLoop Model
    │   ├── WebDataset
    │   └── W&B SDK
    │
    ├── Ray Local Runtime
    │   ├── 数据生成与预处理
    │   ├── 环境 Actor
    │   └── 离线评测与 rollout
    │
    └── Docker Engine
        └── wandb/local 单机服务
            └── http://127.0.0.1:8080
```

各组件的职责严格分离：

| 组件 | 职责 | 不承担的职责 |
|---|---|---|
| PyTorch | 模型、流式状态、损失和优化 | 任务集群调度 |
| Accelerate | 单卡 FP16、梯度累积、设备抽象 | 模型状态跨进程传输 |
| WebDataset | 顺序读取和分片音视频轨迹 | 在线环境交互 |
| Ray | CPU 数据任务、环境 Actor、评测和后期 rollout | 实时训练 step 和 GPU 内部状态管理 |
| W&B Local | 实验配置、指标、少量媒体和模型谱系 | 原始数据集和完整 checkpoint 存储 |
| UI-TARS Operator | 后期执行结构化电脑动作 | 模型训练和策略更新 |

模型的 `KV Cache`、`Z_t`、音频 encoder cache 和 speech local state 必须留在同一个 GPU 训练进程。它们不能逐 step 经过 Ray Object Store，否则会引入序列化、显存复制和调度延迟。

## 2. 当前环境与约束

### 2.1 硬件基线

| 项目 | 当前环境 |
|---|---|
| 操作系统 | Windows 11 + WSL2 Ubuntu 24.04 |
| CPU | Intel Core i9-9820X，10 核 20 线程 |
| WSL 内存 | 约 19 GiB，可用约 18 GiB |
| WSL Swap | 5 GiB |
| GPU | NVIDIA GeForce RTX 2080 SUPER |
| 显存 | 8 GiB 专用显存 |
| Compute Capability | 7.5，Turing |
| WSL 驱动报告 | NVIDIA 591.86，CUDA 13.1 |
| Linux 文件系统 | 约 884 GiB 可用空间 |

Windows 显示的共享 GPU 内存不是可替代显存的训练内存。发生显存溢出时，统一内存迁移会严重降低吞吐和实时性，因此所有本机训练配置都必须以 8 GiB 专用显存为硬上限。

### 2.2 运行边界

- 训练直接运行在 WSL2，不放进 Docker，也不通过 Kind。
- W&B Local 是 CPU 服务，不要求 Docker GPU runtime。
- 当前 Docker GPU passthrough 未配置，不影响原生 WSL 训练和 W&B Local。
- 本机只启动一个 GPU worker；Ray Actor 必须声明 `num_gpus=0`。
- Turing 默认使用 FP16，不把 BF16 作为可用精度。
- Attention 优先使用 PyTorch SDPA 或普通 attention，不依赖 FlashAttention 2。
- 第一阶段不引入 FSDP、DeepSpeed、Megatron 或 verl。

### 2.3 R0 环境门槛

开始任何模型训练前必须通过：

```text
nvidia-smi 可识别 RTX 2080 SUPER
torch.cuda.is_available() == True
torch.cuda.get_device_capability() == (7, 5)
FP16 矩阵乘法、前向、反向和 optimizer.step 成功
峰值显存统计可读取
checkpoint 可写入 Linux 文件系统并原子替换
```

R0 未通过时停止在环境层排错，不进入模型和数据问题排查。

## 3. 工程目录

仓库只保存代码、配置、数据索引和部署声明：

```text
LatentLoop/
├── configs/                   模型、数据、训练和追踪配置
├── docs/                      架构与工程设计
├── infra/wandb/               W&B Local Compose
├── scripts/                   环境、服务和备份入口
├── src/latentloop/
│   ├── data/                  数据契约和 WebDataset
│   ├── model/                 流式多模态模型
│   ├── checkpoint.py          原子保存与恢复
│   ├── codec.py               神经 codec 接口
│   ├── losses.py              联合损失
│   ├── ray_jobs.py            CPU 外围任务
│   ├── tracking.py            W&B 适配
│   └── training.py            单 GPU 训练循环
└── tests/                     单元、恢复和烟雾测试
```

大文件统一位于仓库外的 Linux 文件系统：

```text
~/latentloop-data/
├── raw/                       原始录制和外部数据
├── processed/                 WebDataset shards 与 manifest
├── checkpoints/               模型训练状态
├── runs/                      离线 run、评测报告和临时产物
└── backups/                   W&B Local 与关键 manifest 备份
```

不把训练数据放在 `/mnt/c` 或 `/mnt/d`，避免跨 Windows 文件系统处理大量小文件时的元数据和 I/O 开销。路径通过 `runtime.data_root` 配置覆盖，不在 Python 代码中写死。

## 4. 依赖与复现

### 4.1 Python 环境

采用 `uv + Python 3.11`，提交 `pyproject.toml`、`.python-version` 和 `uv.lock`。依赖分组为：

| 分组 | 主要依赖 |
|---|---|
| core | PyTorch、Accelerate、Hydra/OmegaConf、WebDataset、safetensors |
| tracking | W&B SDK |
| ray | Ray Data 和 Ray Core |
| minicpm | Transformers、PEFT、bitsandbytes，后期启用 |
| dev | pytest、coverage、Ruff |

初始锁定基线：

```text
Python          3.11
PyTorch         2.13.0
Accelerate      1.14.0
Ray             2.56.1
W&B SDK         0.28.1
WebDataset      1.0.2
Hydra Core      1.3.4
```

版本升级必须在独立分支完成以下验证后才能更新 lock：

1. CPU 全量测试；
2. RTX 2080 SUPER FP16 前反向；
3. 固定 checkpoint 的下一步输出一致性；
4. W&B Local run 写入；
5. WebDataset 旧 shard 兼容性。

### 4.2 配置优先级

配置来源按以下顺序合并，后者覆盖前者：

```text
dataclass schema
-> YAML profile
-> CLI --set key=value
```

每次 run 必须保存解析后的完整配置，不只保存命令行差异。配置至少包含：

- 模型宽度、层数、head 数和 FFN 宽度；
- 音频 unit、视觉尺寸和 codec 参数；
- KV unit 窗口和 latent slots；
- TBPTT 长度、梯度累积和混合精度；
- 数据 manifest、随机种子和 split；
- checkpoint、评测和 W&B 策略。

## 5. 多模态时间契约

### 5.1 统一时间轴

MVP 使用 500 ms 主干 unit，即 2 Hz tick：

```text
unit_start_ms <= event_time < unit_end_ms
unit_end_ms = unit_start_ms + 500
```

每个 unit 的原始输入为：

| 字段 | 形状或类型 | 说明 |
|---|---|---|
| `timestamp_ms` | `[B] int64` | unit 起始绝对时间 |
| `delta_ms` | `[B] int64` | 实际时间跨度，正常为 500 |
| `mic_audio` | `[B, 12000] float32` | 24 kHz、单声道、单路混合输入 |
| `screen` | `[B, 3, H, W] float32` | 最新关键帧或占位帧 |
| `screen_valid` | `[B] bool` | 当前 unit 是否有新视觉事件 |
| `screen_revision` | `[B] int64` | 屏幕版本，用于动作时效校验 |

`mic_audio` 是设备实际可获得的一路混合音频，其中自然包含现场说话、其他声音、环境噪声和模型扬声器回流。模型不接收播放参考音频、来源分离通道或额外自声标识。

### 5.2 主干 unit token

感知编码后，主干每个 unit 的固定逻辑结构为：

```text
<TIME>
<AUDIO_0> ... <AUDIO_N>
<VISION_OR_EMPTY>
<STATE_QUERY>
```

MVP 使用 4 个音频 token、1 个视觉 token、1 个时间 token和 1 个状态查询 token，共 7 个主干位置。各位置使用独立 type embedding。

`STATE_QUERY` 的最终 hidden state 定义为 `q_t`，同时驱动：

- `Z_t` 更新；
- Direct Speech Head；
- Action Head；
- Speech、Action 和 Cognitive control heads；
- 长期记忆辅助 probe。

### 5.3 Codec 时间对齐

直接语音 MVP 采用 24 kHz 单声道神经 codec：

```text
frame rate       12.5 Hz
codebooks        8
codebook size    2048
frame size       80 ms
```

直接语音路径要求主模型 unit 与 Mimi codec 帧严格对齐，每个 80 ms unit 包含一个有效 codec 帧。因而：

- 每个 unit 输入 1920 个 24 kHz 麦克风采样；
- `speech_codes` 形状固定为 `[B, 1, 8]`；
- KV 窗口按 16 秒定义，对应 200 个 unit；
- 音频和 codec 时间轴不存在累计四舍五入。

神经 codec decoder 是声学解码器，不是 TTS。主干直接预测声学 codec 帧，中间不存在文本生成再转语音的链路。

## 6. 公共训练接口

### 6.1 `StreamUnit`

```python
@dataclass
class StreamUnit:
    timestamp_ms: Tensor
    delta_ms: Tensor
    mic_audio: Tensor
    screen: Tensor
    screen_valid: Tensor
    screen_revision: Tensor

    speech_codes: Tensor
    speech_mask: Tensor
    action_target: ActionTarget
    control_target: ControlTarget
    memory_target: Tensor
```

所有 target 字段都必须配套 mask。缺少某一监督信号时只屏蔽对应 loss，不能丢弃整个 unit。

### 6.2 `RecurrentState`

```python
@dataclass
class RecurrentState:
    audio_cache: Tensor
    layer_kv: tuple[LayerKV, ...]
    latent: Tensor
    speech_local: Tensor
    unit_index: Tensor
```

状态职责为：

| 状态 | 保存内容 | 是否认知记忆 |
|---|---|---|
| `audio_cache` | 流式卷积或音频 attention 的边界样本 | 否 |
| `layer_kv` | 最近完整多模态 unit 的逐层 K/V | 近期精确上下文 |
| `latent` | 固定 slots 的长期目标、计划和任务状态 | 是 |
| `speech_local` | 相邻 codec 帧或因果 decoder 连续状态 | 否 |
| `unit_index` | 时间与恢复游标 | 否 |

`detach()` 必须同时截断所有浮点递归张量，但不重置数值状态。

### 6.3 `StepOutput`

```python
@dataclass
class StepOutput:
    state: RecurrentState
    speech_logits: Tensor  # [B, max_frames, codebooks, vocab]
    action: ActionProposal
    controls: ControlOutput
    memory_logits: Tensor
    latent_gate: Tensor  # [B, latent_slots]
    query: Tensor  # [B, model_dim]
```

模型唯一训练入口为：

```python
output = model.forward_step(unit, recurrent_state)
```

接口不暴露 Ray object reference，不隐式读取全局会话，也不在 forward 内执行操作系统动作或 W&B 网络调用。

### 6.4 神经 codec 接口

```python
class FrozenNeuralCodec(Protocol):
    sample_rate: int
    frame_rate: int
    codebooks: int
    codebook_size: int

    def encode(self, waveform: Tensor) -> Tensor: ...
    def decode(self, codes: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]: ...
```

codec 在第一阶段冻结并单独版本化。checkpoint 和数据 manifest 都要记录 codec 名称、权重 hash、采样率和 codebook 配置，禁止混用不同 codec 生成的标签。

## 7. 模型结构

### 7.1 数据流

```text
MIC_MIXED ----------------> Streaming Audio Encoder ----┐
                                                       |
Screen -> Change/Valid -> Vision Encoder --------------+-> Unit Builder
Time --------------------------------------------------┘       |
                                                               v
Z_t -> Latent Projector -> Cross Attention -> Streaming Backbone
                                             + bounded layer KV
                                                        |
                                                  STATE_QUERY q_t
                                                        |
                         ┌──────────────────────────────┼───────────────┐
                         v                              v               v
                   Latent Updater               Speech Head       Action Head
                         |                              |               |
                      Z_(t+1)                    codec frames      proposal
```

### 7.2 流式主干

每层 cache 只保留最近 `kv_units × tokens_per_unit` 个位置。淘汰发生在完整 unit 边界，不能从 unit 中间裁掉音频或视觉 token。

每层计算为：

```text
hidden
-> causal self-attention with layer KV
-> optional cross-attention to projected Z_t
-> feed-forward network
```

Latent cross-attention 每隔若干层插入。`Z_t` 不拼接到普通 token KV 中，以保持它的容量和时间尺度独立。

### 7.3 Latent Updater

`Z_t` 使用固定 slot 数量：

$$
Z_t\in\mathbb R^{B\times M\times d_z}
$$

更新器读取当前 unit hidden 和 `q_t`，产生候选状态与逐 slot 门：

$$
\widehat Z_{t+1}=G_\phi(Z_t,H_t,q_t)
$$

$$
\alpha_t=\sigma(W_g[Z_t,H_t,q_t])
$$

$$
Z_{t+1}=\mathrm{Norm}((1-\alpha_t)Z_t+\alpha_t\widehat Z_{t+1})
$$

环境反馈尚未通过麦克风或屏幕返回时，`Z_t` 维持表达意图、动作计划和等待状态；反馈进入后续 unit 后再确认或纠正。

### 7.4 Direct Speech Head

Speech Head 输入 `q_t`、pooled `Z_t` 和局部声学状态，每个 80 ms tick 预测一个 Mimi 帧。输出路径为：

```text
q_t + Z_t + speech_local
-> causal/factorized Speech Head
-> neural codec frames
-> frozen causal codec decoder
-> waveform chunk
```

主干不会在下一 unit 直接接收预测 codec embedding。模型输出只有在解码、播放并经过真实或训练模拟声学环境后，才作为混合麦克风的一部分重新进入模型。

### 7.5 Action Head

完整 Action Head 输出：

```python
@dataclass
class ActionProposal:
    type: ActionType
    coordinates: Tensor | None
    scroll_delta: Tensor | None
    duration_ms: int | None
    text_tokens: Tensor | None
    key_mask: Tensor | None
    confidence: float
    observed_screen_revision: int
```

动作类型固定为：

```text
NOOP, CLICK, DOUBLE_CLICK, RIGHT_CLICK, DRAG,
SCROLL, TYPE, HOTKEY, WAIT, CANCEL
```

规则如下：

- 坐标统一归一化到 `[0,1]`，由 Harness 映射到真实显示器；
- `DRAG` 使用起点和终点四个坐标；
- `TYPE` 使用 Action Head 内部的小型文本参数 decoder；
- `HOTKEY` 使用固定键位词表的多标签输出；
- `observed_screen_revision` 取自生成动作所依据的输入 unit；
- 输出动作不会作为 token 回灌主干；
- UI-TARS Operator 执行后的屏幕和声音变化构成环境反馈。

本机状态闭环 MVP 先训练 `NOOP/CLICK/SCROLL/WAIT/CANCEL`，但数据和输出协议从第一版就保留全部动作字段。

## 8. 模型配置档位

### 8.1 Smoke

用于 CI、CPU 和接口测试：

```text
参数量             小于 5M
model_dim          64
layers             2
latent             4 x 64
KV                 4 units
屏幕               32 x 32
音频样本           测试缩短输入
```

Smoke 只验证张量、梯度、cache、数据和恢复语义，不用于质量结论。

### 8.2 Local 10–50M

用于本机完整结构验证：

```text
目标参数量         10M–50M
model_dim          512
layers             8
attention heads    8
FFN                 2048
latent             8 x 512
KV                 16 units = 8 秒
屏幕               224 x 224
audio unit         12000 samples
```

该档位必须能完成数据 overfit、checkpoint 恢复、W&B 追踪和一小时状态稳定性测试。

### 8.3 Research 0.2B

用于验证本机单卡训练上限：

```text
目标参数量         0.15B–0.25B
model_dim          896
layers             18
attention heads    14
FFN                 3584
latent             16 x 1024
KV                 32 units = 16 秒
屏幕               224 x 224
batch size          1
gradient accumulation 16
```

验收以峰值显存不超过 7.5 GiB 为准。若超出，依次启用 activation checkpointing、减少 TBPTT unit、降低视觉分辨率，再缩小模型；不依赖共享 GPU 内存维持训练。

该档位用于机制研究，不承担从头恢复 MiniCPM-o 级别知识的目标。裁剪后大模型的大规模持续预训练和恢复训练迁移到多 GPU 环境。

## 9. 数据格式

### 9.1 WebDataset shard

每个 episode 使用同一 key 保存：

```text
<episode>.meta.json
<episode>.mic.flac
<episode>.screen.npz
<episode>.speech_codes.npy
<episode>.actions.json
<episode>.controls.npy
```

字段定义：

| 文件 | 内容 |
|---|---|
| `meta.json` | schema、时间戳、设备、场景、unit 数、有效 mask 和版本 hash |
| `mic.flac` | 24 kHz 单声道连续混合音频 |
| `screen.npz` | 按 unit 对齐的关键帧或预编码视觉输入 |
| `speech_codes.npy` | 目标直接语音 codec 帧 |
| `actions.json` | 结构化动作和字段 mask |
| `controls.npy` | Speech、Action、Cognitive control 标签 |

Shard 默认不超过 1 GB，单个 episode 不跨 shard。每组 shard 配套 `manifest.jsonl`，每行记录：

```text
schema_version
episode_id
split
source
scenario
device_id_hash
session_id_hash
duration_ms
unit_count
codec_id
content_sha256
```

### 9.2 Split 规则

- 按完整 session 划分 train/validation/test；
- 同一设备和同一录制会话的相邻片段不能跨 split；
- 合成场景按随机种子和场景模板划分；
- 真实测试集保留未出现在训练中的房间、设备和说话组合；
- 数据 manifest hash 是 run 身份的一部分。

### 9.3 第一批数据

第一阶段由两部分组成：

1. **合成轨迹**：精确控制长期 memory label、视觉目标、动作、延迟、回流和噪声；
2. **少量人工录制**：在本机录制真实单路麦克风、屏幕和操作轨迹，用于验证合成到真实的分布差异。

训练样本始终只有一路混合麦克风作为音频输入。数据构造过程中的辅助信息只能用于生成 target、质量检查或评测，不能进入模型输入张量。

### 9.4 数据校验

入库前必须验证：

- 时间戳严格递增且音视频落入正确 unit；
- 音频采样率、声道数和长度一致；
- codec 帧数量符合全局 12.5 Hz 时间轴；
- 坐标处于 `[0,1]`；
- action 字段与 type mask 匹配；
- screen revision 单调递增；
- 不存在跨 split 的 session/device 泄漏；
- FLAC、NPY、NPZ 和 JSON 均可完整解码；
- manifest hash 与内容一致。

损坏 episode 整体隔离到 quarantine manifest，不在训练中静默跳过并继续计数。

## 10. 训练循环

### 10.1 单卡执行

每个 episode 顺序处理，不打乱 episode 内 unit：

```text
initialize recurrent state
for each TBPTT chunk:
    for each StreamUnit in chronological order:
        output = model.forward_step(unit, state)
        accumulate masked losses
        state = output.state
    backward(chunk_loss)
    optimizer step according to gradient accumulation
    detach recurrent state
```

关键不变量：

- 在同一个 TBPTT chunk 内保留跨 unit 梯度；
- 只在 chunk 边界反向并 detach；
- 不能每个 unit optimizer step 后复用旧计算图；
- episode 边界默认重置 KV、audio cache 和 speech local state；
- 是否继承会话级 `Z_t` 由数据显式标记，默认不继承。

### 10.2 优化配置

初始默认值：

| 项目 | 10–50M | 约 0.2B |
|---|---:|---:|
| optimizer | AdamW | 8-bit AdamW |
| learning rate | `3e-4` | `2e-4` |
| weight decay | `0.1` | `0.1` |
| micro batch | 1 | 1 |
| gradient accumulation | 16 | 16 |
| TBPTT | 4 units | 2–4 units |
| precision | FP16 | FP16 |
| grad clip | 1.0 | 1.0 |
| activation checkpoint | 可选 | 必须 |

8-bit optimizer 如果未通过当前 CUDA/Turing 兼容测试，则回退到标准 AdamW 并进一步缩小模型。该回退不得改变 checkpoint 对外格式。

### 10.3 联合损失

状态优先阶段使用：

$$
\mathcal L=
\mathcal L_{speech}
+\mathcal L_{action-type}
+\mathcal L_{action-param}
+0.25\mathcal L_{controls}
+\mathcal L_{memory}
+0.01\mathcal L_{write}
$$

其中：

- `L_speech`：有效 codec frame 和 codebook 上的交叉熵；
- `L_action-type`：动作类型交叉熵；
- `L_action-param`：只对有效字段计算坐标、滚动、时长、文本或按键损失；
- `L_controls`：三个并发控制 head 的分类损失之和；
- `L_memory`：跨 KV 窗口仍需保持的目标、约束或任务阶段 probe；
- `L_write`：latent 写入预算和 slot 活跃度正则。

每项 loss 都记录未加权值和加权值。缺少标签时使用 task mask，不把空标签当成 `NOOP` 或静音负样本。

### 10.4 状态优先课程

| 阶段 | 训练内容 | 冻结策略 | 进入下一阶段条件 |
|---|---|---|---|
| T0 | 单 unit 张量与所有 Head | 无 | 32 条轨迹可 overfit |
| T1 | 多 unit、短 KV、checkpoint | 无 | 恢复结果一致，KV 有界 |
| T2 | `Z_t`、跨窗口 memory | 先训练 updater/adapter | `Z_t` 消融产生显著差异 |
| T3 | codec 直接语音 | codec decoder 冻结 | codec 小样本准确率达标 |
| T4 | 模拟声学回流 | 逐步加入模型输出 | 无重复响应和队列漂移 |
| T5 | Action Head 模仿学习 | 可冻结声学模块 | 动作和 screen revision 达标 |
| T6 | 真实本机联合轨迹 | 分模块低学习率 | 一小时稳定运行 |

### 10.5 声学闭环课程

训练模拟链路为：

```text
target/model codec frames
-> frozen codec decoder
-> playback delay + room impulse + device response
-> mix with future scene audio
-> next MIC_MIXED units
```

按以下比例逐步替换：

```text
100% target speech feedback
-> 75% target + 25% generated
-> 50% target + 50% generated
-> 25% target + 75% generated
-> closed-loop rollout
```

延迟、音量、混响、时钟漂移、截断和回流缺失均随机化。所有未来音频最终都表现为同一路混合麦克风，不增加模型可见的来源标签。

## 11. Checkpoint 与恢复

### 11.1 保存内容

checkpoint 格式必须版本化，包含：

```text
format_version
model weights
optimizer state
LR scheduler state
AMP scaler state
Python / NumPy / Torch / CUDA RNG
RecurrentState
epoch / shard / sample / unit cursor
global update / consumed units
resolved config + config hash
data manifest hash
codec id + codec weight hash
parent checkpoint hash
git commit
```

`RecurrentState` 用于验证中途恢复语义；正式 epoch checkpoint 默认保存在完整 TBPTT 边界，避免恢复到半个计算图。

### 11.2 原子写入

保存流程固定为：

1. 写入同一文件系统的临时文件；
2. flush 并 `fsync` 文件；
3. 计算 SHA-256；
4. 原子 rename 到最终路径；
5. `fsync` checkpoint 目录；
6. 更新独立 manifest；
7. 最后把路径和 hash 写入 W&B summary。

任何中断都只能留下可识别的临时文件，不能覆盖最近可用 checkpoint。

### 11.3 数据游标

单机 WebDataset 恢复游标为：

```text
epoch
ordered_shard_index
sample_index_in_shard
unit_index_in_episode
```

Shard 顺序由 `seed + epoch` 唯一确定。恢复时直接定位 shard，再顺序跳过 shard 内已消费样本。第一版不依赖不可序列化的生成器内部状态。

### 11.4 保留策略

- 保留最近 3 个周期 checkpoint；
- 保留验证指标最优的 2 个 checkpoint；
- 每个训练阶段保留 1 个人工标记 milestone；
- 删除前确认对应 SHA-256 已写入 checkpoint manifest；
- W&B 只记录本地路径、hash 和父子关系，不上传大模型文件。

## 12. W&B Local

### 12.1 部署定义

本地服务固定使用：

```text
image   wandb/local:0.83.0
digest  sha256:b234c9d084b65164598da6fa5f17d38ec71137b037c059b46889c1495b008c52
listen  127.0.0.1:8080
volume  latentloop-wandb -> /vol
limit   2 CPU / 4 GiB RAM
ready   http://127.0.0.1:8080/ready
```

不使用 `latest`。API key 放在本地环境文件或 shell secret 中，不提交 Git。

该部署定位为个人开发、POC 和小规模研究记录。它不作为生产级 W&B Self-Managed 架构，不开放局域网或公网，也不承担高可用和多人权限治理。

### 12.2 Run 身份

Run 名称为：

```text
<stage>-<model-profile>-<UTC timestamp>-<git-short-sha>
```

Run config 和 summary 必须包含：

- 完整解析配置和 config hash；
- git commit 和 dirty 标记；
- 数据 manifest hash；
- codec id 与权重 hash；
- 参数量、可训练参数量和模型结构版本；
- parent checkpoint 路径与 hash；
- GPU、驱动、PyTorch 和 CUDA 版本；
- random seed 和 split。

### 12.3 指标命名

统一命名空间：

| 前缀 | 内容 |
|---|---|
| `train/` | 总损失、各任务损失、学习率、梯度范数 |
| `val/` | 验证损失和任务成功率 |
| `speech/` | codec accuracy、RTF、首块延迟、连续性 |
| `action/` | 类型准确率、坐标误差、过期动作率 |
| `latent/` | gate、活跃 slots、on/off 差值、probe |
| `stream/` | unit 延迟、KV 长度、积压、丢帧 |
| `system/` | 显存、GPU 利用率、RAM、磁盘 |
| `data/` | shard、样本、损坏率和场景分布 |

指标按 optimizer update 作为 W&B step。unit 和 episode 计数作为独立字段，不能混用为 step。

### 12.4 媒体与隐私

每个验证 run 最多记录：

```text
4 条脱敏轨迹
每条最多 15 秒音频
每条最多 8 张屏幕帧
对应动作、控制状态和时间戳表格
```

原始音视频不上传 W&B，不把完整 checkpoint 写入 Artifact。屏幕样本必须经过敏感区域遮挡和文本脱敏后才能进入 W&B volume。

### 12.5 离线降级

训练开始时探测本地服务：

- 服务正常：`WANDB_MODE=online`；
- 连接失败或初始化超时：自动切换 `WANDB_MODE=offline`；
- W&B 异常不得终止训练；
- 服务恢复后人工同步离线 run；
- 同一 run ID 只同步一次，防止重复实验。

### 12.6 容量与备份

- volume 达到 40 GB 时告警；
- 达到 50 GB 时停止写入新媒体，只保留指标；
- 每周执行一次停服一致性备份；
- 备份写入 `~/latentloop-data/backups`；
- 只保留最近 4 份 W&B 备份；
- 每月至少执行一次空 volume 恢复演练。

W&B 备份不替代 checkpoint 和数据 manifest 备份。

## 13. Ray 外围编排

### 13.1 本地配置

```text
mode                 local single-node
num_cpus             8
num_gpus             0 for all Ray tasks
object_store_memory  1 GiB
dashboard            disabled by default
```

预留 CPU 和内存给训练进程、W&B 和 WSL。Ray object reference 中只保存样本 block、路径或小型评测结果，不保存模型参数、KV、`Z_t` 或 waveform 播放队列。

### 13.2 第一阶段任务

Ray 负责：

- 合成 episode 并行生成；
- 音频重采样、声学随机化和 codec 离线编码；
- 屏幕帧变化检测和压缩；
- shard 校验与 manifest 统计；
- checkpoint 离线评测；
- 后期 UI-TARS 环境 Actor。

Ray 不负责：

- 包装单卡核心训练循环；
- 每个 unit 调用远程 GPU actor；
- 在 Object Store 中维护 recurrent state；
- 同时启动多个竞争同一 GPU 的 trial。

### 13.3 后期环境 Actor

每个 UI 环境 Actor 持有：

```text
isolated desktop/session
task specification
screen revision
action history
safety policy
reward and termination state
```

Actor 接收 `ActionProposal`，由 UI-TARS Operator 执行并返回带时间戳的屏幕、声音、执行状态和 reward。模型推理进程只接收新的环境观察，不接收“动作已成功”的隐藏旁路标签。

## 14. 验收与测试

### 14.1 R0：环境

- 原生 WSL CUDA 可用；
- FP16 前向、反向和 optimizer step 成功；
- W&B Local 重启后 run 仍存在；
- named volume 可完成备份和空 volume 恢复。

### 14.2 R1：数据与替身模型

- 所有数据格式可往返编码；
- 时间、mask、动作和 codec schema 校验通过；
- Smoke 模型所有 trainable Head 获得非空梯度；
- 10–50M 模型能在 32 条固定轨迹上 overfit。

### 14.3 R2：状态闭环

- KV 长度始终不超过配置上限；
- 淘汰只发生在完整 unit 边界；
- `Z_t` 形状和容量不随时间增长；
- TBPTT chunk 内有跨 unit 梯度，chunk 外图已断开；
- 长期合成任务上“短 KV + Z”比“短 KV”准确率至少高 20 个百分点。

### 14.4 R3：语音与动作

- 固定小样本上 codec token accuracy 超过 90%；
- Action type accuracy 超过 95%；
- 点击坐标平均绝对误差低于屏幕宽高的 2%；
- 过期 `screen_revision` 动作全部被 Harness 拒绝；
- 输出语音不通过文本或 TTS 中间链路。

### 14.5 R4：恢复

- checkpoint 包含完整训练和递归状态；
- 中断只留下临时文件，不破坏最近 checkpoint；
- 恢复后的下一 unit loss 与连续训练在 FP16 容差内一致；
- 数据恢复不重复或遗漏 episode/unit；
- checkpoint config 或 codec hash 不匹配时明确拒绝加载。

### 14.6 R5：性能与稳定性

- 约 0.2B 配置峰值显存低于 7.5 GiB；
- Ray CPU worker 不申请 GPU；
- Ray Object Store 不超过 1 GiB；
- 连续运行一小时后 KV、`Z_t` 和队列容量无增长；
- 所有核心指标进入 W&B 或离线 run；
- 发生 OOM、数据损坏或 W&B 断开时给出可定位错误。

### 14.7 测试矩阵

| 类型 | 场景 |
|---|---|
| 单元测试 | 配置校验、mask loss、Action 参数、codec 时间轴 |
| 状态测试 | KV 淘汰、latent detach、会话 reset、空视觉事件 |
| 数据测试 | shard 往返、损坏文件、时间倒退、split 泄漏 |
| 恢复测试 | RNG、optimizer、recurrent state、数据游标 |
| 集成测试 | 合成数据到训练到 checkpoint 到 W&B |
| GPU 测试 | FP16、峰值显存、activation checkpointing |
| 稳定性测试 | 一小时队列、内存、cache 和断线恢复 |

## 15. 实施阶段

### E0：工程基线

- 固定 uv/Python/PyTorch 环境；
- 建立配置 schema、CLI 和目录；
- 实现 CPU Smoke 模型和测试；
- 部署 W&B Local 并验证持久化。

完成标准：R0 全部通过。

### E1：状态闭环 MVP

- 实现 streaming audio/vision encoder；
- 实现逐层有界 KV；
- 实现 `Z_t` cross-attention、updater 和写入门；
- 实现 Action/Control/Memory heads；
- 实现 TBPTT 和完整 checkpoint。

完成标准：R1、R2、R4 通过。

### E2：直接语音

- 固定并版本化 24 kHz 神经 codec；
- 离线编码 target speech；
- 实现因果/factorized Speech Head；
- 实现流式 codec decoder state；
- 记录 codec accuracy、RTF 和边界连续性。

完成标准：R3 语音部分通过。

### E3：真实数据与动作

- 建立本机录制和脱敏管线；
- 补齐全部 Action Head 参数；
- 接入 UI-TARS Operator 协议；
- 训练视觉 grounding、动作时效和失败恢复。

完成标准：R3 动作部分和本机真实轨迹验证通过。

### E4：声学环境闭环

- 生成目标语音回流数据；
- 逐步混入模型生成语音；
- 加入延迟、混响、噪声、重叠和截断；
- 完成用户打断与一小时全双工稳定性测试。

完成标准：R5 通过，重复回应和队列漂移受控。

### E5：约 0.2B 与 MiniCPM 迁移

- 扩展本机研究模型到 0.15B–0.25B；
- 完成显存和吞吐优化；
- 建立 MiniCPM adapter 接口和权重映射；
- 将裁剪、宽度缩减和恢复训练迁移到多 GPU；
- 多卡阶段引入 FSDP2 或 DeepSpeed。

### E6：在线 Agent RL

- Ray Actor 并行运行 UI 环境；
- UI-TARS Operator 执行动作；
- 离线模仿学习稳定后改造 verl；
- RL policy 只优化 Action/Control 或受控模块；
- 模型 recurrent state 显式跨 rollout step 保存。

## 16. 安全与故障处理

### 16.1 数据安全

- 原始屏幕和麦克风数据默认仅保存在本机；
- 数据集使用不可逆 device/session hash；
- W&B 媒体必须脱敏；
- `.env`、API key 和用户内容不提交 Git；
- 删除数据时同步更新 manifest 和数据谱系。

### 16.2 训练故障

| 故障 | 行为 |
|---|---|
| CUDA OOM | 保存诊断，不保存半完成 step；减 TBPTT 后从最近 checkpoint 恢复 |
| NaN/Inf | 跳过 optimizer step，记录输入 key、loss 和 grad norm；连续触发则终止 |
| 数据损坏 | episode 进入 quarantine；训练计数不前进 |
| W&B 断开 | 切换 offline，不中断训练 |
| 磁盘不足 | 停止新 checkpoint 和媒体写入，保持现有文件完整 |
| Ray worker 失败 | 只重试幂等的 CPU 数据任务，不影响训练 recurrent state |

### 16.3 动作安全

Action Head 的输出必须经过 Harness：

- screen revision 校验；
- 坐标范围和动作速率限制；
- 删除、支付、发送、安装和权限修改审批；
- 应用和区域白名单；
- 超时、取消和全局紧急停止；
- 完整动作和环境反馈轨迹。

## 17. 最终工程定义

本地阶段的完整训练闭环为：

```text
WebDataset chronological episodes
-> StreamUnit(500 ms MIC_MIXED + screen)
-> Streaming Audio/Vision Encoders
-> bounded multimodal KV + fixed Z_t
-> Direct Neural-Codec Speech Head
-> Independent Action/Control Heads
-> masked multi-task loss + TBPTT
-> atomic checkpoint
-> W&B Local metrics and lineage
```

Ray 位于该闭环外围，负责数据、环境和评测；W&B Local 位于观测面，不进入模型计算；Docker 不承载本机 GPU 训练。这样本机可以先验证 LatentLoop 的核心机制，并保持未来迁移到 MiniCPM、多 GPU、UI-TARS 和 verl 的接口连续性。
