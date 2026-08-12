# Model Service

Model Service 是模型推理边界，不负责采集设备或执行电脑动作。

输入是 80 ms 的混合麦克风 PCM、完整屏幕图像和时间；输出是 24 kHz
语音 PCM 与已经由 Structured ActionFrame 解码得到的 `ControlSignal`。服务维护每个 session 的
`Z/H/KV/audio/speech-local/action-local` 状态，并在 session reset 时清空。

Mimi token 和 ActionFrame tensor 只存在内部 runtime。Harness 不接收 raw 模型输出。
