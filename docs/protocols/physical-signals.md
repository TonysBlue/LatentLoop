# 物理信号协议

Model Service 数据平面使用版本化 Protobuf over Unix socket。ObservationSignal 包含 24 kHz
mono mic PCM、完整屏幕像素、timestamp 和 delta；ActuationSignal 包含 24 kHz speech PCM
和 decoded ControlSignal。reward、receipt、task success 和隐藏环境字段属于控制平面。
