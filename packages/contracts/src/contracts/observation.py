from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MicSignal:
    samples: bytes
    sample_rate_hz: int = 24_000
    channels: int = 1
    encoding: str = "pcm_f32le"

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 24_000 or self.channels != 1:
            raise ValueError("the realtime audio clock requires 24 kHz mono")
        if self.encoding not in {"pcm_f32le", "pcm_s16le"}:
            raise ValueError("unsupported microphone encoding")
        if not self.samples:
            raise ValueError("microphone samples are required")


@dataclass(frozen=True, slots=True)
class ScreenSignal:
    image: bytes
    width: int
    height: int
    revision: int
    valid: bool = True
    pixel_format: str = "rgb24"
    encoding: str = "raw"

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("screen dimensions must be positive")
        if self.revision < 0:
            raise ValueError("screen revision must be non-negative")
        if self.pixel_format not in {"rgb24", "rgba32"}:
            raise ValueError("unsupported screen pixel format")


@dataclass(frozen=True, slots=True)
class ObservationSignal:
    session_id: str
    unit_index: int
    timestamp_ms: int
    delta_ms: int
    mic: MicSignal
    screen: ScreenSignal

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.unit_index < 0 or self.timestamp_ms < 0 or self.delta_ms <= 0:
            raise ValueError("invalid observation clock")
