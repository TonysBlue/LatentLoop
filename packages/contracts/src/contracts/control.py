from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ControlKind(StrEnum):
    NOOP = "noop"
    POINTER_MOVE = "pointer_move"
    POINTER_BUTTON = "pointer_button"
    POINTER_DRAG = "pointer_drag"
    SCROLL = "scroll"
    TEXT_INPUT = "text_input"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    WAIT = "wait"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class ControlSignal:
    kind: ControlKind
    event_id: str
    screen_revision: int | None = None
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    dx: float | None = None
    dy: float | None = None
    duration_ms: int | None = None
    text: str | None = None
    key: int | None = None
    button: int | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("control event_id is required")
        for value in (self.x, self.y, self.x2, self.y2):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("pointer coordinates must be in [0, 1]")
        for value in (self.dx, self.dy):
            if value is not None and not -1.0 <= value <= 1.0:
                raise ValueError("scroll values must be in [-1, 1]")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration must be non-negative")
        pointer_kinds = {
            ControlKind.POINTER_MOVE,
            ControlKind.POINTER_BUTTON,
            ControlKind.POINTER_DRAG,
        }
        if self.kind in pointer_kinds:
            if self.x is None or self.y is None:
                raise ValueError("pointer control requires x and y")
        if self.kind is ControlKind.POINTER_DRAG and (self.x2 is None or self.y2 is None):
            raise ValueError("pointer drag requires x2 and y2")
        if self.kind is ControlKind.SCROLL and self.dx is None and self.dy is None:
            raise ValueError("scroll control requires dx or dy")
        if self.kind is ControlKind.WAIT and self.duration_ms is None:
            raise ValueError("wait control requires duration_ms")


@dataclass(frozen=True, slots=True)
class SpeechSignal:
    pcm: bytes
    sample_rate_hz: int = 24_000
    channels: int = 1
    encoding: str = "pcm_f32le"
    silent: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 24_000 or self.channels != 1:
            raise ValueError("speech output must be 24 kHz mono")
        if not self.pcm:
            raise ValueError("speech PCM is required")
        if self.encoding not in {"pcm_f32le", "pcm_s16le"}:
            raise ValueError("unsupported speech encoding")
        bytes_per_sample = 4 if self.encoding == "pcm_f32le" else 2
        expected = round(self.sample_rate_hz * 0.08) * self.channels * bytes_per_sample
        if len(self.pcm) != expected:
            raise ValueError("speech PCM must contain exactly one 80 ms unit")


@dataclass(frozen=True, slots=True)
class ActuationSignal:
    session_id: str
    unit_index: int
    speech: SpeechSignal
    controls: tuple[ControlSignal, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.session_id or self.unit_index < 0:
            raise ValueError("invalid actuation identity")
