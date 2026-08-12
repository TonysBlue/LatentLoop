from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ControlKind(StrEnum):
    NOOP = "noop"
    POINTER_MOVE = "pointer_move"
    POINTER_BUTTON = "pointer_button"
    SCROLL = "scroll"
    TEXT_INPUT = "text_input"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"


class ButtonPhase(StrEnum):
    CLICK = "click"
    DOWN = "down"
    UP = "up"


@dataclass(frozen=True, slots=True)
class ControlSignal:
    kind: ControlKind
    event_id: str
    x: float | None = None
    y: float | None = None
    dx: float | None = None
    dy: float | None = None
    text: str | None = None
    key: int | None = None
    button: int | None = None
    button_phase: ButtonPhase | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("control event_id is required")
        for value in (self.x, self.y):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("pointer coordinates must be in [0, 1]")
        for value in (self.dx, self.dy):
            if value is not None and not -1.0 <= value <= 1.0:
                raise ValueError("scroll values must be in [-1, 1]")
        if self.kind is ControlKind.POINTER_MOVE and (self.x is None or self.y is None):
            raise ValueError("pointer move requires x and y")
        if self.kind is ControlKind.POINTER_BUTTON:
            if self.button is None or self.button_phase is None:
                raise ValueError("pointer button requires button and phase")
            if self.button not in {0, 1, 2}:
                raise ValueError("pointer button must be left, middle, or right")
        if self.kind is ControlKind.SCROLL and self.dx is None and self.dy is None:
            raise ValueError("scroll control requires dx or dy")
        if self.kind is ControlKind.TEXT_INPUT and not self.text:
            raise ValueError("text input requires non-empty text")
        if self.kind in {ControlKind.KEY_PRESS, ControlKind.KEY_RELEASE} and self.key is None:
            raise ValueError("keyboard control requires a key")


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
