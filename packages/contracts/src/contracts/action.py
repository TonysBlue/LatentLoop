from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from contracts.control import ButtonPhase, ControlKind, ControlSignal

ACTION_SCHEMA_ID = "structured-action-v1"
COORDINATE_GRID_SIZE = 32
TYPE_BYTES_PER_UNIT = 16
HOTKEY_KEYS_PER_UNIT = 8
KEY_VOCAB_SIZE = 32


class ActionKind(IntEnum):
    NO_ACTION = 0
    NOOP = 1
    POINTER_MOVE = 2
    POINTER_BUTTON = 3
    SCROLL = 4
    TYPE = 5
    HOTKEY = 6


class PointerButton(IntEnum):
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2


class PointerButtonPhase(IntEnum):
    CLICK = 0
    DOWN = 1
    UP = 2


@dataclass(frozen=True, slots=True)
class ActionFrame:
    kind: ActionKind
    coordinate_cell: int = 0
    coordinate_residual: tuple[float, float] = (0.0, 0.0)
    button: PointerButton = PointerButton.LEFT
    button_phase: PointerButtonPhase = PointerButtonPhase.CLICK
    scroll_delta: tuple[float, float] = (0.0, 0.0)
    text_bytes: bytes = b""
    hotkey_keys: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        cell_count = COORDINATE_GRID_SIZE**2
        if not 0 <= self.coordinate_cell < cell_count:
            raise ValueError("coordinate cell is outside the configured grid")
        if any(not 0.0 <= value <= 1.0 for value in self.coordinate_residual):
            raise ValueError("coordinate residual must be in [0, 1]")
        if any(not -1.0 <= value <= 1.0 for value in self.scroll_delta):
            raise ValueError("scroll delta must be in [-1, 1]")
        if len(self.text_bytes) > TYPE_BYTES_PER_UNIT:
            raise ValueError("TYPE exceeds the per-unit byte limit")
        if len(self.hotkey_keys) > HOTKEY_KEYS_PER_UNIT:
            raise ValueError("HOTKEY exceeds the per-unit key limit")
        if any(not 0 <= key < KEY_VOCAB_SIZE for key in self.hotkey_keys):
            raise ValueError("HOTKEY key is outside the configured vocabulary")
        if len(set(self.hotkey_keys)) != len(self.hotkey_keys):
            raise ValueError("HOTKEY keys must be unique")
        if self.kind is ActionKind.TYPE and not self.text_bytes:
            raise ValueError("TYPE requires at least one byte")
        if self.kind is ActionKind.HOTKEY and not self.hotkey_keys:
            raise ValueError("HOTKEY requires at least one key")
        if self.kind is not ActionKind.POINTER_MOVE and (
            self.coordinate_cell != 0 or self.coordinate_residual != (0.0, 0.0)
        ):
            raise ValueError("only POINTER_MOVE accepts coordinate parameters")
        if self.kind is not ActionKind.POINTER_BUTTON and (
            self.button is not PointerButton.LEFT
            or self.button_phase is not PointerButtonPhase.CLICK
        ):
            raise ValueError("only POINTER_BUTTON accepts button parameters")
        if self.kind is not ActionKind.SCROLL and self.scroll_delta != (0.0, 0.0):
            raise ValueError("only SCROLL accepts scroll parameters")
        if self.kind is not ActionKind.TYPE and self.text_bytes:
            raise ValueError("only TYPE accepts text bytes")
        if self.kind is not ActionKind.HOTKEY and self.hotkey_keys:
            raise ValueError("only HOTKEY accepts keys")


@dataclass(frozen=True, slots=True)
class ActionDecodeResult:
    controls: tuple[ControlSignal, ...]
    pending_utf8: bytes = b""


def _utf8_prefix(value: bytes) -> tuple[str, bytes]:
    if not value:
        return "", b""
    try:
        return value.decode("utf-8"), b""
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data" and len(value) - error.start <= 3:
            prefix = value[: error.start].decode("utf-8")
            return prefix, value[error.start:]
        raise ValueError("TYPE contains invalid UTF-8") from error


def decode_action_frame(
    frame: ActionFrame,
    *,
    event_id: str,
    screen_revision: int | None = None,
    pending_utf8: bytes = b"",
) -> ActionDecodeResult:
    """Decode one unit frame into zero or more ordered physical controls."""

    if len(pending_utf8) > 3:
        raise ValueError("pending UTF-8 state exceeds three bytes")
    if frame.kind is not ActionKind.TYPE and pending_utf8:
        raise ValueError("cannot leave TYPE with an incomplete UTF-8 sequence")
    if frame.kind is ActionKind.NO_ACTION:
        return ActionDecodeResult(())
    if frame.kind is ActionKind.NOOP:
        return ActionDecodeResult((ControlSignal(ControlKind.NOOP, event_id),))
    if frame.kind is ActionKind.POINTER_MOVE:
        cell_x = frame.coordinate_cell % COORDINATE_GRID_SIZE
        cell_y = frame.coordinate_cell // COORDINATE_GRID_SIZE
        x = min((cell_x + frame.coordinate_residual[0]) / COORDINATE_GRID_SIZE, 1.0)
        y = min((cell_y + frame.coordinate_residual[1]) / COORDINATE_GRID_SIZE, 1.0)
        return ActionDecodeResult(
            (
                ControlSignal(
                    ControlKind.POINTER_MOVE,
                    event_id,
                    screen_revision=screen_revision,
                    x=x,
                    y=y,
                ),
            )
        )
    if frame.kind is ActionKind.POINTER_BUTTON:
        phase = {
            PointerButtonPhase.CLICK: ButtonPhase.CLICK,
            PointerButtonPhase.DOWN: ButtonPhase.DOWN,
            PointerButtonPhase.UP: ButtonPhase.UP,
        }[frame.button_phase]
        return ActionDecodeResult(
            (
                ControlSignal(
                    ControlKind.POINTER_BUTTON,
                    event_id,
                    screen_revision=screen_revision,
                    button=int(frame.button),
                    button_phase=phase,
                ),
            )
        )
    if frame.kind is ActionKind.SCROLL:
        return ActionDecodeResult(
            (
                ControlSignal(
                    ControlKind.SCROLL,
                    event_id,
                    screen_revision=screen_revision,
                    dx=frame.scroll_delta[0],
                    dy=frame.scroll_delta[1],
                ),
            )
        )
    if frame.kind is ActionKind.TYPE:
        text, remaining = _utf8_prefix(pending_utf8 + frame.text_bytes)
        controls = (
            (ControlSignal(ControlKind.TEXT_INPUT, event_id, text=text),) if text else ()
        )
        return ActionDecodeResult(controls, remaining)
    if frame.kind is ActionKind.HOTKEY:
        pressed = tuple(
            ControlSignal(ControlKind.KEY_PRESS, f"{event_id}-down-{index}", key=key)
            for index, key in enumerate(frame.hotkey_keys)
        )
        released = tuple(
            ControlSignal(ControlKind.KEY_RELEASE, f"{event_id}-up-{index}", key=key)
            for index, key in enumerate(reversed(frame.hotkey_keys))
        )
        return ActionDecodeResult(pressed + released)
    raise AssertionError("unreachable action kind")
