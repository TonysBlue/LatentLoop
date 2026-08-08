from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from latentloop.types import ActionType


class ActionToken(IntEnum):
    PAD = 0
    END_ACTION = 1
    NOOP = 2
    CLICK = 3
    DOUBLE_CLICK = 4
    RIGHT_CLICK = 5
    DRAG = 6
    SCROLL = 7
    TYPE = 8
    HOTKEY = 9
    WAIT = 10
    CANCEL = 11


@dataclass(frozen=True, slots=True)
class ActionEvent:
    type: ActionType
    coordinates: tuple[float, ...] | None = None
    scroll_delta: tuple[float, float] | None = None
    duration_ms: float | None = None
    text: str | None = None
    keys: tuple[int, ...] | None = None


class ActionTokenizer:
    """Fixed, versioned token protocol for every computer action."""

    coordinate_bins = 256
    scroll_bins = 256
    duration_bins = 128
    byte_tokens = 256
    key_tokens = 32
    burst_tokens = 16

    coordinate_base = 12
    scroll_base = coordinate_base + coordinate_bins
    duration_base = scroll_base + scroll_bins
    byte_base = duration_base + duration_bins
    key_base = byte_base + byte_tokens
    vocab_size = key_base + key_tokens

    def __init__(self, max_duration_ms: int, burst_tokens: int = 16) -> None:
        if max_duration_ms <= 0:
            raise ValueError("max_duration_ms must be positive")
        if burst_tokens < 1:
            raise ValueError("burst_tokens must be positive")
        self.max_duration_ms = float(max_duration_ms)
        self.burst_tokens = burst_tokens
        self.end_action = int(ActionToken.END_ACTION)

    @staticmethod
    def _coord(value: float) -> int:
        if not 0.0 <= value <= 1.0:
            raise ValueError("coordinates must be in [0, 1]")
        return int(round(value * (ActionTokenizer.coordinate_bins - 1)))

    @staticmethod
    def _signed(value: float) -> int:
        if not -1.0 <= value <= 1.0:
            raise ValueError("signed action values must be in [-1, 1]")
        return int(round((value + 1.0) * 0.5 * (ActionTokenizer.scroll_bins - 1)))

    def _duration(self, value: float) -> int:
        if not 0.0 <= value <= self.max_duration_ms:
            raise ValueError("duration is outside configured range")
        return int(round(value / self.max_duration_ms * (self.duration_bins - 1)))

    def encode(self, event: ActionEvent) -> list[int]:
        kind = {
            ActionType.NOOP: ActionToken.NOOP,
            ActionType.CLICK: ActionToken.CLICK,
            ActionType.DOUBLE_CLICK: ActionToken.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK: ActionToken.RIGHT_CLICK,
            ActionType.DRAG: ActionToken.DRAG,
            ActionType.SCROLL: ActionToken.SCROLL,
            ActionType.TYPE: ActionToken.TYPE,
            ActionType.HOTKEY: ActionToken.HOTKEY,
            ActionType.WAIT: ActionToken.WAIT,
            ActionType.CANCEL: ActionToken.CANCEL,
        }[event.type]
        tokens = [int(kind)]
        if event.type in {
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.DRAG,
        }:
            expected = 2 if event.type is not ActionType.DRAG else 4
            if event.coordinates is None or len(event.coordinates) != expected:
                raise ValueError(f"{event.type.name} requires {expected} coordinates")
            tokens.extend(self.coordinate_base + self._coord(value) for value in event.coordinates)
        elif event.type is ActionType.SCROLL:
            if event.scroll_delta is None or len(event.scroll_delta) != 2:
                raise ValueError("SCROLL requires two deltas")
            tokens.extend(self.scroll_base + self._signed(value) for value in event.scroll_delta)
        elif event.type is ActionType.WAIT:
            if event.duration_ms is None:
                raise ValueError("WAIT requires duration_ms")
            tokens.append(self.duration_base + self._duration(event.duration_ms))
        elif event.type is ActionType.TYPE:
            if event.text is None:
                raise ValueError("TYPE requires text")
            tokens.extend(self.byte_base + value for value in event.text.encode("utf-8"))
        elif event.type is ActionType.HOTKEY:
            if not event.keys:
                raise ValueError("HOTKEY requires at least one key")
            if any(key < 0 or key >= self.key_tokens for key in event.keys):
                raise ValueError("HOTKEY key is outside the configured vocabulary")
            tokens.extend(self.key_base + key for key in event.keys)
        tokens.append(self.end_action)
        return tokens

    def split_bursts(self, tokens: list[int]) -> list[list[int]]:
        if not tokens:
            raise ValueError("action token sequence cannot be empty")
        return [
            tokens[start : start + self.burst_tokens]
            for start in range(0, len(tokens), self.burst_tokens)
        ]

    def decode(self, tokens: list[int]) -> ActionEvent:
        if not tokens:
            raise ValueError("action token sequence cannot be empty")
        if tokens[-1] != self.end_action:
            raise ValueError("action sequence must end with END_ACTION")
        kind = ActionToken(tokens[0])
        body = tokens[1:-1]
        try:
            action_type = {
                ActionToken.NOOP: ActionType.NOOP,
                ActionToken.CLICK: ActionType.CLICK,
                ActionToken.DOUBLE_CLICK: ActionType.DOUBLE_CLICK,
                ActionToken.RIGHT_CLICK: ActionType.RIGHT_CLICK,
                ActionToken.DRAG: ActionType.DRAG,
                ActionToken.SCROLL: ActionType.SCROLL,
                ActionToken.TYPE: ActionType.TYPE,
                ActionToken.HOTKEY: ActionType.HOTKEY,
                ActionToken.WAIT: ActionType.WAIT,
                ActionToken.CANCEL: ActionType.CANCEL,
            }[kind]
        except (KeyError, ValueError) as error:
            raise ValueError("invalid action type token") from error
        if action_type in {ActionType.NOOP, ActionType.CANCEL}:
            if body:
                raise ValueError(f"{action_type.name} does not accept arguments")
            return ActionEvent(action_type)
        if action_type in {
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.DRAG,
        }:
            expected = 2 if action_type is not ActionType.DRAG else 4
            if len(body) != expected or any(
                not self.coordinate_base <= token < self.scroll_base for token in body
            ):
                raise ValueError(f"{action_type.name} requires {expected} coordinate tokens")
            coordinates = tuple((token - self.coordinate_base) / 255.0 for token in body)
            return ActionEvent(action_type, coordinates=coordinates)
        if action_type is ActionType.SCROLL:
            if len(body) != 2 or any(
                not self.scroll_base <= token < self.duration_base for token in body
            ):
                raise ValueError("SCROLL requires two delta tokens")
            values = tuple((token - self.scroll_base) / 127.5 - 1.0 for token in body)
            return ActionEvent(action_type, scroll_delta=values)  # type: ignore[arg-type]
        if action_type is ActionType.WAIT:
            if len(body) != 1 or not self.duration_base <= body[0] < self.byte_base:
                raise ValueError("WAIT requires one duration token")
            duration = (body[0] - self.duration_base) / 127.0 * self.max_duration_ms
            return ActionEvent(action_type, duration_ms=duration)
        if action_type is ActionType.TYPE:
            if any(not self.byte_base <= token < self.key_base for token in body):
                raise ValueError("TYPE contains a non-byte token")
            return ActionEvent(
                action_type, text=bytes(token - self.byte_base for token in body).decode("utf-8")
            )
        if action_type is ActionType.HOTKEY:
            if not body or any(not self.key_base <= token < self.vocab_size for token in body):
                raise ValueError("HOTKEY requires key tokens")
            return ActionEvent(action_type, keys=tuple(token - self.key_base for token in body))
        raise AssertionError("unreachable action type")
