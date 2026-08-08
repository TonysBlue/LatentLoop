from __future__ import annotations

from dataclasses import dataclass

from contracts.control import ControlKind, ControlSignal


@dataclass(frozen=True, slots=True)
class ActionDecodeResult:
    controls: tuple[ControlSignal, ...]
    consumed: int


def decode_action_tokens(tokens: list[int], *, event_id: str) -> ActionDecodeResult:
    """Decode canonical action tokens to physical ControlSignal events."""

    if not tokens or tokens[-1] != 1:
        raise ValueError("action sequence must be non-empty and end with END_ACTION")
    kind, body = tokens[0], tokens[1:-1]
    if kind in {2, 11}:
        if body:
            raise ValueError("NOOP and CANCEL do not accept arguments")
        event_kind = ControlKind.NOOP if kind == 2 else ControlKind.CANCEL
        return ActionDecodeResult((ControlSignal(event_kind, event_id),), len(tokens))
    if kind in {3, 4, 5, 6}:
        expected = 4 if kind == 6 else 2
        if len(body) != expected or any(token < 12 or token >= 268 for token in body):
            raise ValueError("pointer action has invalid coordinate tokens")
        coords = tuple((token - 12) / 255.0 for token in body)
        values: dict[str, float] = {"x": coords[0], "y": coords[1]}
        event_kind = ControlKind.POINTER_BUTTON
        if kind == 6:
            values.update(x2=coords[2], y2=coords[3])
            event_kind = ControlKind.POINTER_DRAG
        if kind in {3, 4, 5}:
            values["button"] = {3: 0, 4: 0, 5: 2}[kind]
        return ActionDecodeResult((ControlSignal(event_kind, event_id, **values),), len(tokens))
    if kind == 7:
        if len(body) != 2 or any(token < 268 or token >= 524 for token in body):
            raise ValueError("SCROLL action has invalid values")
        return ActionDecodeResult(
            (
                ControlSignal(
                    ControlKind.SCROLL,
                    event_id,
                    dx=(body[0] - 268) / 127.5 - 1,
                    dy=(body[1] - 268) / 127.5 - 1,
                ),
            ),
            len(tokens),
        )
    if kind == 8:
        if any(token < 652 or token >= 908 for token in body):
            raise ValueError("TYPE action has invalid bytes")
        return ActionDecodeResult(
            (
                ControlSignal(
                    ControlKind.TEXT_INPUT,
                    event_id,
                    text=bytes(token - 652 for token in body).decode("utf-8"),
                ),
            ),
            len(tokens),
        )
    if kind == 9:
        if not body or any(token < 908 or token >= 940 for token in body):
            raise ValueError("HOTKEY action has invalid keys")
        return ActionDecodeResult(
            tuple(
                ControlSignal(ControlKind.KEY_PRESS, f"{event_id}-{i}", key=token - 908)
                for i, token in enumerate(body)
            ),
            len(tokens),
        )
    if kind == 10:
        if len(body) != 1 or not 524 <= body[0] < 652:
            raise ValueError("WAIT action has invalid duration")
        return ActionDecodeResult(
            (
                ControlSignal(
                    ControlKind.WAIT,
                    event_id,
                    duration_ms=round((body[0] - 524) / 127 * 10_000),
                ),
            ),
            len(tokens),
        )
    raise ValueError("unknown action token kind")
