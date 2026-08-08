from __future__ import annotations

from contracts import ControlSignal, decode_action_tokens


def action_tokens_to_controls(tokens: list[int], *, event_id: str) -> tuple[ControlSignal, ...]:
    return decode_action_tokens(tokens, event_id=event_id).controls
