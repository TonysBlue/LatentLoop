from __future__ import annotations

import pytest
from model.action_tokens import ActionEvent, ActionTokenizer
from model.types import ActionType


@pytest.mark.parametrize(
    "event",
    [
        ActionEvent(ActionType.NOOP),
        ActionEvent(ActionType.CLICK, coordinates=(0.25, 0.75)),
        ActionEvent(ActionType.DOUBLE_CLICK, coordinates=(0.1, 0.2)),
        ActionEvent(ActionType.RIGHT_CLICK, coordinates=(0.3, 0.4)),
        ActionEvent(ActionType.DRAG, coordinates=(0.1, 0.2, 0.8, 0.9)),
        ActionEvent(ActionType.SCROLL, scroll_delta=(-0.5, 0.75)),
        ActionEvent(ActionType.TYPE, text="你好, LatentLoop"),
        ActionEvent(ActionType.HOTKEY, keys=(1, 7, 12)),
        ActionEvent(ActionType.WAIT, duration_ms=2_500),
        ActionEvent(ActionType.CANCEL),
    ],
)
def test_action_event_round_trip(event: ActionEvent) -> None:
    tokenizer = ActionTokenizer(max_duration_ms=10_000)
    decoded = tokenizer.decode(tokenizer.encode(event))
    assert decoded.type is event.type
    if event.text is not None:
        assert decoded.text == event.text
    if event.keys is not None:
        assert decoded.keys == event.keys


def test_long_action_is_split_into_bounded_bursts() -> None:
    tokenizer = ActionTokenizer(max_duration_ms=10_000, burst_tokens=16)
    tokens = tokenizer.encode(ActionEvent(ActionType.TYPE, text="x" * 40))
    bursts = tokenizer.split_bursts(tokens)
    assert len(bursts) > 1
    assert all(len(burst) <= 16 for burst in bursts)
    assert tokenizer.decode([token for burst in bursts for token in burst]).text == "x" * 40


def test_action_grammar_rejects_incomplete_click() -> None:
    tokenizer = ActionTokenizer(max_duration_ms=10_000)
    tokens = tokenizer.encode(ActionEvent(ActionType.CLICK, coordinates=(0.2, 0.4)))
    with pytest.raises(ValueError, match="CLICK"):
        tokenizer.decode(tokens[:-2] + [tokenizer.end_action])
