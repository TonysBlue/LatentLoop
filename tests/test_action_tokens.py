from __future__ import annotations

import pytest
import torch
from contracts import (
    ActionFrame,
    ActionKind,
    ButtonPhase,
    ControlKind,
    PointerButton,
    PointerButtonPhase,
    decode_action_frame,
)
from data import SyntheticEpisodeDataset
from model.types import ActionFrame as TensorActionFrame
from runtime.action import ActionFrameDecoder
from runtime.config import load_config


def test_structured_action_frames_decode_without_end_token() -> None:
    move = decode_action_frame(
        ActionFrame(
            ActionKind.POINTER_MOVE,
            coordinate_cell=31 * 32 + 8,
            coordinate_residual=(0.5, 1.0),
        ),
        event_id="move",
        screen_revision=7,
    )
    assert move.controls[0].kind is ControlKind.POINTER_MOVE
    assert move.controls[0].x == pytest.approx(8.5 / 32)
    assert move.controls[0].y == 1.0
    assert move.controls[0].screen_revision == 7

    button = decode_action_frame(
        ActionFrame(
            ActionKind.POINTER_BUTTON,
            button=PointerButton.RIGHT,
            button_phase=PointerButtonPhase.DOWN,
        ),
        event_id="button",
    ).controls[0]
    assert button.button == int(PointerButton.RIGHT)
    assert button.button_phase is ButtonPhase.DOWN


def test_hotkey_decodes_to_ordered_press_and_release() -> None:
    controls = decode_action_frame(
        ActionFrame(ActionKind.HOTKEY, hotkey_keys=(1, 7, 12)), event_id="keys"
    ).controls
    assert [control.kind for control in controls] == [
        ControlKind.KEY_PRESS,
        ControlKind.KEY_PRESS,
        ControlKind.KEY_PRESS,
        ControlKind.KEY_RELEASE,
        ControlKind.KEY_RELEASE,
        ControlKind.KEY_RELEASE,
    ]
    assert [control.key for control in controls] == [1, 7, 12, 12, 7, 1]


def test_type_executes_complete_utf8_prefix_each_unit() -> None:
    decoder = ActionFrameDecoder()
    first = decoder.push(
        ActionFrame(ActionKind.TYPE, text_bytes=b"ok\xe4\xbd"), event_id="type-0"
    )
    assert [control.text for control in first] == ["ok"]
    second = decoder.push(
        ActionFrame(ActionKind.TYPE, text_bytes=b"\xa0!"), event_id="type-1"
    )
    assert [control.text for control in second] == ["你!"]


def test_incomplete_type_cannot_switch_kind() -> None:
    decoder = ActionFrameDecoder()
    decoder.push(ActionFrame(ActionKind.TYPE, text_bytes=b"\xf0\x9f"), event_id="type")
    with pytest.raises(ValueError, match="incomplete UTF-8"):
        decoder.push(ActionFrame(ActionKind.NO_ACTION), event_id="wait")


def test_macro_and_irrelevant_parameters_are_absent() -> None:
    assert {kind.name for kind in ActionKind} == {
        "NO_ACTION",
        "NOOP",
        "POINTER_MOVE",
        "POINTER_BUTTON",
        "SCROLL",
        "TYPE",
        "HOTKEY",
    }
    with pytest.raises(ValueError, match="only TYPE"):
        ActionFrame(ActionKind.NO_ACTION, text_bytes=b"x")


def test_episode_validates_utf8_continuation_on_the_unit_timeline() -> None:
    config = load_config("configs/smoke.yaml")
    episode = SyntheticEpisodeDataset(config.data, config.model).make_episode(0)
    first = TensorActionFrame.no_action(1)
    first.kind[0] = int(ActionKind.TYPE)
    first.text_bytes[0, :2] = torch.tensor((0xF0, 0x9F))
    first.text_length[0] = 2
    second = TensorActionFrame.no_action(1)
    second.kind[0] = int(ActionKind.TYPE)
    second.text_bytes[0, :2] = torch.tensor((0x98, 0x80))
    second.text_length[0] = 2
    episode.units[0].action = first
    episode.units[1].action = second
    episode.validate(
        audio_samples=config.data.unit_audio_samples,
        speech_frames=config.model.speech_frames_per_unit,
        speech_codebooks=config.model.speech_codebooks,
    )

    episode.units[1].action = TensorActionFrame.no_action(1)
    with pytest.raises(ValueError, match="incomplete UTF-8"):
        episode.validate(
            audio_samples=config.data.unit_audio_samples,
            speech_frames=config.model.speech_frames_per_unit,
            speech_codebooks=config.model.speech_codebooks,
        )
