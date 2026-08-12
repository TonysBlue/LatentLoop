from __future__ import annotations

from dataclasses import dataclass

from contracts import ActionFrame, ControlSignal, decode_action_frame


def action_frame_to_controls(
    frame: ActionFrame,
    *,
    event_id: str,
    pending_utf8: bytes = b"",
) -> tuple[tuple[ControlSignal, ...], bytes]:
    result = decode_action_frame(
        frame,
        event_id=event_id,
        pending_utf8=pending_utf8,
    )
    return result.controls, result.pending_utf8


@dataclass(slots=True)
class ActionFrameDecoder:
    """Decode and immediately release every complete control in one 80 ms frame."""

    pending_utf8: bytes = b""

    def push(
        self,
        frame: ActionFrame,
        *,
        event_id: str,
    ) -> tuple[ControlSignal, ...]:
        controls, self.pending_utf8 = action_frame_to_controls(
            frame,
            event_id=event_id,
            pending_utf8=self.pending_utf8,
        )
        return controls

    def reset(self) -> None:
        self.pending_utf8 = b""
