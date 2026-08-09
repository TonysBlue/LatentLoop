from __future__ import annotations

from dataclasses import dataclass, field

from contracts import ControlSignal, decode_action_tokens


def action_tokens_to_controls(tokens: list[int], *, event_id: str) -> tuple[ControlSignal, ...]:
    return decode_action_tokens(tokens, event_id=event_id).controls


@dataclass(slots=True)
class ActionStreamDecoder:
    """Assemble one canonical action that may span multiple 80 ms bursts."""

    max_tokens: int = 4096
    _tokens: list[int] = field(default_factory=list)

    def push(self, tokens: list[int], *, event_id: str) -> tuple[ControlSignal, ...]:
        for token in tokens:
            if token == 0 and not self._tokens:
                continue
            if token == 0:
                self.reset()
                raise ValueError("PAD cannot appear inside an active action")
            self._tokens.append(int(token))
            if len(self._tokens) > self.max_tokens:
                self.reset()
                raise ValueError("action stream exceeds its maximum length")
            if token == 1:
                complete, self._tokens = self._tokens, []
                return action_tokens_to_controls(complete, event_id=event_id)
        return ()

    def reset(self) -> None:
        self._tokens.clear()
