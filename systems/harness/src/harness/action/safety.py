from __future__ import annotations

from contracts import ControlKind, ControlSignal


class SafetyGate:
    def __init__(
        self,
        *,
        require_approval_for: set[ControlKind] | None = None,
    ) -> None:
        self.require_approval_for = require_approval_for or set()

    def validate(
        self,
        signal: ControlSignal,
        *,
        current_revision: int,
        approved: bool = False,
    ) -> None:
        if signal.screen_revision is not None and signal.screen_revision != current_revision:
            raise PermissionError("screen revision is stale")
        if signal.kind in self.require_approval_for and not approved:
            raise PermissionError(f"control requires approval: {signal.kind}")
        if signal.kind is ControlKind.TEXT_INPUT and not signal.text:
            raise ValueError("text input requires text")
        if signal.kind in {ControlKind.KEY_PRESS, ControlKind.KEY_RELEASE} and signal.key is None:
            raise ValueError("keyboard signal requires a key")
