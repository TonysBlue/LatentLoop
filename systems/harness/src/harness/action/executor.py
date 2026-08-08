from __future__ import annotations

from contracts import ActuationSignal, EnvironmentReceipt
from harness.action.safety import SafetyGate


class ControlExecutor:
    """Validated execution adapter; OS/VM injection is supplied by deployment."""

    def __init__(self, gate: SafetyGate, backend) -> None:
        self.gate = gate
        self.backend = backend

    def apply(
        self,
        output: ActuationSignal,
        *,
        current_revision: int = 0,
        approved: bool = False,
    ) -> EnvironmentReceipt:
        try:
            for signal in output.controls:
                self.gate.validate(signal, current_revision=current_revision, approved=approved)
            if hasattr(self.backend, "apply"):
                self.backend.apply(output)
            else:
                for signal in output.controls:
                    self.backend.execute(signal)
        except (PermissionError, ValueError) as error:
            return EnvironmentReceipt(
                output.session_id,
                output.unit_index,
                False,
                safety_violation=str(error),
            )
        return EnvironmentReceipt(output.session_id, output.unit_index, True)

    def close(self) -> None:
        if hasattr(self.backend, "close"):
            self.backend.close()
