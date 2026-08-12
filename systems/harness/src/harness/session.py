from __future__ import annotations

from dataclasses import dataclass

from contracts import ActuationSignal, ObservationSignal


@dataclass(slots=True)
class Session:
    session_id: str
    next_unit: int = 0

    def accept_observation(self, observation: ObservationSignal) -> None:
        if observation.session_id != self.session_id or observation.unit_index != self.next_unit:
            raise ValueError("observation session or unit is out of order")

    def accept_actuation(self, output: ActuationSignal) -> None:
        if output.session_id != self.session_id or output.unit_index != self.next_unit:
            raise ValueError("actuation session or unit is out of order")
        self.next_unit += 1
