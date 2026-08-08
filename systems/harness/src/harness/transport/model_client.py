from __future__ import annotations

from contracts import ActuationSignal, ObservationSignal


class ModelClient:
    """Minimal Harness-side interface for a physical signal model service."""

    def __init__(self, client) -> None:
        self.client = client

    def infer(self, observation: ObservationSignal) -> ActuationSignal:
        return self.client.infer(observation)
