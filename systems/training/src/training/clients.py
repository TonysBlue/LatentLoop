from __future__ import annotations

from contracts import ActuationSignal, ObservationSignal


class ModelServiceClient:
    """Training-side policy interface; physical signal transport is injectable."""

    def __init__(self, infer):
        self._infer = infer

    def infer(self, observation: ObservationSignal) -> ActuationSignal:
        return self._infer(observation)
