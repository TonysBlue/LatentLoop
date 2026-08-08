from __future__ import annotations

from contracts import ActuationSignal, EnvironmentReceipt, ObservationSignal, RewardBreakdown

from harness.action.executor import ControlExecutor
from harness.session import Session


class HarnessService:
    """Coordinates sensors, Model Service client, actuators, and environment."""

    def __init__(self, environment, model_client, executor: ControlExecutor) -> None:
        self.environment = environment
        self.model_client = model_client
        self.executor = executor
        self.session: Session | None = None

    def reset(self, task_id: str, seed: int, session_id: str) -> ObservationSignal:
        try:
            observation = self.environment.reset(task_id, seed, session_id)
        except TypeError:
            # Test-only in-process adapters may still expose the minimal
            # two-argument backend protocol; formal QEMU uses the identity-aware form.
            observation = self.environment.reset(task_id, seed)
        if observation.session_id != session_id:
            raise ValueError("environment observation has wrong session identity")
        self.session = Session(session_id)
        self.session.accept_observation(observation)
        if hasattr(self.model_client, "open_session"):
            self.model_client.open_session(session_id)
        return observation

    def step(
        self,
        observation: ObservationSignal,
    ) -> tuple[ObservationSignal, EnvironmentReceipt, ActuationSignal]:
        if self.session is None:
            raise RuntimeError("Harness session is not active")
        self.session.accept_observation(observation)
        output = self.model_client.infer(observation)
        for control in output.controls:
            self.executor.gate.validate(control, current_revision=observation.screen.revision)
        next_observation, receipt = self.environment.apply(
            output, current_revision=observation.screen.revision
        )
        self.session.accept_actuation(output)
        return next_observation, receipt, output

    def evaluate(self, task_id: str) -> RewardBreakdown:
        return self.environment.evaluate(task_id)

    def close(self) -> None:
        if self.session is not None and hasattr(self.model_client, "close_session"):
            self.model_client.close_session(self.session.session_id)
        self.environment.close()
        self.session = None
