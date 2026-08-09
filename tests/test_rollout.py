from __future__ import annotations

from training.rollout import Rollout


def test_physical_rollout_trace_records_group_identity() -> None:
    rollout = Rollout(rollout_id="r", group_id="g", task_id="task", seed=1)
    assert rollout.group_id == "g"
