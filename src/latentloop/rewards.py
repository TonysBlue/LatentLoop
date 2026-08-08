from __future__ import annotations

from latentloop.environment import RewardBreakdown

REWARD_SPEC_ID = "realtime-v1"


def total_reward(breakdown: RewardBreakdown) -> float:
    if breakdown.spec_id != REWARD_SPEC_ID:
        raise ValueError(f"unsupported reward spec: {breakdown.spec_id}")
    return breakdown.total
