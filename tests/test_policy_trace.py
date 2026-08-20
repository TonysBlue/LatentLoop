from __future__ import annotations

import json

import pytest
from data import PolicySampleTrace


def test_policy_sample_trace_is_separate_contiguous_hash_chain(tmp_path) -> None:
    trace = PolicySampleTrace("life", "session", tmp_path)
    first = trace.append(
        unit_index=0,
        policy_version="policy-1",
        observation_payload_sha256="a" * 64,
        sample={"speech_mode": 0, "speech_codes": [[0]], "action": {"kind": 0}},
    )
    second = trace.append(
        unit_index=1,
        policy_version="policy-1",
        observation_payload_sha256="b" * 64,
        sample={"speech_mode": 1, "speech_codes": [[3]], "action": {"kind": 0}},
    )
    assert second.previous_chain_sha256 == first.chain_sha256
    assert PolicySampleTrace("life", "session", tmp_path).records == trace.records
    with pytest.raises(ValueError, match="contiguous"):
        trace.append(
            unit_index=3,
            policy_version="policy-1",
            observation_payload_sha256="c" * 64,
            sample={"speech_mode": 0},
        )


def test_policy_sample_trace_rejects_tampering(tmp_path) -> None:
    trace = PolicySampleTrace("life", "session", tmp_path)
    trace.append(
        unit_index=0,
        policy_version="policy-1",
        observation_payload_sha256="a" * 64,
        sample={"speech_mode": 0},
    )
    path = tmp_path / "policy-samples.jsonl"
    item = json.loads(path.read_text(encoding="utf-8"))
    item["sample"]["speech_mode"] = 1
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        PolicySampleTrace("life", "session", tmp_path)
