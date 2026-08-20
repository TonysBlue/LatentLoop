from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicySampleRecord:
    lineage_id: str
    session_id: str
    unit_index: int
    policy_version: str
    observation_payload_sha256: str
    sample: dict[str, Any]
    sample_sha256: str
    previous_chain_sha256: str
    chain_sha256: str


class PolicySampleTrace:
    """Append-only sampled-output trace kept separate from perceptual observations."""

    def __init__(self, lineage_id: str, session_id: str, root: str | Path) -> None:
        if not lineage_id or not session_id:
            raise ValueError("policy trace lineage and session identity are required")
        self.lineage_id = lineage_id
        self.session_id = session_id
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = self.root / "policy-samples.jsonl"
        self._records: list[PolicySampleRecord] = []
        if self._manifest.exists():
            self._load()

    @property
    def records(self) -> tuple[PolicySampleRecord, ...]:
        return tuple(self._records)

    @property
    def chain_sha256(self) -> str:
        return self._records[-1].chain_sha256 if self._records else "0" * 64

    def append(
        self,
        *,
        unit_index: int,
        policy_version: str,
        observation_payload_sha256: str,
        sample: dict[str, Any],
    ) -> PolicySampleRecord:
        if unit_index != len(self._records):
            raise ValueError("policy sample trace unit is not contiguous")
        if not policy_version or len(observation_payload_sha256) != 64:
            raise ValueError("policy sample trace identity is incomplete")
        sample_payload = json.dumps(
            sample, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        sample_sha256 = hashlib.sha256(sample_payload).hexdigest()
        previous = self.chain_sha256
        identity = json.dumps(
            {
                "lineage_id": self.lineage_id,
                "session_id": self.session_id,
                "unit_index": unit_index,
                "policy_version": policy_version,
                "observation_payload_sha256": observation_payload_sha256,
                "sample_sha256": sample_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        chain = hashlib.sha256(bytes.fromhex(previous) + identity).hexdigest()
        record = PolicySampleRecord(
            self.lineage_id,
            self.session_id,
            unit_index,
            policy_version,
            observation_payload_sha256,
            sample,
            sample_sha256,
            previous,
            chain,
        )
        with self._manifest.open("a", encoding="utf-8") as target:
            target.write(
                json.dumps(
                    {
                        "lineage_id": record.lineage_id,
                        "session_id": record.session_id,
                        "unit_index": record.unit_index,
                        "policy_version": record.policy_version,
                        "observation_payload_sha256": record.observation_payload_sha256,
                        "sample": record.sample,
                        "sample_sha256": record.sample_sha256,
                        "previous_chain_sha256": record.previous_chain_sha256,
                        "chain_sha256": record.chain_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            target.flush()
            os.fsync(target.fileno())
        self._records.append(record)
        return record

    def _load(self) -> None:
        for line in self._manifest.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            sample = item["sample"]
            sample_payload = json.dumps(
                sample, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            sample_sha256 = hashlib.sha256(sample_payload).hexdigest()
            previous = self.chain_sha256
            identity = json.dumps(
                {
                    "lineage_id": item["lineage_id"],
                    "session_id": item["session_id"],
                    "unit_index": item["unit_index"],
                    "policy_version": item["policy_version"],
                    "observation_payload_sha256": item["observation_payload_sha256"],
                    "sample_sha256": item["sample_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            chain = hashlib.sha256(bytes.fromhex(previous) + identity).hexdigest()
            record = PolicySampleRecord(
                str(item["lineage_id"]),
                str(item["session_id"]),
                int(item["unit_index"]),
                str(item["policy_version"]),
                str(item["observation_payload_sha256"]),
                sample,
                str(item["sample_sha256"]),
                str(item["previous_chain_sha256"]),
                str(item["chain_sha256"]),
            )
            if (
                record.lineage_id != self.lineage_id
                or record.session_id != self.session_id
                or record.unit_index != len(self._records)
                or record.sample_sha256 != sample_sha256
                or record.previous_chain_sha256 != previous
                or record.chain_sha256 != chain
            ):
                raise ValueError("persisted policy sample hash chain is invalid")
            self._records.append(record)


__all__ = ["PolicySampleRecord", "PolicySampleTrace"]
