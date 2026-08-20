from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from contracts import ObservationSignal
from contracts.protocol import observation_to_payload


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    lineage_id: str
    session_id: str
    unit_index: int
    timestamp_ms: int
    payload: bytes
    payload_sha256: str
    previous_chain_sha256: str
    chain_sha256: str
    policy_version: str


class ObservationTimeline:
    """Append-only canonical ObservationSignal hash chain."""

    def __init__(
        self, lineage_id: str, session_id: str, root: str | Path | None = None
    ) -> None:
        if not lineage_id or not session_id:
            raise ValueError("timeline lineage and session identity are required")
        self.lineage_id = lineage_id
        self.session_id = session_id
        self._records: list[ObservationRecord] = []
        self.root = Path(root).expanduser().resolve() if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._manifest = self.root / "observations.jsonl"
            if self._manifest.exists():
                self._load()
        else:
            self._manifest = None

    @property
    def records(self) -> tuple[ObservationRecord, ...]:
        return tuple(self._records)

    @property
    def chain_sha256(self) -> str:
        return self._records[-1].chain_sha256 if self._records else "0" * 64

    def append(self, observation: ObservationSignal, policy_version: str) -> ObservationRecord:
        if observation.session_id != self.session_id:
            raise ValueError("timeline observation session does not match")
        if observation.unit_index != len(self._records):
            raise ValueError("timeline observation unit is not contiguous")
        if self._records and observation.timestamp_ms <= self._records[-1].timestamp_ms:
            raise ValueError("timeline observation timestamp is not increasing")
        if not policy_version:
            raise ValueError("timeline policy version is required")
        payload = observation_to_payload(observation)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        previous = self.chain_sha256
        chain = hashlib.sha256(bytes.fromhex(previous) + payload).hexdigest()
        record = ObservationRecord(
            self.lineage_id,
            self.session_id,
            observation.unit_index,
            observation.timestamp_ms,
            payload,
            payload_sha256,
            previous,
            chain,
            policy_version,
        )
        self._records.append(record)
        self._persist(record)
        return record

    def _persist(self, record: ObservationRecord) -> None:
        if self.root is None or self._manifest is None:
            return
        payload_path = self.root / f"unit-{record.unit_index:012d}.pb"
        if payload_path.exists():
            raise ValueError("timeline payload already exists")
        with payload_path.open("xb") as target:
            target.write(record.payload)
            target.flush()
            os.fsync(target.fileno())
        item = {
            "lineage_id": record.lineage_id,
            "session_id": record.session_id,
            "unit_index": record.unit_index,
            "timestamp_ms": record.timestamp_ms,
            "payload": payload_path.name,
            "payload_sha256": record.payload_sha256,
            "previous_chain_sha256": record.previous_chain_sha256,
            "chain_sha256": record.chain_sha256,
            "policy_version": record.policy_version,
        }
        with self._manifest.open("a", encoding="utf-8") as target:
            target.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
            target.flush()
            os.fsync(target.fileno())

    def _load(self) -> None:
        assert self.root is not None and self._manifest is not None
        for line in self._manifest.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            payload = (self.root / str(item["payload"])).read_bytes()
            record = ObservationRecord(
                str(item["lineage_id"]),
                str(item["session_id"]),
                int(item["unit_index"]),
                int(item["timestamp_ms"]),
                payload,
                str(item["payload_sha256"]),
                str(item["previous_chain_sha256"]),
                str(item["chain_sha256"]),
                str(item["policy_version"]),
            )
            if record.lineage_id != self.lineage_id or record.session_id != self.session_id:
                raise ValueError("persisted timeline identity does not match")
            expected_payload = hashlib.sha256(payload).hexdigest()
            previous = self._records[-1].chain_sha256 if self._records else "0" * 64
            expected_chain = hashlib.sha256(bytes.fromhex(previous) + payload).hexdigest()
            if (
                record.unit_index != len(self._records)
                or record.payload_sha256 != expected_payload
                or record.previous_chain_sha256 != previous
                or record.chain_sha256 != expected_chain
            ):
                raise ValueError("persisted timeline hash chain is invalid")
            self._records.append(record)


__all__ = ["ObservationRecord", "ObservationTimeline"]
