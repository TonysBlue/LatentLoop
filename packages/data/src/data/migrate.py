"""Explicit trajectory metadata migration.

There is deliberately no implicit v4 fallback in readers.  This utility
creates a new v5 manifest and records the migration source hash so the result
can be audited before it is used for training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data.schema import SCHEMA_VERSION


def migrate_metadata_v4_to_v5(metadata: dict[str, Any]) -> dict[str, Any]:
    if int(metadata.get("schema_version", -1)) != 4:
        raise ValueError("only schema v4 metadata can be explicitly migrated")
    migrated = dict(metadata)
    migrated.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": migrated.get("protocol_version", "realtime-v1"),
            "action_vocabulary_id": migrated.get("action_vocabulary_id", "unified-action-v4"),
            "runtime_identity": {
                "protocol_version": migrated.get("protocol_version", "realtime-v1"),
                "environment_id": migrated.get("environment_id", "recorded"),
                "environment_version": migrated.get("environment_version", "1"),
                "action_vocabulary_id": migrated.get(
                    "action_vocabulary_id", "unified-action-v4"
                ),
            },
        }
    )
    return migrated


def migrate_manifest_v4_to_v5(source: str | Path, destination: str | Path) -> str:
    source_path, destination_path = Path(source), Path(destination)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open(encoding="utf-8") as input_file, destination_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue
            item = migrate_metadata_v4_to_v5(json.loads(line))
            output_file.write(json.dumps(item, sort_keys=True) + "\n")
    return digest
