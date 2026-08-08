from __future__ import annotations

import json

from data.migrate import migrate_manifest_v4_to_v5, migrate_metadata_v4_to_v5


def test_v4_migration_is_explicit_and_records_runtime_identity(tmp_path) -> None:
    metadata = migrate_metadata_v4_to_v5(
        {
            "schema_version": 4,
            "environment_id": "recorded",
            "environment_version": "1",
        }
    )
    assert metadata["schema_version"] == 5
    assert metadata["runtime_identity"]["protocol_version"] == "realtime-v1"

    source = tmp_path / "v4.jsonl"
    destination = tmp_path / "v5.jsonl"
    source.write_text(json.dumps({"schema_version": 4}) + "\n", encoding="utf-8")
    migrate_manifest_v4_to_v5(source, destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == 5
