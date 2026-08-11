from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProtocolIdentity:
    protocol_version: str = "realtime-v1"
    action_schema_id: str = "structured-action-v1"
    codec_id: str = "mimi-24khz-8x2048"

    def __post_init__(self) -> None:
        if not all((self.protocol_version, self.action_schema_id, self.codec_id)):
            raise ValueError("protocol identity fields are required")
