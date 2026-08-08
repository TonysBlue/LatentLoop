"""Shared data domain facade used by training and Harness capture."""

from data.migrate import migrate_manifest_v4_to_v5, migrate_metadata_v4_to_v5
from latentloop.data import (
    EpisodeShardReader,
    SyntheticEpisodeDataset,
    encode_target_speech,
    import_speech_manifest,
    write_episode_shards,
)

__all__ = [
    "EpisodeShardReader",
    "SyntheticEpisodeDataset",
    "encode_target_speech",
    "import_speech_manifest",
    "write_episode_shards",
    "migrate_manifest_v4_to_v5",
    "migrate_metadata_v4_to_v5",
]
