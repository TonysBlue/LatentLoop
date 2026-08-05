from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def archive(url: str, filename: str, sha256: str) -> dict[str, str]:
    return {"source_url": url, "filename": filename, "archive_sha256": sha256}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    licenses = root / "raw" / "license-records"
    revisions = {
        "aishell1": "bbe295d530192a4cd41644b711c9aecd087df653",
        "librispeech": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        "aishell4": "0cf6e53877b50888a93cc835d8d6beb08d3fb917",
        "dailytalk": "1f0d958a51aac5c6d203fa11bc8d1d453397006b",
    }
    bases = {
        "aishell1": (
            "https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/"
            + revisions["aishell1"]
        ),
        "librispeech": (
            "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/"
            + revisions["librispeech"]
        ),
        "aishell4": (
            "https://huggingface.co/datasets/argmaxinc/aishell-4/resolve/"
            + revisions["aishell4"]
        ),
        "dailytalk": (
            "https://huggingface.co/datasets/"
            "DynamicSuperbPrivate/DialogueActClassification_DailyTalk/resolve/"
            + revisions["dailytalk"]
        ),
    }
    records = {
        "aishell1": {
            "source_version": "hf-AISHELL-1@bbe295d5",
            "license_path": str(licenses / "aishell1.md"),
            "license_sha256": "e082373580239e771028331709c53d3771ca93e9405ced6d4baac5c8ac6685fd",
            "archives": [
                archive(
                    f"{bases['aishell1']}/data_aishell/wav/S0002.tar.gz",
                    "S0002.tar.gz",
                    "5700ffa081f42c5a1be701147e680490fa494db959ca3c10ae205c3658af159f",
                ),
                archive(
                    f"{bases['aishell1']}/data_aishell/wav/S0003.tar.gz",
                    "S0003.tar.gz",
                    "c15938c88bcdd0f5b2ad4b64862d2bdd0a577fdb966777aceb84572aea105517",
                ),
                archive(
                    f"{bases['aishell1']}/data_aishell/wav/S0004.tar.gz",
                    "S0004.tar.gz",
                    "c9169b501da24ad2b91ffd802a06bf4ed145860b96dd088188be30465d7ef058",
                ),
                archive(
                    f"{bases['aishell1']}/data_aishell/transcript/aishell_transcript_v0.8.txt",
                    "transcript.txt",
                    "b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4",
                ),
            ],
        },
        "librispeech_train_clean_100": {
            "source_version": "hf-librispeech_asr@71cacbfb",
            "license_path": str(licenses / "librispeech.md"),
            "license_sha256": "549fb1bc160952bec75d51d188632d80ac10ceec8519588fa85687565a7c6af0",
            "archives": [
                archive(
                    f"{bases['librispeech']}/all/train.clean.100/0000.parquet",
                    "librispeech.parquet",
                    "3098c6e44d1d49f8c62bd123775f1c492bda2cabd80f3ee70fe7800572371401",
                )
            ],
        },
        "aishell4_train_l": {
            "source_version": "hf-argmaxinc-aishell-4@0cf6e538",
            "license_path": str(licenses / "aishell4.md"),
            "license_sha256": "61f179be821d5c590e1d77ec0620476c6581a4bff28f9fe5e955898125efbb1e",
            "archives": [
                archive(
                    f"{bases['aishell4']}/data/test-00000-of-00011.parquet",
                    "aishell4.parquet",
                    "d56fd3500c00987fd4c4eef534941b7ab210c2dbc36e1585841f4a3ad2cbc570",
                ),
                archive(
                    f"{bases['aishell4']}/data/test-00001-of-00011.parquet",
                    "aishell4-1.parquet",
                    "69c0a0a95bfe025bdb788fcb64008fe088157003ac54158c7bc97be9a695cd4a",
                ),
                archive(
                    f"{bases['aishell4']}/data/test-00002-of-00011.parquet",
                    "aishell4-2.parquet",
                    "b67dffbc76adf3574eacbdcc01d5af0742623267b4828ae269d76dec36711986",
                ),
            ],
        },
        "dailytalk": {
            "source_version": "hf-DailyTalk@1f0d958a",
            "license_path": str(licenses / "dailytalk.md"),
            "license_sha256": "ab5460cb0e68d63f0edd0794d92f3b368dc636f04af59ebdb10e97061c77e607",
            "archives": [
                archive(
                    f"{bases['dailytalk']}/data/train-00000-of-00010-fff2765a31d81131.parquet",
                    "dailytalk.parquet",
                    "68db922578f94a2e17d04656a7d3f61b61e730e3d8da1082ba152134026dde69",
                )
            ],
        },
    }
    source_files = {
        "aishell1": cache / "aishell1-small",
        "librispeech_train_clean_100": cache,
        "aishell4_train_l": cache,
        "dailytalk": cache,
    }
    for source_id, record in records.items():
        destination = root / "raw" / source_id
        destination.mkdir(parents=True, exist_ok=True)
        for item in record["archives"]:
            source = source_files[source_id] / item["filename"]
            target = destination / item["filename"]
            if target.is_symlink():
                target.unlink()
            if not target.exists():
                shutil.copy2(source, target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
