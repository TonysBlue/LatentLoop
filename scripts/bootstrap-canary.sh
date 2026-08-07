#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${LATENTLOOP_STORAGE_ROOT:-$HOME/latentloop-data}"
ROOT="${LATENTLOOP_DATA_ROOT:-$STORAGE_ROOT/datasets}"
ASSET_ROOT="${LATENTLOOP_ASSET_ROOT:-$STORAGE_ROOT/assets}"
CACHE="$ASSET_ROOT/sources"
LOCK="$ROOT/registry/source-lock.json"
LICENSES="$ROOT/registry/licenses"

mkdir -p "$ROOT/registry" "$LICENSES" "$CACHE"

download() {
  local url="$1" output="$2" expected="$3"
  mkdir -p "$(dirname "$output")"
  if [[ -f "$output" ]] && [[ "$(sha256sum "$output" | cut -d' ' -f1)" == "$expected" ]]; then
    printf 'Verified %s\n' "$output"
    return
  fi
  curl --fail --location --retry 5 --retry-all-errors --continue-at - \
    --output "$output" "$url"
  local actual
  actual="$(sha256sum "$output" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Resume did not produce the expected SHA-256; retrying %s from zero.\n' \
      "$output" >&2
    rm -f "$output"
    curl --fail --location --retry 5 --retry-all-errors \
      --output "$output" "$url"
    actual="$(sha256sum "$output" | cut -d' ' -f1)"
  fi
  [[ "$actual" == "$expected" ]] || {
    printf 'SHA-256 mismatch for %s: expected %s, got %s\n' \
      "$output" "$expected" "$actual" >&2
    exit 2
  }
}

AISHELL1_REV="bbe295d530192a4cd41644b711c9aecd087df653"
LIBRISPEECH_REV="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
AISHELL4_REV="0cf6e53877b50888a93cc835d8d6beb08d3fb917"
DAILYTALK_REV="1f0d958a51aac5c6d203fa11bc8d1d453397006b"

AISHELL1_BASE="https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/$AISHELL1_REV"
LIBRISPEECH_BASE="https://huggingface.co/datasets/openslr/librispeech_asr/resolve/$LIBRISPEECH_REV"
AISHELL4_BASE="https://huggingface.co/datasets/argmaxinc/aishell-4/resolve/$AISHELL4_REV"
DAILYTALK_BASE="https://huggingface.co/datasets/DynamicSuperbPrivate/DialogueActClassification_DailyTalk/resolve/$DAILYTALK_REV"

download "$AISHELL1_BASE/data_aishell/wav/S0002.tar.gz" \
  "$CACHE/aishell1-small/S0002.tar.gz" \
  "5700ffa081f42c5a1be701147e680490fa494db959ca3c10ae205c3658af159f"
download "$AISHELL1_BASE/data_aishell/wav/S0003.tar.gz" \
  "$CACHE/aishell1-small/S0003.tar.gz" \
  "c15938c88bcdd0f5b2ad4b64862d2bdd0a577fdb966777aceb84572aea105517"
download "$AISHELL1_BASE/data_aishell/wav/S0004.tar.gz" \
  "$CACHE/aishell1-small/S0004.tar.gz" \
  "c9169b501da24ad2b91ffd802a06bf4ed145860b96dd088188be30465d7ef058"
download "$AISHELL1_BASE/data_aishell/transcript/aishell_transcript_v0.8.txt" \
  "$CACHE/aishell1-small/transcript.txt" \
  "b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4"
download "$LIBRISPEECH_BASE/all/train.clean.100/0000.parquet" \
  "$CACHE/librispeech.parquet" \
  "3098c6e44d1d49f8c62bd123775f1c492bda2cabd80f3ee70fe7800572371401"
download "$AISHELL4_BASE/data/test-00000-of-00011.parquet" \
  "$CACHE/aishell4.parquet" \
  "d56fd3500c00987fd4c4eef534941b7ab210c2dbc36e1585841f4a3ad2cbc570"
download "$AISHELL4_BASE/data/test-00001-of-00011.parquet" \
  "$CACHE/aishell4-1.parquet" \
  "69c0a0a95bfe025bdb788fcb64008fe088157003ac54158c7bc97be9a695cd4a"
download "$AISHELL4_BASE/data/test-00002-of-00011.parquet" \
  "$CACHE/aishell4-2.parquet" \
  "b67dffbc76adf3574eacbdcc01d5af0742623267b4828ae269d76dec36711986"
download "$DAILYTALK_BASE/data/train-00000-of-00010-fff2765a31d81131.parquet" \
  "$CACHE/dailytalk.parquet" \
  "68db922578f94a2e17d04656a7d3f61b61e730e3d8da1082ba152134026dde69"

download "$AISHELL1_BASE/README.md" "$LICENSES/aishell1.md" \
  "e082373580239e771028331709c53d3771ca93e9405ced6d4baac5c8ac6685fd"
download "$LIBRISPEECH_BASE/README.md" "$LICENSES/librispeech.md" \
  "549fb1bc160952bec75d51d188632d80ac10ceec8519588fa85687565a7c6af0"
download "$AISHELL4_BASE/README.md" "$LICENSES/aishell4.md" \
  "61f179be821d5c590e1d77ec0620476c6581a4bff28f9fe5e955898125efbb1e"
download "$DAILYTALK_BASE/README.md" "$LICENSES/dailytalk.md" \
  "ab5460cb0e68d63f0edd0794d92f3b368dc636f04af59ebdb10e97061c77e607"
download \
  "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/asset/zero_shot_prompt.wav" \
  "$ROOT/registry/voices/bootstrap/zero_shot_prompt.wav" \
  "c7b31d6dbe7cc6a716dded00550db5b50940bf209e424e4ad207b12e657c8ff6"

uv run --project "$REPO/tools/curation" python "$REPO/tools/curation/write_source_lock.py" \
  --root "$ROOT" --cache "$CACHE" --output "$LOCK"

printf 'Canary public sources and lock are ready: %s\n' "$LOCK"
