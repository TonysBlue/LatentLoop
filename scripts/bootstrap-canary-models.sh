#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${LATENTLOOP_MODEL_ROOT:-$HOME/latentloop-data/models}"
SOURCE_ROOT="${LATENTLOOP_VENDOR_ROOT:-$HOME/latentloop-data/vendor}"
COSY_MODEL="$MODEL_ROOT/CosyVoice2-0.5B"
SENSE_MODEL="$MODEL_ROOT/SenseVoiceSmall"
COSY_SOURCE="$SOURCE_ROOT/CosyVoice"

verify_assets() {
  local path expected
  while read -r expected path; do
    [[ -f "$path" ]] || return 1
    [[ "$(sha256sum "$path" | cut -d' ' -f1)" == "$expected" ]] || return 1
  done <<EOF
b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608 $COSY_MODEL/llm.pt
ff4c2f867674411e0a08cee702996df13fa67c1cd864c06108da88d16d088541 $COSY_MODEL/flow.pt
3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879 $COSY_MODEL/hift.pt
d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71 $COSY_MODEL/speech_tokenizer_v2.onnx
a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73 $COSY_MODEL/campplus.onnx
130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96 $COSY_MODEL/CosyVoice-BlankEN/model.safetensors
833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea $SENSE_MODEL/model.pt
f71e239ba36705564b5bf2d2ffd07eece07b8e3f2bbf6d2c99d8df856339ac19 $SENSE_MODEL/config.yaml
aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8 $SENSE_MODEL/chn_jpn_yue_eng_ko_spectok.bpe.model
29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5 $SENSE_MODEL/am.mvn
EOF
}

mkdir -p "$MODEL_ROOT" "$SOURCE_ROOT"
uv python install 3.10 3.11
uv sync --project "$REPO/tools/cosyvoice" --frozen
uv sync --project "$REPO/tools/asr" --frozen

if [[ ! -d "$COSY_SOURCE/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/FunAudioLLM/CosyVoice.git "$COSY_SOURCE"
fi
if [[ "$(git -C "$COSY_SOURCE" rev-parse HEAD 2>/dev/null || true)" != \
  "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc" ]]; then
  git -C "$COSY_SOURCE" fetch --depth 1 origin \
    074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
  git -C "$COSY_SOURCE" checkout --detach \
    074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
fi
if [[ ! -d "$COSY_SOURCE/third_party/Matcha-TTS/.git" ]]; then
  git clone --no-checkout https://github.com/shivammehta25/Matcha-TTS.git \
    "$COSY_SOURCE/third_party/Matcha-TTS"
fi
if [[ "$(git -C "$COSY_SOURCE/third_party/Matcha-TTS" rev-parse HEAD 2>/dev/null || true)" != \
  "dd9105b34bf2be2230f4aa1e4769fb586a3c824e" ]]; then
  git -C "$COSY_SOURCE/third_party/Matcha-TTS" fetch --depth 1 origin \
    dd9105b34bf2be2230f4aa1e4769fb586a3c824e
  git -C "$COSY_SOURCE/third_party/Matcha-TTS" checkout --detach \
    dd9105b34bf2be2230f4aa1e4769fb586a3c824e
fi

if ! verify_assets; then
  uvx --from huggingface-hub==0.36.0 huggingface-cli download \
    FunAudioLLM/CosyVoice2-0.5B \
    --revision eec1ae6c79877dbd9379285cf8789c9e0879293d \
    --local-dir "$COSY_MODEL"
  uvx --from huggingface-hub==0.36.0 huggingface-cli download \
    FunAudioLLM/SenseVoiceSmall \
    --revision 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7 \
    --local-dir "$SENSE_MODEL"
fi

verify_assets

printf 'Canary TTS and ASR models are ready under %s\n' "$MODEL_ROOT"
