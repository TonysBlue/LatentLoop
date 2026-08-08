# Local training platform contract

This runbook implements the final architecture in
[latent-loop-architecture.md](latent-loop-architecture.md). It is intentionally
CLI-friendly and contains no legacy control or memory-probe protocol.

## Configuration

Use `configs/smoke.yaml` for a fast synthetic check and the shared training
entry point for local, Canary, Pilot and Production profiles. Production uses
`memory_horizon_units=750` and `tbptt_units=750`; smoke only shrinks dimensions
and the horizon while exercising the same code path.

## Data contract

Each 80 ms unit contains microphone audio, screen metadata, speech mode and
codec targets, and a padded unified action-token burst:

```text
speech_mode          [B]
speech_mode_mask     [B]
speech_codes         [B, 1, 8]
speech_codec_mask    [B, 1]
action_tokens        [B, <=16]
action_token_mask    [B, <=16]
```

WebDataset schema version is 3. Shards using `controls.npy`, structured action
JSON, memory targets or schema versions 1/2 are rejected and must be regenerated.

## Training and losses

Run `latentloop train` or `scripts/run-training.sh`. The only objective is:

```text
L_total = speech_weight * (mode_CE + speech_codec_CE)
        + action_weight * action_token_CE
```

Codec CE is masked out on SILENCE units. No auxiliary memory, diversity,
write-budget, control or regression loss is configured.

## Checkpoints and verification

Checkpoint format 4 includes model weights, optimizer/scheduler state, RNG,
cursor and all recurrent state (`Z`, `H`, KV, audio cache, speech-local and
action-local). Older formats are incompatible. Verify changes with:

```text
.venv/bin/pytest -q
.venv/bin/ruff check src tests tools/curation
.venv/bin/python -m compileall -q src tools/curation
git diff --check
```
