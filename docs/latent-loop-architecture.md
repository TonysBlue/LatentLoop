# LatentLoop final architecture

This document is normative for the v0.2 protocol. The previous multi-control,
memory-probe and structured-action design is incompatible and must not be
loaded at runtime.

## State transition

At the beginning of unit `t`, the recurrent state is:

```text
Z_(t-1)       fixed-size long-term latent memory
KV_(t-1)      bounded per-layer Transformer cache
H_(t-1)       complete final-normalized hidden from the previous unit
audio_cache   streaming audio encoder cache
speech_local  previous codec frame and temporal state
action_local  unfinished action decoder state
```

The complete forward step is:

```text
E_t       = InputEncoder(U_t, audio_cache_(t-1))
Z_t       = MemoryUpdater(Z_(t-1), H_(t-1))
H_t, KV_t = Backbone(E_t, KV_(t-1), Z_t)
speech_t  = SpeechHead(H_t, speech_local_(t-1))
action_t  = ActionHead(H_t, action_local_(t-1))
```

`H_t` is the complete `[B, tokens_per_unit, model_dim]` output after
`final_norm`. It is stored in the state and is the only hidden input to the
next memory update. There is no `r_t` or `r_(t-1)` summary. `Z_0` and `H_0`
are zero; KV starts empty. Learned slot identity breaks symmetry between
zero-initialized memory slots. Memory affects heads only through Backbone
latent cross-attention.

The production context is 750 units (60 seconds). KV eviction is bounded.
State is detached only at an episode boundary or after a configured TBPTT
segment; production TBPTT is 750 units so future output losses can supervise
the memory updater across the complete long-memory horizon.

## Output spaces

There are exactly two output heads:

Speech head:

```text
mode: SILENCE or SPEECH, one decision every 80 ms
codec: one Mimi frame when mode=SPEECH, 8 codebooks x 2048 classes
```

SILENCE contributes no codec loss and does not invoke codec decoding.

Action head:

```text
one unified discrete vocabulary
action kind: NOOP, CLICK, DOUBLE_CLICK, RIGHT_CLICK, DRAG, SCROLL,
             TYPE, HOTKEY, WAIT, CANCEL
coordinates: 256 bins
scroll: 256 bins
duration: 128 bins
TYPE payload: UTF-8 byte tokens
HOTKEY payload: 32 key tokens
burst: at most 16 tokens per 80 ms unit
```

An event ends with `END_ACTION`. Events that cross a unit boundary continue
through `action_local`; PAD after an end is masked. No structured action
regression head, confidence head or separate control head exists.

## Loss and supervision

```text
L_total = speech_loss_weight * (L_speech_mode + L_speech_codec)
        + action_loss_weight * L_action_token
```

`L_speech_mode` uses `speech_mode_mask`. `L_speech_codec` uses
`speech_codec_mask AND (speech_mode == SPEECH)`. `L_action_token` uses the
valid action-token mask. There is no memory target, write-budget loss,
diversity loss, probe loss or speech-control loss. Gradients from current and
future speech/action losses reach `MemoryUpdater` through `Z_t -> Backbone ->
H_t`.

## Data and checkpoint contracts

Schema version is 3. A unit stores `speech_mode`, `speech_mode_mask`,
`speech_codes`, `speech_codec_mask`, `action_tokens` and `action_token_mask`.
WebDataset samples contain the timeline, codec codes, audio and screens; old
`controls.npy`, structured action JSON and memory labels are rejected.

Checkpoint format is 4 and stores model weights plus `latent`, `hidden`,
`layer_kv`, `audio_cache`, `speech_local`, `action_local` and cursor state.
Formats 1--3 are explicitly unsupported.

## Test design

Required tests cover state timing (`Z_t` reads only previous `H`, current `H`
is saved), learned slot asymmetry, gradient flow from future output loss,
speech silence masking, action quantization/grammar/burst continuation, v3
WebDataset round-trip and v4 checkpoint restoration/rejection. Smoke tests may
shrink dimensions and horizon but use the same code path and cannot claim
60-second memory validation.
