# Realtime multimodal LatentLoop

This filename is retained as a research-note entry point. The former v0.1
document described separate speech/action/cognitive controls and auxiliary
memory objectives; that protocol is retired and incompatible.

The normative design is documented in
[latent-loop-architecture.md](latent-loop-architecture.md), including:

```text
Z_t       = MemoryUpdater(Z_(t-1), H_(t-1))
H_t, KV_t = Backbone(E_t, KV_(t-1), Z_t)
speech_t  = SpeechHead(H_t)
action_t  = ActionHead(H_t)
```

There are exactly two output heads, schema version 3, checkpoint format 4, and
the single weighted Speech/Action loss described in the normative document.
