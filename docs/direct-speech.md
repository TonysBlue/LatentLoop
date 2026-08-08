# Direct speech protocol

Speech is an independent output head. Every 80 ms it predicts `SILENCE` or
`SPEECH`; only SPEECH predicts and decodes one eight-codebook Mimi frame.
SILENCE contributes only mode cross-entropy and outputs a zero-duration audio
unit. There is no SpeechControl state machine or separate control loss.

Training uses codec teacher forcing and causal codebook factorization. Runtime
uses greedy or configured sampling. The direct-speech overfit gate evaluates
mode accuracy and per-codebook codec accuracy through the same streaming model
used by other profiles.

Speech import manifests may include assistant target segments for deriving the
binary mode label. The v3 WebDataset stores `speech_mode`, mode mask, codec
codes and codec mask; old controls files are incompatible.
