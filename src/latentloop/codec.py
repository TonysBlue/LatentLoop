from __future__ import annotations

from typing import Protocol

from torch import Tensor


class FrozenNeuralCodec(Protocol):
    """Boundary for a separately versioned, frozen causal neural codec."""

    sample_rate: int
    frame_rate: int
    codebooks: int
    codebook_size: int

    def encode(self, waveform: Tensor) -> Tensor: ...

    def decode(self, codes: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]: ...


def validate_codec_codes(
    codes: Tensor,
    *,
    frames: int,
    codebooks: int,
    codebook_size: int,
) -> None:
    if codes.shape[-2:] != (frames, codebooks):
        raise ValueError(f"codec codes must end in [{frames}, {codebooks}]")
    if codes.min().item() < 0 or codes.max().item() >= codebook_size:
        raise ValueError("codec code is outside the configured codebook")


def codec_frame_bounds(start_ms: int, delta_ms: int, frame_rate: int) -> tuple[int, int]:
    if start_ms < 0 or delta_ms <= 0 or frame_rate <= 0:
        raise ValueError("codec frame timing values must be positive")
    start = start_ms * frame_rate // 1_000
    end = (start_ms + delta_ms) * frame_rate // 1_000
    return start, end


def codec_frame_mask(start_ms: int, delta_ms: int, frame_rate: int, maximum_frames: int) -> Tensor:
    import torch

    start, end = codec_frame_bounds(start_ms, delta_ms, frame_rate)
    frame_count = end - start
    if frame_count > maximum_frames:
        raise ValueError("unit contains more codec frames than the configured tensor")
    mask = torch.zeros(maximum_frames, dtype=torch.bool)
    mask[:frame_count] = True
    return mask
