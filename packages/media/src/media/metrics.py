"""Codec and streaming audio metrics owned by the Media package."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from runtime.codec_worker import CodecWorkerClient
from torch import Tensor


def codec_accuracy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("speech logits and targets have incompatible shapes")
    correct = logits.argmax(dim=-1).eq(targets)
    valid = mask[:, :, None].expand_as(correct)
    return (correct & valid).sum(dim=(0, 1)) / valid.sum(dim=(0, 1)).clamp_min(1)


def boundary_discontinuity_db(previous: Tensor, current: Tensor) -> float:
    if previous.numel() < 2 or current.numel() < 2:
        raise ValueError("waveform chunks must contain at least two samples")
    internal = torch.cat((previous.flatten(), current.flatten()))
    baseline = internal.diff().abs().quantile(0.95).clamp_min(1e-8)
    boundary = (current.flatten()[0] - previous.flatten()[-1]).abs().clamp_min(1e-8)
    return float((20 * torch.log10(boundary / baseline)).item())


@dataclass(frozen=True, slots=True)
class CodecBenchmark:
    frames: int
    elapsed_seconds: float
    rtf: float
    mean_frame_ms: float
    p95_frame_ms: float
    p95_boundary_db: float
    maximum_boundary_db: float


def benchmark_decoder(
    client: CodecWorkerClient, codes: Tensor, session_id: str = "benchmark"
) -> CodecBenchmark:
    if codes.ndim != 3 or codes.shape[1:] != (client.identity.codebooks, 1):
        raise ValueError("benchmark codes must have shape [frames, codebooks, 1]")
    client.reset(f"{session_id}-warmup", replay=False)
    for index in range(min(3, len(codes))):
        client.decode_step(codes[index : index + 1], f"{session_id}-warmup")
    client.reset(session_id, replay=False)
    latencies: list[float] = []
    boundaries: list[float] = []
    previous: Tensor | None = None
    started = time.perf_counter()
    for frame in codes:
        frame_started = time.perf_counter()
        waveform = client.decode_step(frame[None], session_id)
        latencies.append(time.perf_counter() - frame_started)
        if previous is not None:
            boundaries.append(boundary_discontinuity_db(previous, waveform))
        previous = waveform
    elapsed = time.perf_counter() - started
    measured = torch.tensor(latencies)
    duration = len(codes) / client.identity.frame_rate
    boundary_values = torch.tensor(boundaries) if boundaries else torch.zeros(1)
    return CodecBenchmark(
        frames=len(codes),
        elapsed_seconds=elapsed,
        rtf=elapsed / duration,
        mean_frame_ms=float(measured.mean().item() * 1_000),
        p95_frame_ms=float(measured.quantile(0.95).item() * 1_000),
        p95_boundary_db=float(boundary_values.quantile(0.95).item()),
        maximum_boundary_db=max(boundaries, default=0.0),
    )
