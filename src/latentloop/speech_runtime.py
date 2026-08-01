from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from latentloop.codec_worker import CodecWorkerClient
from latentloop.types import SpeechControl


@dataclass(slots=True)
class SpeechRuntimeResult:
    waveform: Tensor
    decode_seconds: float
    submitted: bool


class DirectSpeechRuntime:
    """Turns generated codec frames into playback-ready 80 ms waveform chunks."""

    def __init__(self, client: CodecWorkerClient, session_id: str) -> None:
        self.client = client
        self.session_id = session_id
        self.active = False
        self.client.reset(session_id, replay=False)

    def reset(self) -> None:
        self.active = False
        self.client.reset(self.session_id, replay=False)

    def decode(self, codes: Tensor, control: Tensor) -> SpeechRuntimeResult:
        if control.numel() != 1:
            raise ValueError("speech runtime currently accepts one session at a time")
        selected = SpeechControl(int(control.item()))
        if selected is SpeechControl.SILENT:
            self.active = False
            return SpeechRuntimeResult(torch.zeros(1, 1, 1_920), 0.0, False)
        if selected is SpeechControl.START:
            self.client.reset(self.session_id, replay=False)
            self.active = True
        elif not self.active:
            raise RuntimeError("speech CONTINUE/PAUSE/STOP received without START")
        started = time.perf_counter()
        waveform = self.client.decode_step(codes.transpose(1, 2), self.session_id)
        elapsed = time.perf_counter() - started
        if selected is SpeechControl.STOP:
            self.active = False
        return SpeechRuntimeResult(waveform, elapsed, True)


class PlaybackQueue:
    def __init__(self, maximum_chunks: int = 4) -> None:
        self.maximum_chunks = maximum_chunks
        self._chunks: deque[Tensor] = deque()

    def push(self, chunk: Tensor) -> None:
        if len(self._chunks) >= self.maximum_chunks:
            raise RuntimeError("playback queue exceeded its bounded capacity")
        self._chunks.append(chunk)

    def pop(self) -> Tensor | None:
        return self._chunks.popleft() if self._chunks else None

    def __len__(self) -> int:
        return len(self._chunks)
