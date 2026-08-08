from __future__ import annotations

import numpy as np


def validate_pcm_unit(samples: bytes, *, sample_rate_hz: int = 24_000, channels: int = 1) -> None:
    if sample_rate_hz != 24_000 or channels != 1:
        raise ValueError("realtime PCM must be 24 kHz mono")
    if len(samples) != 1_920 * 4:
        raise ValueError("one PCM unit must contain 1920 float32 samples")
    values = np.frombuffer(samples, dtype=np.float32)
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
        raise ValueError("PCM must contain finite samples in [-1, 1]")


def silence_pcm(samples: int = 1_920) -> bytes:
    if samples != 1_920:
        raise ValueError("realtime speech output is exactly one 80 ms unit")
    return np.zeros(samples, dtype=np.float32).tobytes()
