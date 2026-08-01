from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

from latentloop.codec_worker import CodecWorkerClient
from latentloop.types import Episode


def encode_target_speech(
    episodes: Iterable[Episode], client: CodecWorkerClient
) -> Iterator[Episode]:
    frame_samples = client.identity.frame_samples
    for episode in episodes:
        if episode.target_speech is None:
            raise ValueError(f"episode {episode.episode_id} has no target_speech waveform")
        waveform = episode.target_speech.flatten()
        expected_samples = len(episode.units) * frame_samples
        if waveform.numel() > expected_samples:
            raise ValueError(
                f"episode {episode.episode_id} target speech is longer than its timeline"
            )
        if waveform.numel() < expected_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, expected_samples - waveform.numel())
            )
        client.reset(episode.episode_id, replay=False)
        codes = []
        for offset in range(0, expected_samples, frame_samples):
            frame = waveform[offset : offset + frame_samples].reshape(1, 1, frame_samples)
            encoded = client.encode_step(frame, episode.episode_id)
            codes.append(encoded.transpose(1, 2))
        all_codes = torch.cat(codes, dim=1)
        for index, unit in enumerate(episode.units):
            unit.speech_codes.copy_(all_codes[:, index : index + 1])
        episode.metadata["speech_codes_encoded"] = True
        yield episode
