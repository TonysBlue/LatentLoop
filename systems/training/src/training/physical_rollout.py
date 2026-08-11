"""Physical rollout client used by Online GRPO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from contracts import (
    ActuationSignal,
    EnvironmentReceipt,
    ObservationSignal,
    RewardBreakdown,
    SpeechSignal,
)
from harness.transport.control import HarnessControlClient
from model.types import ActionFrame, SpeechMode, StreamUnit
from PIL import Image
from runtime.action import ActionFrameDecoder
from runtime.codec_worker import CodecWorkerClient
from runtime.config import ProjectConfig


class PhysicalRolloutClient:
    """Bridges Model Core token samples to Harness physical signals.

    The Harness never receives token arrays.  Token log-probabilities remain in
    the training trace while this client decodes speech/action output at the
    training-to-environment boundary.
    """

    def __init__(self, config: ProjectConfig) -> None:
        socket_path = config.training.rl.environment_socket
        codec_socket = config.training.rl.codec_socket
        if not socket_path:
            raise ValueError("Online GRPO requires training.rl.environment_socket")
        if not codec_socket:
            raise ValueError("Online GRPO requires training.rl.codec_socket")
        self.config = config
        self.harness = HarnessControlClient(socket_path)
        from runtime.codec import CodecIdentity

        self.codec = CodecWorkerClient(
            Path(codec_socket).expanduser(),
            CodecIdentity(
                config.data.codec_id,
                config.data.codec_weight_hash,
                config.data.codec_revision,
                sample_rate=config.data.audio_sample_rate,
                frame_rate=config.data.codec_frame_rate,
                frame_samples=config.data.unit_audio_samples,
                codebooks=config.data.codec_codebooks,
                codebook_size=config.data.codec_codebook_size,
            ),
        )
        self.codec.health()
        self.session_id: str | None = None
        self.actions = ActionFrameDecoder()

    def identity(self) -> dict[str, str]:
        return self.harness.identity()

    def reset(self, task_id: str, seed: int, session_id: str) -> ObservationSignal:
        self.session_id = session_id
        self.codec.reset(session_id, replay=False)
        self.actions.reset()
        return self.harness.reset(task_id, seed, session_id)

    def actuation(
        self,
        observation: ObservationSignal,
        mode: torch.Tensor,
        codes: torch.Tensor,
        action_frame: ActionFrame,
    ) -> ActuationSignal:
        selected_mode = int(mode.reshape(-1)[0].item())
        if selected_mode == int(SpeechMode.SILENCE):
            pcm = np.zeros(self.config.data.unit_audio_samples, dtype=np.float32).tobytes()
            speech = SpeechSignal(pcm, silent=True)
        else:
            code_tensor = codes.detach().to("cpu").long().reshape(
                self.config.model.speech_frames_per_unit, self.config.model.speech_codebooks
            )
            decoded = self.codec.decode_step(
                code_tensor.transpose(0, 1).unsqueeze(0), observation.session_id
            )
            values = decoded.detach().cpu().float().reshape(-1).numpy()
            if values.size != self.config.data.unit_audio_samples:
                raise RuntimeError("codec decoder returned an invalid speech unit")
            if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
                raise RuntimeError("codec decoder returned invalid PCM values")
            speech = SpeechSignal(values.tobytes())
        controls = self.actions.push(
            action_frame.as_contract(),
            event_id=f"{observation.session_id}-{observation.unit_index}",
            screen_revision=observation.screen.revision,
        )
        return ActuationSignal(observation.session_id, observation.unit_index, speech, controls)

    def step(
        self,
        observation: ObservationSignal,
        mode: torch.Tensor,
        codes: torch.Tensor,
        action_frame: ActionFrame,
    ) -> tuple[ObservationSignal, EnvironmentReceipt, ActuationSignal]:
        output = self.actuation(observation, mode, codes, action_frame)
        next_observation, receipt = self.harness.apply(output)
        return next_observation, receipt, output

    def evaluate(self, task_id: str) -> RewardBreakdown:
        if self.session_id is None:
            raise RuntimeError("physical rollout session is not active")
        return self.harness.evaluate(task_id, self.session_id)

    def close(self) -> None:
        if self.session_id is not None:
            self.harness.close(self.session_id)
            self.session_id = None


def observation_to_stream_unit(observation: ObservationSignal, config: ProjectConfig) -> StreamUnit:
    dtype = np.float32 if observation.mic.encoding == "pcm_f32le" else np.int16
    mic = np.frombuffer(observation.mic.samples, dtype=dtype)
    if dtype == np.int16:
        mic = mic.astype(np.float32) / 32768.0
    if mic.size != config.data.unit_audio_samples:
        raise ValueError("physical observation microphone has wrong unit size")
    import io

    expected = observation.screen.width * observation.screen.height * 3
    if observation.screen.encoding == "raw" and len(observation.screen.image) == expected:
        pixels = np.frombuffer(observation.screen.image, dtype=np.uint8).reshape(
            observation.screen.height, observation.screen.width, 3
        )
    else:
        with Image.open(io.BytesIO(observation.screen.image)) as image:
            pixels = np.asarray(image.convert("RGB"))
    screen = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    return StreamUnit(
        timestamp_ms=torch.tensor([observation.timestamp_ms], dtype=torch.long),
        delta_ms=torch.tensor([observation.delta_ms], dtype=torch.long),
        mic_audio=torch.from_numpy(mic.copy()).reshape(1, -1),
        screen=screen,
        screen_valid=torch.tensor([observation.screen.valid]),
        screen_revision=torch.tensor([observation.screen.revision], dtype=torch.long),
        speech_mode=torch.zeros(1, dtype=torch.long),
        speech_mode_mask=torch.zeros(1, dtype=torch.bool),
        speech_codes=torch.zeros(
            1, config.model.speech_frames_per_unit, config.model.speech_codebooks, dtype=torch.long
        ),
        speech_codec_mask=torch.zeros(1, config.model.speech_frames_per_unit, dtype=torch.bool),
        action=ActionFrame.no_action(1),
        action_supervision_mask=torch.zeros(1, dtype=torch.bool),
    )
