from __future__ import annotations

import io
import threading
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch
from contracts import (
    ActuationSignal,
    ObservationSignal,
    SpeechSignal,
)
from model import StreamingLatentLoop
from model.types import ActionFrame, SpeechMode, SpeechSamplingConfig, StreamUnit
from PIL import Image
from runtime.action import ActionFrameDecoder
from runtime.config import ProjectConfig


@dataclass(slots=True)
class ModelSession:
    session_id: str
    state: object
    next_unit: int = 0
    speech_active: bool = False
    action_decoder: ActionFrameDecoder = field(default_factory=ActionFrameDecoder)


class ModelService:
    """Owns model inference state but never captures or executes devices."""

    def __init__(
        self,
        config: ProjectConfig,
        checkpoint: str | None = None,
        device: str = "cpu",
        speech_decoder: object | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = StreamingLatentLoop(config.model).to(self.device)
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("format_version") != 8:
                raise ValueError("model service requires a format v8 checkpoint")
            if payload.get("metadata", {}).get("action_schema_id") != config.model.action_schema_id:
                raise ValueError("checkpoint action schema does not match model service")
            weights = payload.get("model")
            if not isinstance(weights, dict):
                raise ValueError("model checkpoint has no state dict")
            self.model.load_state_dict(weights, strict=True)
        self.model.eval()
        self._sessions: dict[str, ModelSession] = {}
        self._lock = threading.RLock()
        self._speech_decoder = speech_decoder

    def identity(self) -> dict[str, str]:
        return {
            "service": "model-service",
            "version": "1",
            "protocol_version": "realtime-v2",
            "action_schema_id": self.config.model.action_schema_id,
            "codec_id": self.config.data.codec_id,
        }

    def open_session(self, session_id: str | None = None) -> str:
        selected = session_id or uuid.uuid4().hex
        with self._lock:
            if selected in self._sessions:
                raise ValueError("session already exists")
            self._sessions[selected] = ModelSession(
                selected,
                self.model.initial_state(1, self.device),
            )
            if self._speech_decoder is not None and hasattr(self._speech_decoder, "reset"):
                self._speech_decoder.reset(selected, replay=False)
        return selected

    def close_session(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError("unknown session")
            del self._sessions[session_id]

    @staticmethod
    def _screen_tensor(observation: ObservationSignal) -> torch.Tensor:
        screen = observation.screen
        expected = screen.width * screen.height * 3
        if (
            screen.encoding == "raw"
            and screen.pixel_format == "rgb24"
            and len(screen.image) == expected
        ):
            array = np.frombuffer(screen.image, dtype=np.uint8).reshape(
                screen.height, screen.width, 3
            )
        else:
            with Image.open(io.BytesIO(screen.image)) as image:
                array = np.asarray(image.convert("RGB"))
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)

    def _unit(self, observation: ObservationSignal) -> StreamUnit:
        dtype = np.float32 if observation.mic.encoding == "pcm_f32le" else np.int16
        mic = np.frombuffer(observation.mic.samples, dtype=dtype)
        if dtype == np.int16:
            mic = mic.astype(np.float32) / 32768.0
        if mic.size != self.config.data.unit_audio_samples:
            raise ValueError("mic signal must contain exactly one 80 ms unit")
        screen = self._screen_tensor(observation)
        zeros = torch.zeros(1, 1, self.config.model.speech_codebooks, dtype=torch.long)
        return StreamUnit(
            timestamp_ms=torch.tensor([observation.timestamp_ms], dtype=torch.long),
            delta_ms=torch.tensor([observation.delta_ms], dtype=torch.long),
            mic_audio=torch.from_numpy(mic.copy()).reshape(1, -1),
            screen=screen,
            speech_mode=torch.zeros(1, dtype=torch.long),
            speech_mode_mask=torch.zeros(1, dtype=torch.bool),
            speech_codes=zeros,
            speech_codec_mask=torch.zeros(1, 1, dtype=torch.bool),
            action=ActionFrame.no_action(1),
            action_supervision_mask=torch.zeros(1, dtype=torch.bool),
        )

    @torch.inference_mode()
    def infer(self, observation: ObservationSignal) -> ActuationSignal:
        with self._lock:
            session = self._sessions.get(observation.session_id)
            if session is None:
                raise KeyError("unknown model session")
            if observation.unit_index != session.next_unit:
                raise ValueError("model unit index is out of order")
            unit = self._unit(observation).to(self.device)
            generated = self.model.generate_step(
                unit,
                session.state,
                SpeechSamplingConfig(temperature=0.8, top_k=250, greedy=False),
            )
            session.state = generated.output.state
            session.next_unit += 1
            mode = int(generated.speech_mode.item())
            if mode == int(SpeechMode.SILENCE):
                session.speech_active = False
                speech = SpeechSignal(
                    np.zeros(self.config.data.unit_audio_samples, dtype=np.float32).tobytes(),
                    silent=True,
                )
            else:
                if self._speech_decoder is None:
                    raise RuntimeError("speech codec decoder is not configured")
                if not session.speech_active and hasattr(self._speech_decoder, "reset"):
                    self._speech_decoder.reset(observation.session_id, replay=False)
                codes = generated.output.state.speech_local.previous_codes[:, :, None]
                decoder = self._speech_decoder
                if hasattr(decoder, "decode_step"):
                    waveform = decoder.decode_step(codes, observation.session_id)
                else:
                    waveform = decoder(codes, observation.session_id)
                values = waveform.detach().cpu().float().reshape(-1).numpy()
                if values.size != self.config.data.unit_audio_samples:
                    raise RuntimeError("speech codec decoder returned an invalid PCM unit")
                if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
                    raise RuntimeError("speech codec decoder returned invalid PCM values")
                speech = SpeechSignal(values.tobytes(), silent=False)
                session.speech_active = True
            controls = session.action_decoder.push(
                generated.action_frame.as_contract(),
                event_id=f"{observation.session_id}-{observation.unit_index}",
            )
            return ActuationSignal(
                observation.session_id,
                observation.unit_index,
                speech,
                controls,
            )
