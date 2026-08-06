from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(slots=True)
class ModelConfig:
    model_dim: int = 256
    latent_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 1024
    cross_attention_every: int = 2
    audio_tokens: int = 4
    audio_kernel: int = 400
    audio_stride: int = 160
    latent_slots: int = 8
    # Production streaming profile retains the most recent 60 seconds.
    kv_units: int = 750
    kv_window_ms: int = 60_000
    speech_frames_per_unit: int = 1
    speech_codebooks: int = 8
    speech_codebook_size: int = 2048
    speech_depth_layers: int = 2
    speech_depth_heads: int = 4
    speech_depth_ffn_dim: int = 1024
    action_text_tokens: int = 16
    action_text_vocab_size: int = 256
    action_key_vocab_size: int = 32
    max_action_duration_ms: int = 10_000
    memory_classes: int = 64
    dropout: float = 0.1
    activation_checkpointing: bool = False

    @property
    def tokens_per_unit(self) -> int:
        return self.audio_tokens + 3  # time, vision, and state-query tokens


@dataclass(slots=True)
class DataConfig:
    source: str = "synthetic"
    shards: str | None = None
    manifest: str | None = None
    schema_version: int = 2
    audio_sample_rate: int = 24_000
    codec_frame_rate: float = 12.5
    codec_id: str = "mimi-24khz-8x2048"
    codec_weight_hash: str = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
    codec_revision: str = "a49141e28b3d9c947cf9aa5314431e1b11cbd2f5"
    codec_codebooks: int = 8
    codec_codebook_size: int = 2048
    unit_ms: int = 80
    unit_audio_samples: int = 1_920
    screen_height: int = 224
    screen_width: int = 224
    episode_units: int = 32
    train_episodes: int = 1_000
    seed: int = 17


@dataclass(slots=True)
class TrainingConfig:
    max_updates: int = 10_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 16
    tbptt_units: int = 16
    mixed_precision: str = "fp16"
    max_grad_norm: float = 1.0
    checkpoint_every: int = 500
    log_every: int = 10
    head_learning_rate: float = 1e-4
    backbone_learning_rate: float = 3e-5
    warmup_ratio: float = 0.03
    codec_scheduled_sampling: float = 0.0
    codec_scheduled_sampling_start: float = 0.7
    backbone_train_mode: str = "all"
    speech_control_class_weights: list[float] = field(default_factory=lambda: [1.0] * 5)
    speech_control_loss_weight: float = 0.25
    latent_write_loss_weight: float = 0.0
    min_learning_rate_ratio: float = 0.1


@dataclass(slots=True)
class TrackingConfig:
    enabled: bool = True
    project: str = "latentloop"
    mode: str = "online"
    base_url: str = "http://127.0.0.1:8080"
    media_samples_per_run: int = 4
    media_warning_gb: float = 40.0
    media_stop_gb: float = 50.0


@dataclass(slots=True)
class RuntimeConfig:
    data_root: str = "~/latentloop-data"

    def root_path(self) -> Path:
        return Path(self.data_root).expanduser().resolve()


@dataclass(slots=True)
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.model.model_dim % self.model.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.model.audio_kernel < self.model.audio_stride:
            raise ValueError("audio_kernel must be >= audio_stride")
        if self.model.kv_units < 1 or self.model.latent_slots < 1:
            raise ValueError("kv_units and latent_slots must be positive")
        expected_kv_units = -(-self.model.kv_window_ms // self.data.unit_ms)
        if self.model.kv_units != expected_kv_units:
            raise ValueError("kv_units must exactly cover kv_window_ms at the configured unit_ms")
        if self.model.action_text_tokens < 1 or self.model.action_key_vocab_size < 1:
            raise ValueError("action text and key dimensions must be positive")
        if self.data.source not in {"synthetic", "webdataset"}:
            raise ValueError("data.source must be synthetic or webdataset")
        if self.data.source == "webdataset" and not self.data.shards:
            raise ValueError("data.shards is required for webdataset")
        if not self.data.codec_id or not self.data.codec_weight_hash:
            raise ValueError("codec identity and weight hash are required")
        if self.data.unit_audio_samples * 1_000 != self.data.audio_sample_rate * self.data.unit_ms:
            raise ValueError("unit_audio_samples must exactly match the audio clock")
        if self.data.codec_codebooks != self.model.speech_codebooks:
            raise ValueError("data and model codec codebook counts must match")
        if self.data.codec_codebook_size != self.model.speech_codebook_size:
            raise ValueError("data and model codec vocabularies must match")
        if self.data.unit_ms * self.data.codec_frame_rate != 1_000:
            raise ValueError("direct speech requires exactly one codec frame per stream unit")
        if self.model.speech_frames_per_unit != 1:
            raise ValueError("direct speech requires exactly one speech frame per stream unit")
        if (
            self.model.speech_depth_heads < 1
            or self.model.model_dim % self.model.speech_depth_heads
        ):
            raise ValueError("model_dim must be divisible by speech_depth_heads")
        if self.training.tbptt_units < 1:
            raise ValueError("tbptt_units must be positive")
        if not 0 <= self.training.codec_scheduled_sampling <= 1:
            raise ValueError("codec_scheduled_sampling must be in [0, 1]")
        if not 0 <= self.training.codec_scheduled_sampling_start < 1:
            raise ValueError("codec_scheduled_sampling_start must be in [0, 1)")
        if self.training.backbone_train_mode not in {"frozen", "selective", "all"}:
            raise ValueError("backbone_train_mode must be frozen, selective, or all")
        if len(self.training.speech_control_class_weights) != 5 or any(
            weight <= 0 for weight in self.training.speech_control_class_weights
        ):
            raise ValueError("speech_control_class_weights must contain five positive values")
        if self.training.speech_control_loss_weight <= 0:
            raise ValueError("speech_control_loss_weight must be positive")
        if self.training.latent_write_loss_weight < 0:
            raise ValueError("latent_write_loss_weight cannot be negative")
        if not 0 <= self.training.min_learning_rate_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if not 0 <= self.training.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.training.mixed_precision not in {"no", "fp16"}:
            raise ValueError("RTX 2080 SUPER profiles support mixed_precision=no or fp16")


def load_config(path: str | Path, overrides: list[str] | None = None) -> ProjectConfig:
    schema = OmegaConf.structured(ProjectConfig)
    loaded = OmegaConf.load(Path(path))
    merged = OmegaConf.merge(schema, loaded, OmegaConf.from_dotlist(overrides or []))
    raw = OmegaConf.to_container(merged, resolve=True)
    assert isinstance(raw, dict)
    config = ProjectConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        tracking=TrackingConfig(**raw["tracking"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )
    config.validate()
    return config
