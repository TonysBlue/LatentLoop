"""Model Core public surface.

Only tensor code is exported here.  Serving and training systems depend on
this package boundary, so they cannot accidentally acquire device or reward
dependencies.
"""

from model.action import ActionHead, action_frame_log_prob, action_log_prob_components
from model.core import FactorizedSpeechHead, StreamingLatentLoop
from model.losses import compute_losses
from model.types import (
    ActionFrame,
    ActionHeadOutput,
    ActionLocalState,
    Episode,
    GenerationOutput,
    LayerKV,
    RecurrentState,
    SpeechLocalState,
    SpeechMode,
    SpeechSamplingConfig,
    StepOutput,
    StreamUnit,
)

__all__ = [
    "ActionFrame", "ActionHead", "ActionHeadOutput", "ActionLocalState", "Episode",
    "FactorizedSpeechHead",
    "GenerationOutput", "LayerKV", "RecurrentState", "SpeechLocalState", "SpeechMode",
    "SpeechSamplingConfig", "StepOutput", "StreamUnit", "StreamingLatentLoop",
    "action_frame_log_prob", "action_log_prob_components", "compute_losses",
]
