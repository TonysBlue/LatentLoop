"""Model Core public surface.

Only tensor code is exported here.  Serving and training systems depend on
this package boundary, so they cannot accidentally acquire device or reward
dependencies.
"""

from latentloop.model import FactorizedSpeechHead, StreamingLatentLoop
from latentloop.model.action import ActionHead
from latentloop.types import (
    GenerationOutput,
    RecurrentState,
    SpeechMode,
    SpeechSamplingConfig,
    StepOutput,
    StreamUnit,
)

__all__ = [
    "ActionHead", "FactorizedSpeechHead", "GenerationOutput", "RecurrentState", "SpeechMode",
    "SpeechSamplingConfig", "StepOutput", "StreamUnit", "StreamingLatentLoop",
]
