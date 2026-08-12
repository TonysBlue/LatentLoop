"""Strict protobuf serialization for the physical signal boundary.

Signal encoders return protobuf payloads only.  ``framing`` is exclusively a
socket concern; keeping the two layers separate prevents accidental nested or
double length headers between Model Service and Harness.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

from contracts.control import (
    ActuationSignal,
    ButtonPhase,
    ControlKind,
    ControlSignal,
    SpeechSignal,
)
from contracts.observation import MicSignal, ObservationSignal, ScreenSignal
from contracts.realtime_pb2 import (
    ActuationSignal as ActuationMessage,
)
from contracts.realtime_pb2 import (
    ControlSignal as ControlMessage,
)
from contracts.realtime_pb2 import (
    EnvironmentReceipt as ReceiptMessage,
)
from contracts.realtime_pb2 import (
    ObservationSignal as ObservationMessage,
)
from contracts.realtime_pb2 import (
    RewardBreakdown as RewardMessage,
)
from contracts.receipt import EnvironmentReceipt, RewardBreakdown


def _set_optional(message: Any, field: str, value: Any) -> None:
    if value is not None:
        setattr(message, field, value)


def observation_to_payload(value: ObservationSignal) -> bytes:
    message = ObservationMessage(
        session_id=value.session_id,
        unit_index=value.unit_index,
        timestamp_ms=value.timestamp_ms,
        delta_ms=value.delta_ms,
    )
    message.mic.sample_rate_hz = value.mic.sample_rate_hz
    message.mic.channels = value.mic.channels
    message.mic.encoding = value.mic.encoding
    message.mic.samples = value.mic.samples
    message.screen.width = value.screen.width
    message.screen.height = value.screen.height
    message.screen.pixel_format = value.screen.pixel_format
    message.screen.encoding = value.screen.encoding
    message.screen.image = value.screen.image
    return message.SerializeToString()


def observation_to_message(value: ObservationSignal) -> bytes:
    """Return the strict protobuf payload (without transport framing)."""
    return observation_to_payload(value)


def message_to_observation(payload: bytes) -> ObservationSignal:
    message = ObservationMessage.FromString(payload)
    if not message.HasField("mic") or not message.HasField("screen"):
        raise ValueError("ObservationSignal requires mic and screen messages")
    return ObservationSignal(
        session_id=message.session_id,
        unit_index=message.unit_index,
        timestamp_ms=message.timestamp_ms,
        delta_ms=message.delta_ms,
        mic=MicSignal(
            message.mic.samples,
            sample_rate_hz=message.mic.sample_rate_hz,
            channels=message.mic.channels,
            encoding=message.mic.encoding,
        ),
        screen=ScreenSignal(
            message.screen.image,
            width=message.screen.width,
            height=message.screen.height,
            pixel_format=message.screen.pixel_format,
            encoding=message.screen.encoding,
        ),
    )


def _control_to_message(value: ControlSignal) -> ControlMessage:
    message = ControlMessage(
        kind=value.kind.value,
        event_id=value.event_id,
        text=value.text or "",
        button_phase=value.button_phase.value if value.button_phase is not None else "",
    )
    optional_fields = ("x", "y", "dx", "dy", "key", "button")
    for field in optional_fields:
        _set_optional(message, field, getattr(value, field))
    return message


def _message_to_control(message: ControlMessage) -> ControlSignal:
    values: dict[str, Any] = {"kind": ControlKind(message.kind), "event_id": message.event_id}
    optional_fields = ("x", "y", "dx", "dy", "key", "button")
    for field in optional_fields:
        if message.HasField(field):
            values[field] = getattr(message, field)
    if message.text:
        values["text"] = message.text
    if message.button_phase:
        values["button_phase"] = ButtonPhase(message.button_phase)
    return ControlSignal(**values)


def actuation_to_payload(value: ActuationSignal) -> bytes:
    message = ActuationMessage(session_id=value.session_id, unit_index=value.unit_index)
    message.speech.sample_rate_hz = value.speech.sample_rate_hz
    message.speech.channels = value.speech.channels
    message.speech.encoding = value.speech.encoding
    message.speech.silent = value.speech.silent
    message.speech.pcm = value.speech.pcm
    message.controls.extend(_control_to_message(control) for control in value.controls)
    return message.SerializeToString()


def actuation_to_message(value: ActuationSignal) -> bytes:
    return actuation_to_payload(value)


def message_to_actuation(payload: bytes) -> ActuationSignal:
    message = ActuationMessage.FromString(payload)
    if not message.HasField("speech"):
        raise ValueError("ActuationSignal requires a speech message")
    return ActuationSignal(
        session_id=message.session_id,
        unit_index=message.unit_index,
        speech=SpeechSignal(
            message.speech.pcm,
            sample_rate_hz=message.speech.sample_rate_hz,
            channels=message.speech.channels,
            encoding=message.speech.encoding,
            silent=message.speech.silent,
        ),
        controls=tuple(_message_to_control(control) for control in message.controls),
    )


def receipt_to_payload(value: EnvironmentReceipt) -> bytes:
    return ReceiptMessage(
        session_id=value.session_id,
        unit_index=value.unit_index,
        accepted=value.accepted,
        execution_latency_ms=value.execution_latency_ms,
        safety_violation=value.safety_violation or "",
        terminated=value.terminated,
        infrastructure_failure=value.infrastructure_failure or "",
    ).SerializeToString()


def payload_to_receipt(payload: bytes) -> EnvironmentReceipt:
    message = ReceiptMessage.FromString(payload)
    return EnvironmentReceipt(
        session_id=message.session_id,
        unit_index=message.unit_index,
        accepted=message.accepted,
        execution_latency_ms=message.execution_latency_ms,
        safety_violation=message.safety_violation or None,
        terminated=message.terminated,
        infrastructure_failure=message.infrastructure_failure or None,
    )


def reward_to_payload(value: RewardBreakdown) -> bytes:
    return RewardMessage(
        task=value.task,
        speech_quality=value.speech_quality,
        latency_quality=value.latency_quality,
        action_efficiency=value.action_efficiency,
        safety=value.safety,
        spec_id=value.spec_id,
    ).SerializeToString()


def payload_to_reward(payload: bytes) -> RewardBreakdown:
    message = RewardMessage.FromString(payload)
    return RewardBreakdown(
        task=message.task,
        speech_quality=message.speech_quality,
        latency_quality=message.latency_quality,
        action_efficiency=message.action_efficiency,
        safety=message.safety,
        spec_id=message.spec_id or "realtime-v2",
    )


def serve_socket(
    connection: socket.socket,
    handler: Callable[[bytes], bytes],
) -> None:
    """Serve one length-framed payload, invoking a strict bytes handler."""

    from contracts.framing import frame, read_frame

    def read_exact(size: int) -> bytes:
        output = bytearray()
        while len(output) < size:
            block = connection.recv(size - len(output))
            if not block:
                raise ConnectionError("client disconnected")
            output.extend(block)
        return bytes(output)

    connection.sendall(frame(handler(read_frame(read_exact))))
