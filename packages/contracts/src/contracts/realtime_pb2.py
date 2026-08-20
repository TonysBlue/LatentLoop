"""Runtime-generated protobuf classes for the realtime physical-signal API.

The repository intentionally keeps the generated descriptor in source so a
system installation does not need ``protoc`` at runtime.  The resulting
classes are ordinary protobuf messages and are wire-compatible with
``proto/realtime.proto``.
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool
from google.protobuf.message_factory import GetMessageClass


def _field(
    msg: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    typ: int,
    *,
    label: int = 1,
    proto3_optional: bool = False,
    oneof_index: int | None = None,
) -> None:
    field = msg.field.add(name=name, number=number, type=typ, label=label)
    if proto3_optional:
        field.proto3_optional = True
        if oneof_index is None:
            oneof_index = len(msg.oneof_decl)
            msg.oneof_decl.add(name=f"_{name}")
        field.oneof_index = oneof_index


def _build() -> tuple[object, tuple[object, ...]]:
    file = descriptor_pb2.FileDescriptorProto(
        name="realtime.proto", package="latentloop.realtime.v2", syntax="proto3"
    )
    t = descriptor_pb2.FieldDescriptorProto
    mic = file.message_type.add(name="MicPcm")
    _field(mic, "sample_rate_hz", 1, t.TYPE_UINT32)
    _field(mic, "channels", 2, t.TYPE_UINT32)
    _field(mic, "encoding", 3, t.TYPE_STRING)
    _field(mic, "samples", 4, t.TYPE_BYTES)
    screen = file.message_type.add(name="ScreenFrame")
    _field(screen, "width", 1, t.TYPE_UINT32)
    _field(screen, "height", 2, t.TYPE_UINT32)
    screen.reserved_range.add(start=3, end=5)
    _field(screen, "pixel_format", 5, t.TYPE_STRING)
    _field(screen, "encoding", 6, t.TYPE_STRING)
    _field(screen, "image", 7, t.TYPE_BYTES)
    obs = file.message_type.add(name="ObservationSignal")
    _field(obs, "session_id", 1, t.TYPE_STRING)
    _field(obs, "unit_index", 2, t.TYPE_UINT64)
    _field(obs, "timestamp_ms", 3, t.TYPE_UINT64)
    _field(obs, "delta_ms", 4, t.TYPE_UINT32)
    _field(obs, "mic", 5, t.TYPE_MESSAGE)
    obs.field[-1].type_name = ".latentloop.realtime.v2.MicPcm"
    _field(obs, "screen", 6, t.TYPE_MESSAGE)
    obs.field[-1].type_name = ".latentloop.realtime.v2.ScreenFrame"
    control = file.message_type.add(name="ControlSignal")
    _field(control, "kind", 1, t.TYPE_STRING)
    _field(control, "event_id", 2, t.TYPE_STRING)
    optional = (
        ("x", 4, t.TYPE_FLOAT),
        ("y", 5, t.TYPE_FLOAT),
        ("dx", 8, t.TYPE_FLOAT),
        ("dy", 9, t.TYPE_FLOAT),
        ("key", 12, t.TYPE_UINT32),
        ("button", 13, t.TYPE_UINT32),
    )
    control.reserved_range.add(start=3, end=4)
    for name, number, typ in optional:
        _field(control, name, number, typ, proto3_optional=True)
    _field(control, "text", 11, t.TYPE_STRING)
    _field(control, "button_phase", 14, t.TYPE_STRING)
    speech = file.message_type.add(name="SpeechSignal")
    _field(speech, "sample_rate_hz", 1, t.TYPE_UINT32)
    _field(speech, "channels", 2, t.TYPE_UINT32)
    _field(speech, "encoding", 3, t.TYPE_STRING)
    _field(speech, "silent", 4, t.TYPE_BOOL)
    _field(speech, "pcm", 5, t.TYPE_BYTES)
    act = file.message_type.add(name="ActuationSignal")
    _field(act, "session_id", 1, t.TYPE_STRING)
    _field(act, "unit_index", 2, t.TYPE_UINT64)
    _field(act, "speech", 3, t.TYPE_MESSAGE)
    act.field[-1].type_name = ".latentloop.realtime.v2.SpeechSignal"
    _field(act, "controls", 4, t.TYPE_MESSAGE, label=t.LABEL_REPEATED)
    act.field[-1].type_name = ".latentloop.realtime.v2.ControlSignal"
    receipt = file.message_type.add(name="EnvironmentReceipt")
    _field(receipt, "session_id", 1, t.TYPE_STRING)
    _field(receipt, "unit_index", 2, t.TYPE_UINT64)
    _field(receipt, "accepted", 3, t.TYPE_BOOL)
    _field(receipt, "execution_latency_ms", 4, t.TYPE_DOUBLE)
    _field(receipt, "safety_violation", 5, t.TYPE_STRING)
    _field(receipt, "terminated", 6, t.TYPE_BOOL)
    _field(receipt, "infrastructure_failure", 7, t.TYPE_STRING)
    pool = descriptor_pool.Default()
    try:
        descriptor = pool.Add(file)
    except Exception:
        descriptor = pool.FindFileByName(file.name)
    names = (
        "MicPcm",
        "ScreenFrame",
        "ObservationSignal",
        "ControlSignal",
        "SpeechSignal",
        "ActuationSignal",
        "EnvironmentReceipt",
    )
    return descriptor, tuple(
        GetMessageClass(descriptor.message_types_by_name[name]) for name in names
    )


DESCRIPTOR, _MESSAGE_CLASSES = _build()
(
    MicPcm,
    ScreenFrame,
    ObservationSignal,
    ControlSignal,
    SpeechSignal,
    ActuationSignal,
    EnvironmentReceipt,
) = _MESSAGE_CLASSES

__all__ = [
    "DESCRIPTOR",
    "MicPcm", "ScreenFrame", "ObservationSignal", "ControlSignal", "SpeechSignal",
    "ActuationSignal", "EnvironmentReceipt",
]
