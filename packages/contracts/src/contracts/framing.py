from __future__ import annotations

import struct


def frame(payload: bytes) -> bytes:
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("payload exceeds 64 MiB")
    return struct.pack(">I", len(payload)) + payload


def read_frame(read_exact) -> bytes:
    header = read_exact(4)
    if len(header) != 4:
        raise ConnectionError("incomplete frame header")
    size = struct.unpack(">I", header)[0]
    if size > 64 * 1024 * 1024:
        raise ValueError("payload exceeds 64 MiB")
    return read_exact(size)
