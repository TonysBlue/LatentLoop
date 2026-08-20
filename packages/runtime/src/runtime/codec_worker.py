from __future__ import annotations

import json
import socket
import struct
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from runtime.codec import CodecIdentity

PROTOCOL_VERSION = 1
_HEADER = struct.Struct("!I")


def _read_exact(connection: socket.socket, size: int) -> bytes:
    blocks = bytearray()
    while len(blocks) < size:
        block = connection.recv(size - len(blocks))
        if not block:
            raise ConnectionError("codec worker closed the connection")
        blocks.extend(block)
    return bytes(blocks)


def send_message(connection: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    header = {**header, "protocol_version": PROTOCOL_VERSION, "payload_bytes": len(payload)}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    connection.sendall(_HEADER.pack(len(encoded)) + encoded + payload)


def receive_message(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_size = _HEADER.unpack(_read_exact(connection, _HEADER.size))[0]
    if header_size > 64 * 1024:
        raise ValueError("codec worker header is too large")
    header = json.loads(_read_exact(connection, header_size))
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("codec worker protocol version mismatch")
    payload_size = int(header.get("payload_bytes", 0))
    if payload_size > 64 * 1024 * 1024:
        raise ValueError("codec worker payload is too large")
    return header, _read_exact(connection, payload_size)


class CodecWorkerClient:
    def __init__(
        self,
        socket_path: str | Path,
        identity: CodecIdentity,
        *,
        timeout_seconds: float = 10.0,
        replay_frames: int = 250,
    ) -> None:
        self.socket_path = str(Path(socket_path).expanduser())
        self.identity = identity
        self.timeout_seconds = timeout_seconds
        self.history: deque[np.ndarray] = deque(maxlen=replay_frames)
        self.session_id: str | None = None

    def health(self) -> dict[str, Any]:
        response, _ = self._request("health", None)
        worker_identity = response.get("identity")
        if worker_identity != asdict(self.identity):
            raise RuntimeError("codec worker identity does not match the configured codec")
        return response

    def reset(self, session_id: str, *, replay: bool = True) -> None:
        if not session_id:
            raise ValueError("codec session_id is required")
        same_session = session_id == self.session_id
        self._request("reset", session_id)
        self.session_id = session_id
        if replay and same_session:
            for codes in self.history:
                self._decode(codes, discard=True)
        else:
            self.history.clear()

    def restore_decode_history(self, session_id: str, codes: Iterable[Tensor]) -> None:
        """Reset a decoder and replay the bounded sampled-code history."""
        values = list(codes)
        limit = self.history.maxlen
        if limit is not None:
            values = values[-limit:]
        self.reset(session_id, replay=False)
        for item in values:
            self.decode_step(item, session_id)

    def encode_step(self, waveform: Tensor, session_id: str) -> Tensor:
        expected = (1, 1, self.identity.frame_samples)
        if tuple(waveform.shape) != expected:
            raise ValueError(f"codec waveform must have shape {expected}")
        if not torch.isfinite(waveform).all():
            raise ValueError("codec waveform contains a non-finite sample")
        if waveform.numel() and waveform.detach().abs().max().item() > 1.0:
            raise ValueError("codec waveform must be in [-1, 1]")
        values = np.ascontiguousarray(waveform.detach().float().cpu().numpy(), dtype=np.float32)
        response, payload = self._request(
            "encode_step", session_id, values, dtype="float32", shape=expected
        )
        codes = torch.from_numpy(self._array(response, payload, np.uint16).copy()).long()
        self._validate_codes(codes)
        return codes

    def decode_step(self, codes: Tensor, session_id: str) -> Tensor:
        expected = (1, self.identity.codebooks, 1)
        if tuple(codes.shape) != expected:
            raise ValueError(f"codec codes must have shape {expected}")
        self._validate_codes(codes)
        values = np.ascontiguousarray(codes.detach().cpu().numpy(), dtype=np.uint16)
        output = self._decode(values, session_id=session_id)
        self.history.append(values.copy())
        return output

    def _decode(
        self,
        values: np.ndarray,
        *,
        session_id: str | None = None,
        discard: bool = False,
    ) -> Tensor:
        session_id = session_id or self.session_id
        if session_id is None:
            raise RuntimeError("codec session has not been initialized")
        response, payload = self._request(
            "decode_step", session_id, values, dtype="uint16", shape=values.shape
        )
        decoded = self._array(response, payload, np.float32).copy()
        if discard:
            return torch.empty(0)
        return torch.from_numpy(decoded)

    def _validate_codes(self, codes: Tensor) -> None:
        if codes.numel() and (
            codes.detach().min().item() < 0
            or codes.detach().max().item() >= self.identity.codebook_size
        ):
            raise ValueError("codec code is outside the configured codebook")

    def _request(
        self,
        operation: str,
        session_id: str | None,
        values: np.ndarray | None = None,
        **fields: Any,
    ) -> tuple[dict[str, Any], bytes]:
        payload = values.tobytes(order="C") if values is not None else b""
        header = {"operation": operation, "session_id": session_id, **fields}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.connect(self.socket_path)
                send_message(connection, header, payload)
                response, response_payload = receive_message(connection)
        except (ConnectionError, OSError, TimeoutError) as error:
            raise RuntimeError(f"codec worker request failed: {error}") from error
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "codec worker failed")))
        return response, response_payload

    @staticmethod
    def _array(header: dict[str, Any], payload: bytes, dtype: np.dtype[Any]) -> np.ndarray:
        if header.get("dtype") != np.dtype(dtype).name:
            raise RuntimeError("codec worker returned an unexpected dtype")
        shape = tuple(int(value) for value in header["shape"])
        values = np.frombuffer(payload, dtype=dtype)
        if values.size != int(np.prod(shape)):
            raise RuntimeError("codec worker returned an invalid tensor size")
        return values.reshape(shape)
