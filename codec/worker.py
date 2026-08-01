from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from moshi.models import loaders

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Identity:
    codec_id: str
    weight_sha256: str
    revision: str
    sample_rate: int = 24_000
    frame_rate: float = 12.5
    frame_samples: int = 1_920
    codebooks: int = 8
    codebook_size: int = 2_048


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_exact(connection: socket.socket, size: int) -> bytes:
    values = bytearray()
    while len(values) < size:
        block = connection.recv(size - len(values))
        if not block:
            raise ConnectionError("client disconnected")
        values.extend(block)
    return bytes(values)


def receive(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_size = int.from_bytes(read_exact(connection, 4), "big")
    if header_size > 64 * 1024:
        raise ValueError("header is too large")
    header = json.loads(read_exact(connection, header_size))
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol version mismatch")
    payload_size = int(header.get("payload_bytes", 0))
    if payload_size < 0 or payload_size > 64 * 1024 * 1024:
        raise ValueError("payload is too large")
    return header, read_exact(connection, payload_size)


def send(connection: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    header = {**header, "protocol_version": PROTOCOL_VERSION, "payload_bytes": len(payload)}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    connection.sendall(len(encoded).to_bytes(4, "big") + encoded + payload)


class MimiService:
    def __init__(self, weights: Path, identity: Identity, device: str) -> None:
        if file_sha256(weights) != identity.weight_sha256:
            raise ValueError("Mimi weight SHA-256 does not match the configured identity")
        self.identity = identity
        self.device = torch.device(device)
        self.model = loaders.get_mimi(weights, device=self.device)
        self.model.set_num_codebooks(identity.codebooks)
        self._streaming = self.model.streaming(batch_size=1)
        self._streaming.__enter__()
        self.session_id: str | None = None

    def reset(self, session_id: str) -> None:
        self.model.reset_streaming()
        self.session_id = session_id

    def execute(
        self, header: dict[str, Any], payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        operation = header["operation"]
        if operation == "health":
            memory: dict[str, int] = {}
            if self.device.type == "cuda":
                memory = {
                    "memory_allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(self.device),
                }
            return {
                "ok": True,
                "identity": asdict(self.identity),
                "device": str(self.device),
                "torch": torch.__version__,
                **memory,
            }, b""
        session_id = str(header.get("session_id") or "")
        if operation == "reset":
            if not session_id:
                raise ValueError("session_id is required")
            self.reset(session_id)
            return {"ok": True}, b""
        if session_id != self.session_id:
            raise ValueError("session is not active; reset and replay it first")
        if operation == "encode_step":
            waveform = self._array(header, payload, np.float32, (1, 1, 1_920))
            if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) > 1.0:
                raise ValueError("waveform must contain finite samples in [-1, 1]")
            tensor = torch.from_numpy(waveform.copy()).to(self.device)
            with torch.inference_mode():
                output = self.model.encode(tensor)
            return self._response(output, np.uint16)
        if operation == "decode_step":
            codes = self._array(header, payload, np.uint16, (1, 8, 1))
            if np.max(codes) >= self.identity.codebook_size:
                raise ValueError("codec code is outside the configured codebook")
            tensor = torch.from_numpy(codes.astype(np.int64)).to(self.device)
            with torch.inference_mode():
                output = self.model.decode(tensor)
            return self._response(output, np.float32)
        raise ValueError(f"unsupported codec operation: {operation}")

    @staticmethod
    def _array(
        header: dict[str, Any], payload: bytes, dtype: Any, expected: tuple[int, ...]
    ) -> np.ndarray:
        if header.get("dtype") != np.dtype(dtype).name:
            raise ValueError("input dtype mismatch")
        shape = tuple(int(value) for value in header.get("shape", []))
        if shape != expected:
            raise ValueError(f"input shape must be {expected}")
        values = np.frombuffer(payload, dtype=dtype)
        if values.size != int(np.prod(shape)):
            raise ValueError("input payload size mismatch")
        return values.reshape(shape)

    @staticmethod
    def _response(tensor: torch.Tensor, dtype: Any) -> tuple[dict[str, Any], bytes]:
        values = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=dtype)
        return {"ok": True, "dtype": values.dtype.name, "shape": values.shape}, values.tobytes()


def serve(service: MimiService, socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(8)
    server.settimeout(0.5)
    stopping = False

    def request_stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stopping:
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                try:
                    header, payload = receive(connection)
                    response, response_payload = service.execute(header, payload)
                except Exception as error:
                    response, response_payload = {"ok": False, "error": str(error)}, b""
                try:
                    send(connection, response, response_payload)
                except (BrokenPipeError, ConnectionResetError):
                    continue
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--revision", default="a49141e28b3d9c947cf9aa5314431e1b11cbd2f5"
    )
    parser.add_argument(
        "--weight-sha256",
        default="09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
    )
    args = parser.parse_args()
    identity = Identity("mimi-24khz-8x2048", args.weight_sha256, args.revision)
    serve(MimiService(args.weights, identity, args.device), args.socket)


if __name__ == "__main__":
    main()
