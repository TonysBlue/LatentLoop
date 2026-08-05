from __future__ import annotations

import argparse
import json
import os
import socket
import struct
from pathlib import Path

HEADER = struct.Struct("!I")


def read_exact(connection: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        block = connection.recv(size - len(value))
        if not block:
            raise ConnectionError("worker disconnected")
        value.extend(block)
    return bytes(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    payload = json.dumps({"request": request, "output": str(args.output)}).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(300)
        connection.connect(os.environ["COSYVOICE_SOCKET"])
        connection.sendall(HEADER.pack(len(payload)) + payload)
        response = json.loads(read_exact(connection, HEADER.unpack(read_exact(connection, 4))[0]))
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "CosyVoice worker failed")))


if __name__ == "__main__":
    main()
