from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import struct
from pathlib import Path
from typing import Any

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

HEADER = struct.Struct("!I")


def read_exact(connection: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        block = connection.recv(size - len(value))
        if not block:
            raise ConnectionError("client disconnected")
        value.extend(block)
    return bytes(value)


def distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for row, first in enumerate(left, start=1):
        current = [row]
        for column, second in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (first != second),
                )
            )
        previous = current
    return previous[-1]


def transcribe(model: Any, value: dict[str, Any]) -> None:
    request = value["request"]
    result = model.generate(
        input=request["audio"], cache={}, language="auto", use_itn=True, batch_size_s=60
    )
    transcript = rich_transcription_postprocess(result[0]["text"])
    if request["language"] == "zh":
        reference = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", request["text"])).lower()
        hypothesis = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", transcript)).lower()
        metric = "cer"
    else:
        reference = re.findall(r"[a-z0-9']+", request["text"].lower())
        hypothesis = re.findall(r"[a-z0-9']+", transcript.lower())
        metric = "wer"
    score = min(1.0, distance(reference, hypothesis) / max(len(reference), 1))
    output = Path(value["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"metric": metric, "score": score, "transcript": transcript},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = AutoModel(model=str(args.model), device=args.device, disable_update=True)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    os.chmod(args.socket, 0o600)
    server.listen(4)
    server.settimeout(0.5)
    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while not stopping:
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                try:
                    header_size = HEADER.unpack(read_exact(connection, 4))[0]
                    request = json.loads(read_exact(connection, header_size))
                    transcribe(model, request)
                    response = {"ok": True}
                except Exception as error:
                    response = {"ok": False, "error": str(error)}
                encoded = json.dumps(response).encode()
                connection.sendall(HEADER.pack(len(encoded)) + encoded)
    finally:
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
