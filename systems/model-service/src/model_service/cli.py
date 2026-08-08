from __future__ import annotations

import argparse
from pathlib import Path

from latentloop.codec import CodecIdentity
from latentloop.codec_worker import CodecWorkerClient
from latentloop.config import load_config
from model_service.service import ModelService
from model_service.transport.server import UnixModelServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-service")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--codec-socket", help="Mimi codec worker Unix socket")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    decoder = None
    if args.codec_socket:
        decoder = CodecWorkerClient(
            Path(args.codec_socket),
            CodecIdentity(
                config.data.codec_id,
                config.data.codec_weight_hash,
                config.data.codec_revision,
                sample_rate=config.data.audio_sample_rate,
                frame_rate=config.data.codec_frame_rate,
                frame_samples=config.data.unit_audio_samples,
                codebooks=config.data.codec_codebooks,
                codebook_size=config.data.codec_codebook_size,
            ),
        )
        decoder.health()
    UnixModelServer(
        ModelService(config, args.checkpoint, args.device, speech_decoder=decoder), args.socket
    ).serve_forever()
    return 0
