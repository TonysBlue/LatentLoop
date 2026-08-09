from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from harness.environment.qemu import QemuBackend, QemuConfig
from harness.transport.control import HarnessControlServer


def _load_backend(config_path: Path, adapter_module: str | None = None):
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Harness service config must be a mapping")
    module_name = adapter_module or "harness.deployment.qemu"
    module = __import__(module_name, fromlist=["create_backend"])
    if hasattr(module, "create_backend"):
        backend = module.create_backend(config)
        _validate_backend(backend, module_name)
        return backend
    if not hasattr(module, "create_adapters"):
        raise RuntimeError(
            f"adapter module {module_name!r} must expose create_backend(config) "
            "or create_adapters(config)"
        )
    sensor, actuator, evaluator = module.create_adapters(config)
    qemu = QemuConfig(
        base_image=Path(str(config["base_image"])).expanduser(),
        runtime_root=Path(str(config["runtime_root"])).expanduser(),
    )
    for name, adapter in (("sensor", sensor), ("actuator", actuator), ("evaluator", evaluator)):
        if adapter is None:
            raise RuntimeError(f"deployment adapter {name} is not configured")
    return QemuBackend(qemu, sensor, actuator, evaluator)


def _validate_backend(backend: object, adapter_module: str) -> None:
    required = ("environment_id", "environment_version", "reset", "apply", "evaluate", "close")
    missing = [name for name in required if not hasattr(backend, name)]
    if missing:
        raise RuntimeError(f"deployment module {adapter_module!r} backend is missing: {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-module", help="test-only deployment module override")
    parser.add_argument("--socket", help="override control socket path")
    args = parser.parse_args(argv)
    config = Path(args.config).expanduser()
    if not config.is_file():
        raise FileNotFoundError(config)
    raw = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Harness service config must be a mapping")
    socket_path = args.socket or str(raw.get("control_socket", raw.get("model_service_socket", "")))
    if not socket_path:
        raise ValueError("Harness service config requires control_socket")
    HarnessControlServer(
        lambda: _load_backend(config, args.adapter_module),
        socket_path,
        expected_environment_id=str(raw.get("environment_id", "")) or None,
        expected_environment_version=str(raw.get("environment_version", "")) or None,
        expected_protocol_version=str(raw.get("protocol_version", "realtime-v1")),
    ).serve_forever()
    return 0
