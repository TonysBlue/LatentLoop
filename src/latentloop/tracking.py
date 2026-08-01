from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch
import wandb

from latentloop.checkpoint import config_hash
from latentloop.config import ProjectConfig


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _server_is_healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/ready", timeout=2) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


class Tracker:
    def __init__(
        self,
        config: ProjectConfig,
        stage: str,
        model_name: str,
        *,
        parameter_count: int,
        data_identity: str,
    ) -> None:
        self.config = config
        self.run: wandb.sdk.wandb_run.Run | None = None
        if not config.tracking.enabled or config.tracking.mode == "disabled":
            return
        os.environ["WANDB_BASE_URL"] = config.tracking.base_url
        wandb_directory = config.runtime.root_path() / "runs" / "wandb"
        wandb_directory.mkdir(parents=True, exist_ok=True)
        effective_mode = config.tracking.mode
        if effective_mode == "online" and not _server_is_healthy(config.tracking.base_url):
            effective_mode = "offline"
        run_config = {
            **config.as_dict(),
            "identity": {
                "config_sha256": config_hash(config.as_dict()),
                "data_identity": data_identity,
                "codec_id": config.data.codec_id,
                "codec_weight_hash": config.data.codec_weight_hash,
                "codec_revision": config.data.codec_revision,
                "git_commit": _git_short_sha(),
                "git_dirty": _git_dirty(),
                "parameter_count": parameter_count,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu",
            },
        }
        try:
            self.run = wandb.init(
                project=config.tracking.project,
                name=(
                    f"{stage}-{model_name}-"
                    f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{_git_short_sha()}"
                ),
                config=run_config,
                mode=effective_mode,
                dir=str(wandb_directory),
                settings=wandb.Settings(init_timeout=10),
            )
        except Exception:
            effective_mode = "offline"
            self.run = wandb.init(
                project=config.tracking.project,
                config=run_config,
                mode="offline",
                dir=str(wandb_directory),
            )
        if self.run is not None:
            self.run.summary["tracking/effective_mode"] = effective_mode

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def record_checkpoint(self, path: Path, digest: str, step: int) -> None:
        if self.run is not None:
            self.run.summary[f"checkpoint/{step}/path"] = str(path)
            self.run.summary[f"checkpoint/{step}/sha256"] = digest

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    @property
    def effective_mode(self) -> str | None:
        if self.run is None:
            return None
        return str(self.run.summary.get("tracking/effective_mode"))

    def media_allowed(self) -> bool:
        if self.run is None:
            return False
        volume = self.config.runtime.root_path() / "runs" / "wandb"
        total_bytes = sum(path.stat().st_size for path in volume.rglob("*") if path.is_file())
        return total_bytes < self.config.tracking.media_stop_gb * 1024**3


def manifest_hash(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()
