"""Self-describing Gazebo PPO checkpoint helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch


def _command_version(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5.0
        ).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def architecture_config(model_config: dict) -> dict:
    return {key: value for key, value in model_config.items() if key != "encoder_chunk_size"}


def load_model_weights(model, checkpoint: dict, model_config: dict) -> None:
    stored = checkpoint.get("model_config")
    if stored is not None and architecture_config(stored) != architecture_config(model_config):
        raise ValueError("checkpoint visual architecture differs from current configuration")
    result = model.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict model restore failed: {result}")


def payload(model, optimizer, cfg, global_step, seed, best_hit_rate, metrics):
    return {
        "phase": 3,
        "checkpoint_schema": 5,
        "backend": "gazebo_harmonic_px4_sitl",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "config": {
            "model": dict(cfg.end_to_end.model),
            "ppo": dict(cfg.end_to_end.ppo),
            "auxiliary": dict(cfg.end_to_end.auxiliary),
            "gazebo": dict(cfg.gazebo),
        },
        "model_config": dict(cfg.end_to_end.model),
        "render_config": dict(cfg.end_to_end.render),
        "action_config": dict(cfg.gazebo.action),
        "label_config": dict(cfg.end_to_end.labels),
        "safety_config": dict(cfg.end_to_end.safety),
        "random_seed": int(seed),
        "image_size": [640, 640],
        "camera_transform": "full_frame_letterbox_v1",
        "gazebo_version": _command_version(["gz", "sim", "--versions"]),
        "ros_distribution": "humble",
        "px4_interface": "PX4 SITL / px4_msgs",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "best_hit_rate": float(best_hit_rate),
        "hit_rate": float(metrics.get("hit_rate", 0.0)),
        "metrics": dict(metrics),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
        },
    }


def save(path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload(**kwargs), temporary)
    temporary.replace(path)
