"""Configuration loading for the Gazebo overlay."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from aerointercept.config import DEFAULT_CONFIG, DotDict


DEFAULT_GAZEBO_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "gazebo_e2e.yaml"
)


def _merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_gazebo_config(path: str | Path | None = None) -> DotDict:
    """Deep-merge the Gazebo contract over the unchanged legacy defaults."""
    with Path(DEFAULT_CONFIG).open("r", encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    overlay_path = Path(path) if path is not None else DEFAULT_GAZEBO_CONFIG
    with overlay_path.open("r", encoding="utf-8") as stream:
        overlay = yaml.safe_load(stream)
    result = DotDict(_merge(base, overlay))
    render = result.end_to_end.render
    if (int(render.image_width), int(render.image_height)) != (640, 640):
        raise ValueError("Gazebo actor input must remain 640x640")
    if render.channel_order != "RGB" or int(render.history_frames) != 2:
        raise ValueError("Gazebo observation protocol must be two RGB frames")
    return result
