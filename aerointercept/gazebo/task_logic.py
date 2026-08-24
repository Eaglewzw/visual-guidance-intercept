"""Backend-neutral reward, termination, Critic, and auxiliary-label logic.

All positions and velocities use PX4 NED coordinates.  These functions are
pure NumPy so they can be tested without starting ROS, Gazebo, or PX4.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aerointercept.end_to_end.actions import ned_to_body


CRITIC_DIM = 15


def wrap_yaw(angle: float) -> float:
    """Wrap one NED yaw angle to ``[-pi, pi)``."""
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def camera_target_yaw_geometry(
    interceptor_position: Any,
    target_position: Any,
    interceptor_yaw: float,
    camera_mount_yaw_offset: float = 0.0,
) -> tuple[float, float, float]:
    """Return target bearing, desired body yaw, and camera-axis error in NED.

    The camera offset is the fixed yaw from the body-forward axis to the
    optical axis.  PX4 ``x500_depth`` uses zero, but keeping it explicit makes
    the reset controller safe for a later camera mount change.
    """
    relative = np.asarray(target_position, dtype=np.float64) - np.asarray(
        interceptor_position, dtype=np.float64
    )
    if relative.shape != (3,) or not np.isfinite(relative).all():
        raise ValueError("camera look-at positions must be finite NED three-vectors")
    bearing = math.atan2(float(relative[1]), float(relative[0]))
    desired_body_yaw = wrap_yaw(bearing - float(camera_mount_yaw_offset))
    camera_axis_yaw = float(interceptor_yaw) + float(camera_mount_yaw_offset)
    error = wrap_yaw(bearing - camera_axis_yaw)
    return float(bearing), desired_body_yaw, error


def quaternion_wxyz_to_yaw(quaternion: Any) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def target_visibility(
    relative_body: Any, horizontal_fov: float, vertical_fov: float
) -> tuple[bool, float]:
    forward, right, down = np.asarray(relative_body, dtype=np.float64)
    horizontal = math.atan2(right, forward)
    vertical = math.atan2(down, max(math.hypot(forward, right), 1.0e-12))
    visible = bool(
        forward > 0.0
        and abs(horizontal) <= 0.5 * horizontal_fov
        and abs(vertical) <= 0.5 * vertical_fov
    )
    center_error = (
        (horizontal / (0.5 * horizontal_fov)) ** 2
        + (vertical / (0.5 * vertical_fov)) ** 2
    )
    return visible, float(center_error)


def segment_minimum_distance(
    interceptor_previous: Any,
    interceptor_now: Any,
    target_previous: Any,
    target_now: Any,
) -> float:
    start = np.asarray(interceptor_previous, dtype=np.float64) - np.asarray(
        target_previous, dtype=np.float64
    )
    delta = (
        np.asarray(interceptor_now, dtype=np.float64)
        - np.asarray(interceptor_previous, dtype=np.float64)
        - np.asarray(target_now, dtype=np.float64)
        + np.asarray(target_previous, dtype=np.float64)
    )
    denominator = float(np.dot(delta, delta))
    fraction = 0.0 if denominator < 1.0e-12 else np.clip(
        -float(np.dot(start, delta)) / denominator, 0.0, 1.0
    )
    return float(np.linalg.norm(start + fraction * delta))


def build_training_info(state: dict, cfg, visible: bool) -> dict[str, np.ndarray]:
    interceptor_position = np.asarray(state["interceptor_position"], dtype=np.float64)
    interceptor_velocity = np.asarray(state["interceptor_velocity"], dtype=np.float64)
    target_position = np.asarray(state["target_position"], dtype=np.float64)
    target_velocity = np.asarray(state["target_velocity"], dtype=np.float64)
    yaw = float(state["interceptor_yaw"])
    relative = target_position - interceptor_position
    relative_velocity = target_velocity - interceptor_velocity
    relative_body = ned_to_body(relative, yaw)
    relative_velocity_body = ned_to_body(relative_velocity, yaw)
    target_velocity_body = ned_to_body(target_velocity, yaw)
    interceptor_velocity_body = ned_to_body(interceptor_velocity, yaw)
    distance = float(np.linalg.norm(relative))
    altitude = -float(interceptor_position[2])
    critic = np.concatenate((
        relative_body / float(cfg.position_norm),
        relative_velocity_body / float(cfg.velocity_norm),
        target_velocity_body / float(cfg.target_velocity_norm),
        interceptor_velocity_body / float(cfg.velocity_norm),
        np.array([
            altitude / float(cfg.altitude_norm),
            distance / float(cfg.position_norm),
            float(visible),
        ]),
    )).astype(np.float32)
    if critic.shape != (CRITIC_DIM,):
        raise RuntimeError(f"critic protocol changed: {critic.shape}")

    horizon = float(cfg.future_horizon_s)
    future_relative = relative + relative_velocity * horizon
    future_body = ned_to_body(future_relative, yaw)
    risk_horizon = float(cfg.risk_horizon_s)
    speed_squared = float(np.dot(relative_velocity, relative_velocity))
    closest_time = 0.0 if speed_squared < 1.0e-12 else np.clip(
        -float(np.dot(relative, relative_velocity)) / speed_squared,
        0.0,
        risk_horizon,
    )
    closest_distance = np.linalg.norm(relative + relative_velocity * closest_time)
    return {
        "critic_obs": np.clip(critic, -5.0, 5.0),
        "future_position": np.clip(
            future_body / float(cfg.position_norm), -1.0, 1.0
        ).astype(np.float32),
        "collision_risk": np.float32(closest_distance < float(cfg.risk_radius)),
        "confidence": np.float32(visible),
    }


def termination_flags(
    *,
    step_minimum_distance: float,
    lost_count: int,
    interceptor_position: Any,
    invalid: bool,
    episode_step: int,
    cfg,
) -> dict[str, bool]:
    position = np.asarray(interceptor_position, dtype=np.float64)
    altitude = -float(position[2]) if np.isfinite(position[2]) else -math.inf
    result = {
        "hit": step_minimum_distance <= float(cfg.hit_radius),
        "fov_lost": lost_count >= int(cfg.lost_steps),
        "ground": altitude <= float(cfg.ground_height),
        "invalid": bool(invalid),
        "out_of_bounds": bool(
            np.linalg.norm(position[:2]) > float(cfg.scene_boundary)
            or altitude > float(cfg.maximum_altitude)
        ),
        "timed_out": episode_step >= int(cfg.episode_max_steps),
    }
    result["terminated"] = any(
        result[name] for name in ("hit", "fov_lost", "ground", "invalid", "out_of_bounds")
    )
    return result


def compute_reward(
    *,
    previous_distance: float,
    distance: float,
    visible: bool,
    center_error: float,
    action: Any,
    previous_action: Any,
    flags: dict[str, bool],
    cfg,
) -> tuple[float, dict[str, float]]:
    terms = {
        "close": float(cfg.close) * (previous_distance - distance),
        "hit": float(cfg.hit) * float(flags["hit"]),
        "time": float(cfg.time),
        "fov_center": float(cfg.fov_center) * min(center_error, 9.0),
        "lost": float(cfg.lost) * float(not visible),
        "smooth": float(cfg.smooth) * float(np.sum(
            (np.asarray(action) - np.asarray(previous_action)) ** 2
        )),
        "ground": float(cfg.ground) * float(flags["ground"]),
        "invalid": float(cfg.invalid) * float(flags["invalid"]),
        "out_of_bounds": float(cfg.out_of_bounds) * float(flags["out_of_bounds"]),
        "timeout": float(cfg.timeout) * float(flags["timed_out"]),
    }
    return float(sum(terms.values())), terms
