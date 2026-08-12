"""Detector-independent action protocol for the end-to-end actor.

An image alone does not encode the aircraft's global NED yaw.  Asking a pure
vision model to emit global NED velocity would therefore make the learning
problem unobservable.  The actor emits a body-heading velocity command
(``forward, right, down``) plus yaw rate.  A deterministic yaw rotation maps it
to PX4's NED velocity interface; no detector, bbox, or LOS value is involved.
"""
from dataclasses import dataclass
import math

import numpy as np


ACTION_DIM = 4


@dataclass(frozen=True)
class BodyVelocityCommand:
    """Decoded command in both actor/body and PX4/NED coordinates."""

    forward: float
    right: float
    down: float
    yaw_rate: float
    north: float
    east: float

    @property
    def body_velocity(self) -> np.ndarray:
        return np.array([self.forward, self.right, self.down], dtype=np.float64)

    @property
    def ned_velocity(self) -> np.ndarray:
        return np.array([self.north, self.east, self.down], dtype=np.float64)


def body_to_ned(velocity_body, yaw: float) -> np.ndarray:
    """Rotate heading-frame ``[forward, right, down]`` velocity into NED."""
    forward, right, down = np.asarray(velocity_body, dtype=np.float64)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        cy * forward - sy * right,
        sy * forward + cy * right,
        down,
    ], dtype=np.float64)


def ned_to_body(velocity_ned, yaw: float) -> np.ndarray:
    """Rotate NED velocity into ``[forward, right, down]`` heading axes."""
    north, east, down = np.asarray(velocity_ned, dtype=np.float64)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        cy * north + sy * east,
        -sy * north + cy * east,
        down,
    ], dtype=np.float64)


def decode_action(action, yaw: float, *, velocity_max: float,
                  yaw_rate_max: float) -> BodyVelocityCommand:
    """Decode ``[-1, 1]^4`` into a bounded body velocity and NED command.

    The vector norm is limited after component decoding, so even a saturated
    diagonal network output cannot exceed ``velocity_max``.
    """
    if velocity_max <= 0.0 or yaw_rate_max <= 0.0:
        raise ValueError("velocity_max and yaw_rate_max must be positive")
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"expected action shape ({ACTION_DIM},), got {action.shape}")

    velocity_body = action[:3] * float(velocity_max)
    speed = float(np.linalg.norm(velocity_body))
    if speed > velocity_max:
        velocity_body *= float(velocity_max) / speed
    velocity_ned = body_to_ned(velocity_body, yaw)
    return BodyVelocityCommand(
        forward=float(velocity_body[0]),
        right=float(velocity_body[1]),
        down=float(velocity_body[2]),
        yaw_rate=float(action[3] * yaw_rate_max),
        north=float(velocity_ned[0]),
        east=float(velocity_ned[1]),
    )


def encode_velocity_command(velocity_ned, yaw_rate: float, yaw: float, *,
                            velocity_max: float,
                            yaw_rate_max: float) -> np.ndarray:
    """Encode a conventional NED command as a full-frame training action."""
    if velocity_max <= 0.0 or yaw_rate_max <= 0.0:
        raise ValueError("velocity_max and yaw_rate_max must be positive")
    velocity_body = ned_to_body(velocity_ned, yaw)
    speed = float(np.linalg.norm(velocity_body))
    if speed > velocity_max:
        velocity_body *= float(velocity_max) / speed
    action = np.empty(ACTION_DIM, dtype=np.float32)
    action[:3] = np.clip(velocity_body / float(velocity_max), -1.0, 1.0)
    action[3] = np.clip(yaw_rate / float(yaw_rate_max), -1.0, 1.0)
    return action
