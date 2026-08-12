"""Full-frame, detector-free interception stack.

The actor consumes only a short history of RGB frames.  Simulator truth and
PNG detections are exposed exclusively as training labels or critic inputs.
"""

from .actions import (
    ACTION_DIM,
    BodyVelocityCommand,
    decode_action,
    encode_velocity_command,
)
from .environment import EndToEndInterceptEnv, VecEndToEndInterceptEnv

__all__ = [
    "ACTION_DIM",
    "BodyVelocityCommand",
    "decode_action",
    "encode_velocity_command",
    "EndToEndInterceptEnv",
    "VecEndToEndInterceptEnv",
]
