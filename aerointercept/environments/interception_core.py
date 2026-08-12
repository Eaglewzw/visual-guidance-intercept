"""Shared interception simulation state and dynamics.

The two workflows differ in what the actor observes and how an action is
encoded.  They should not, however, carry separate copies of target spawning,
dynamics integration, and continuous collision detection.  ``InterceptionCore``
is the small, observation-agnostic kernel shared by the learned-guidance and
full-frame end-to-end tasks.
"""
from dataclasses import dataclass
import math

import numpy as np

from ..geometry import segment_min_distance
from .interceptor_dynamics import InterceptorDynamics
from .target_motion import TargetMotion


@dataclass(frozen=True)
class InterceptionTransition:
    """Physical result of applying one velocity command."""

    previous_distance: float
    distance: float
    step_min_distance: float
    hit: bool
    touched_ground: bool
    timed_out: bool


class InterceptionCore:
    """Observation-free interception dynamics and episode bookkeeping."""

    def __init__(self, cfg, mode: str = "mixed", seed=None):
        self.cfg = cfg
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.dynamics = InterceptorDynamics(cfg.dynamics)

        self.target = None
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.target_vel = np.zeros(3, dtype=np.float64)
        self.episode_mode = ""
        self.steps = 0
        self.min_dist = float("inf")
        self.previous_distance = float("inf")

    def reset(self) -> None:
        """Sample a new interceptor/target initial condition."""
        cfg = self.cfg
        rng = self.rng

        yaw0 = rng.uniform(-math.pi, math.pi)
        self.dynamics.reset(
            np.array([0.0, 0.0, cfg.env.standby_alt], dtype=np.float64), yaw0)

        mode = self.mode
        if mode == "mixed":
            mode = str(rng.choice(cfg.target.modes, p=cfg.target.mode_probs))
        self.episode_mode = mode

        spawn_range = rng.uniform(*cfg.env.spawn_range)
        azimuth = yaw0 + rng.uniform(
            -cfg.env.spawn_az_jitter, cfg.env.spawn_az_jitter)
        altitude = cfg.target.alt + rng.uniform(
            -cfg.target.alt_jitter, cfg.target.alt_jitter)
        center = self.dynamics.pos + np.array([
            spawn_range * math.cos(azimuth),
            spawn_range * math.sin(azimuth),
            0.0,
        ])
        center[2] = altitude

        self.target = TargetMotion(mode, cfg.target, rng)
        self.target_pos = self.target.reset(center)
        self.target_vel = np.zeros(3, dtype=np.float64)

        self.steps = 0
        self.min_dist = float(np.linalg.norm(
            self.target_pos - self.dynamics.pos))
        self.previous_distance = self.min_dist

    def step(self, velocity_ned, yaw_rate: float) -> InterceptionTransition:
        """Advance the shared physics by one configured control interval."""
        dt = self.cfg.dynamics.dt
        interceptor_previous, interceptor_now = self.dynamics.step(
            np.asarray(velocity_ned, dtype=np.float64), float(yaw_rate), dt)
        target_previous = self.target_pos.copy()
        self.target_pos, self.target_vel = self.target.step(dt, interceptor_now)

        step_min = segment_min_distance(
            interceptor_previous, interceptor_now,
            target_previous, self.target_pos,
        )
        distance = float(np.linalg.norm(self.target_pos - interceptor_now))
        previous_distance = self.previous_distance

        self.steps += 1
        self.min_dist = min(self.min_dist, step_min)
        self.previous_distance = distance

        return InterceptionTransition(
            previous_distance=previous_distance,
            distance=distance,
            step_min_distance=step_min,
            hit=step_min < self.cfg.env.hit_radius,
            touched_ground=interceptor_now[2] > self.cfg.env.ground_z,
            timed_out=self.steps >= self.cfg.env.episode_max_steps,
        )
