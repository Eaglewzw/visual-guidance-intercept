"""Gazebo/PX4 environment exposing only full-frame RGB to the Actor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .client import GazeboBridgeClient
from .protocol import BridgeProtocolError, image_from_snapshot
from .task_logic import (
    build_training_info,
    compute_reward,
    segment_minimum_distance,
    camera_target_yaw_geometry,
    target_visibility,
    termination_flags,
)


@dataclass
class EpisodeStats:
    reward: float = 0.0
    length: int = 0
    minimum_distance: float = float("inf")


class GazeboInterceptEnv:
    """One physical PX4 pair in one running Gazebo world.

    Simulator truth is returned separately in ``training_info`` and never
    enters the Actor observation.  ``reset`` is a physical PX4 position
    setpoint return, not an object teleport or renderer reset.
    """

    observation_shape = (2, 3, 640, 640)
    action_shape = (4,)

    def __init__(
        self,
        cfg,
        socket_path: str | Path,
        *,
        client_factory: Callable[..., GazeboBridgeClient] = GazeboBridgeClient,
        connect: bool = True,
    ):
        self.cfg = cfg
        self.task_cfg = cfg.gazebo.task
        self.reward_cfg = cfg.gazebo.rewards
        self.label_cfg = cfg.end_to_end.labels
        self.client = client_factory(socket_path, timeout=float(cfg.gazebo.bridge.timeout_s))
        if connect:
            status = self.client.connect(float(cfg.gazebo.bridge.startup_timeout_s))
            if status.get("backend") != "gazebo_px4":
                raise BridgeProtocolError("connected process is not the Gazebo/PX4 bridge")
        self._history = np.empty(self.observation_shape, dtype=np.uint8)
        self._last_snapshot: dict | None = None
        self._last_action = np.zeros(4, dtype=np.float32)
        self._previous_distance = float("inf")
        self._lost_count = 0
        self._episode = EpisodeStats()
        self.camera_frames = 0

    @staticmethod
    def _physical_state(snapshot: dict) -> dict:
        keys = (
            "interceptor_position", "interceptor_velocity",
            "target_position", "target_velocity", "interceptor_yaw",
        )
        missing = [key for key in keys if key not in snapshot]
        if missing:
            raise BridgeProtocolError(f"snapshot missing physical state: {missing}")
        return {key: snapshot[key] for key in keys}

    def _observation_and_training(self, snapshot: dict) -> tuple[np.ndarray, dict, dict]:
        image = image_from_snapshot(snapshot)
        self.camera_frames += 1
        state = self._physical_state(snapshot)
        relative = np.asarray(state["target_position"], dtype=np.float64) - np.asarray(
            state["interceptor_position"], dtype=np.float64
        )
        yaw = float(state["interceptor_yaw"])
        camera_yaw = yaw + float(self.cfg.gazebo.camera.mount_yaw_offset_rad)
        cy, sy = np.cos(camera_yaw), np.sin(camera_yaw)
        relative_body = np.array([
            cy * relative[0] + sy * relative[1],
            -sy * relative[0] + cy * relative[1],
            relative[2],
        ])
        visible, center_error = target_visibility(
            relative_body,
            float(self.cfg.gazebo.camera.horizontal_fov),
            float(self.cfg.gazebo.camera.vertical_fov),
        )
        training = build_training_info(state, self.label_cfg, visible)
        metrics = {
            "visible": visible,
            "fov_center_error": center_error,
            "distance": float(np.linalg.norm(relative)),
        }
        return image, training, metrics

    def reset(self) -> tuple[np.ndarray, dict, dict]:
        home = np.asarray(self.task_cfg.reset_position_ned, dtype=np.float64)
        look_at_target = bool(getattr(self.task_cfg, "look_at_target", True))
        mount_yaw_offset = float(self.cfg.gazebo.camera.mount_yaw_offset_rad)
        self.client.reset(
            home, float(self.task_cfg.reset_yaw), look_at_target=look_at_target
        )
        deadline = time.monotonic() + float(self.task_cfg.reset_timeout_s)
        sequence = -1
        snapshot = None
        last_position = None
        last_velocity = None
        last_target_distance = None
        while time.monotonic() < deadline:
            snapshot = self.client.snapshot(sequence, timeout=2.0)
            sequence = int(snapshot["sequence"])
            position = np.asarray(snapshot["interceptor_position"], dtype=np.float64)
            velocity = np.asarray(snapshot["interceptor_velocity"], dtype=np.float64)
            last_position = position
            last_velocity = velocity
            target_position = np.asarray(snapshot["target_position"], dtype=np.float64)
            target_distance = float(np.linalg.norm(target_position - position))
            last_target_distance = target_distance
            target_status = snapshot.get("target_vehicle_status")
            _, desired_yaw, yaw_error = camera_target_yaw_geometry(
                position, target_position, float(snapshot["interceptor_yaw"]),
                mount_yaw_offset,
            )
            target_ready = (
                -float(target_position[2]) >= float(self.task_cfg.target_minimum_altitude)
                and (
                    target_status is None
                    or (target_status.get("armed") and target_status.get("offboard"))
                )
            )
            if (
                np.linalg.norm(position - home) <= float(self.task_cfg.reset_tolerance_m)
                and np.linalg.norm(velocity) <= float(self.task_cfg.reset_speed_tolerance)
                and abs(
                    target_distance - float(self.task_cfg.reset_target_distance_m)
                ) <= float(self.task_cfg.reset_target_distance_tolerance_m)
                and target_ready
                and (
                    not look_at_target
                    or abs(yaw_error) <= float(self.task_cfg.reset_yaw_tolerance_rad)
                )
            ):
                break
        else:
            raise TimeoutError(
                "PX4 did not physically return to the reset setpoint; "
                f"last_position={last_position} last_velocity={last_velocity} "
                f"status={None if snapshot is None else snapshot.get('vehicle_status')} "
                f"target={None if snapshot is None else snapshot.get('target_position')} "
                f"target_distance={last_target_distance} "
                f"camera_target_yaw_error={None if snapshot is None else snapshot.get('camera_target_yaw_error')} "
                f"target_status={None if snapshot is None else snapshot.get('target_vehicle_status')}"
            )
        assert snapshot is not None
        first, _, _ = self._observation_and_training(snapshot)
        # A second, newer camera buffer is mandatory: reset never duplicates one image.
        second_snapshot = self.client.snapshot(int(snapshot["sequence"]), timeout=3.0)
        second, training, metrics = self._observation_and_training(second_snapshot)
        self._history[0] = first
        self._history[1] = second
        self._last_snapshot = second_snapshot
        self._last_action.fill(0.0)
        self._previous_distance = metrics["distance"]
        self._lost_count = 0 if metrics["visible"] else 1
        self._episode = EpisodeStats(minimum_distance=metrics["distance"])
        return self._history.copy(), training, {
            "camera": second_snapshot.get("camera_metadata", {}),
            "backend": "gazebo_px4",
            "look_at_target": look_at_target,
            "target_bearing_rad": float(second_snapshot.get("target_bearing", desired_yaw)),
            "interceptor_yaw_rad": float(second_snapshot["interceptor_yaw"]),
            "camera_target_yaw_error_rad": float(
                second_snapshot.get("camera_target_yaw_error", yaw_error)
            ),
            "target_distance_m": float(metrics["distance"]),
            "requested_target_distance_m": float(
                self.task_cfg.reset_target_distance_m
            ),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict, dict]:
        if self._last_snapshot is None:
            raise RuntimeError("reset must be called before step")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_shape or not np.isfinite(action).all():
            raise ValueError("Actor action must be a finite [forward,right,down,yaw_rate] vector")
        action = np.clip(action, -1.0, 1.0)
        decoded = self.client.action(action)
        previous_snapshot = self._last_snapshot
        snapshot = self.client.snapshot(int(previous_snapshot["sequence"]), timeout=3.0)
        image, training, metrics = self._observation_and_training(snapshot)
        previous_state = self._physical_state(previous_snapshot)
        state = self._physical_state(snapshot)
        finite = all(
            np.isfinite(np.asarray(value, dtype=np.float64)).all()
            for value in state.values()
        )
        minimum = segment_minimum_distance(
            previous_state["interceptor_position"],
            state["interceptor_position"],
            previous_state["target_position"],
            state["target_position"],
        ) if finite else float("inf")
        self._lost_count = 0 if metrics["visible"] else self._lost_count + 1
        self._episode.length += 1
        self._episode.minimum_distance = min(self._episode.minimum_distance, minimum)
        flags = termination_flags(
            step_minimum_distance=minimum,
            lost_count=self._lost_count,
            interceptor_position=state["interceptor_position"],
            invalid=not finite,
            episode_step=self._episode.length,
            cfg=self.task_cfg,
        )
        reward, reward_terms = compute_reward(
            previous_distance=self._previous_distance,
            distance=metrics["distance"],
            visible=metrics["visible"],
            center_error=metrics["fov_center_error"],
            action=action,
            previous_action=self._last_action,
            flags=flags,
            cfg=self.reward_cfg,
        )
        self._episode.reward += reward
        self._history[0] = self._history[1]
        self._history[1] = image
        self._last_snapshot = snapshot
        self._last_action = action.copy()
        self._previous_distance = metrics["distance"]
        terminated = bool(flags["terminated"])
        truncated = bool(flags["timed_out"])
        info = {
            **metrics,
            "reward_terms": reward_terms,
            "termination": flags,
            "decoded_action": decoded,
        }
        if terminated or truncated:
            outcome_order = ("hit", "fov_lost", "ground", "invalid", "out_of_bounds")
            outcome = next((name for name in outcome_order if flags[name]), "timeout")
            info["final"] = {
                "outcome": outcome,
                "episode_reward": self._episode.reward,
                "episode_length": self._episode.length,
                "minimum_distance": self._episode.minimum_distance,
            }
        return self._history.copy(), reward, terminated, truncated, training, info

    def close(self) -> None:
        self.client.close()


class GazeboVectorEnv:
    """Vector façade over independently launched Gazebo/PX4 worlds.

    Each socket must represent a distinct Gazebo partition, ROS_DOMAIN_ID,
    Micro XRCE port and PX4 pair.  This avoids pretending one asynchronous
    physical world is an in-process cloned environment.
    """

    def __init__(self, cfg, socket_paths: Sequence[str | Path]):
        if not socket_paths:
            raise ValueError("at least one Gazebo bridge socket is required")
        self.environments = [GazeboInterceptEnv(cfg, path) for path in socket_paths]
        self.num_envs = len(self.environments)

    def reset(self):
        results = [environment.reset() for environment in self.environments]
        frames = np.stack([result[0] for result in results])
        training = self._stack_training([result[1] for result in results])
        return frames, training, [result[2] for result in results]

    @staticmethod
    def _stack_training(items: list[dict]) -> dict[str, np.ndarray]:
        return {key: np.stack([item[key] for item in items]).astype(np.float32)
                for key in items[0]}

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 4):
            raise ValueError(f"expected actions [{self.num_envs},4], got {actions.shape}")
        results = [env.step(action) for env, action in zip(self.environments, actions)]
        frames, rewards, terminated, truncated, training, infos = map(list, zip(*results))
        dones = np.logical_or(terminated, truncated)
        # Preserve final episode data, then physically return completed PX4 vehicles home.
        for index in np.flatnonzero(dones):
            final = infos[index].get("final")
            reset_frame, reset_training, reset_info = self.environments[index].reset()
            frames[index] = reset_frame
            training[index] = reset_training
            infos[index]["reset"] = reset_info
            infos[index]["final"] = final
        return (
            np.stack(frames),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            self._stack_training(training),
            infos,
        )

    @property
    def camera_frames(self) -> int:
        return sum(environment.camera_frames for environment in self.environments)

    def close(self) -> None:
        for environment in self.environments:
            environment.close()
