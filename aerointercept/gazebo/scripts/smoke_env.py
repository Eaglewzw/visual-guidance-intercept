"""Short physical Gazebo/PX4 rollout and reset smoke test."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.environment import GazeboInterceptEnv
from aerointercept.gazebo.process import maybe_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--socket", default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", default="circle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_gazebo_config(args.config)
    socket_path = args.socket or cfg.gazebo.bridge.socket
    stack = maybe_launch(args, socket_path)
    environment = None
    try:
        environment = GazeboInterceptEnv(cfg, socket_path)
        frames, training, reset_info = environment.reset()
        if frames.shape != (2, 3, 640, 640) or frames.dtype != np.uint8:
            raise AssertionError(f"invalid observation {frames.shape} {frames.dtype}")
        if training["critic_obs"].shape != (15,):
            raise AssertionError("privileged Critic protocol is not 15 dimensions")
        started = time.perf_counter()
        rewards = []
        outcomes = []
        info = {}
        for step in range(args.steps):
            # This protocol smoke uses bounded commands, not target truth guidance.
            action = np.array([0.12, 0.0, 0.0, 0.0], dtype=np.float32)
            frames, reward, terminated, truncated, training, info = environment.step(action)
            rewards.append(reward)
            if terminated or truncated:
                outcomes.append(info["final"]["outcome"])
                frames, training, _ = environment.reset()
        elapsed = time.perf_counter() - started
        snapshot = environment._last_snapshot
        report = {
            "backend": "gazebo_px4",
            "physics_steps": args.steps,
            "camera_shape": list(frames.shape),
            "camera_dtype": str(frames.dtype),
            "critic_shape": list(training["critic_obs"].shape),
            "reward_sum": float(np.sum(rewards)),
            "outcomes": outcomes,
            "control_fps": args.steps / elapsed,
            "camera_frames": environment.camera_frames,
            "reset": reset_info,
            "last_visible": info.get("visible"),
            "last_distance": info.get("distance"),
            "last_fov_center_error": info.get("fov_center_error"),
            "debug_truth": None if snapshot is None else {
                "interceptor_position_ned": snapshot["interceptor_position"],
                "target_position_ned": snapshot["target_position"],
                "interceptor_yaw": snapshot["interceptor_yaw"],
            },
        }
        print("AEROINTERCEPT_GAZEBO_ENV=" + json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        if environment is not None:
            environment.close()
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    main()
