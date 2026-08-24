"""Evaluate a detector-free Actor in the same physical Gazebo/PX4 backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import load_model_weights
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.environment import GazeboInterceptEnv
from aerointercept.gazebo.process import maybe_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--socket", default=None)
    parser.add_argument("--output", default="results/gazebo_e2e_eval.json")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", choices=("circle", "sinusoidal", "random_walk", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    cfg = load_gazebo_config(args.config)
    socket_path = args.socket or cfg.gazebo.bridge.socket
    stack = maybe_launch(args, socket_path)
    environment = None
    try:
        environment = GazeboInterceptEnv(cfg, socket_path)
        device = torch.device(args.device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = EndToEndActorCritic(cfg.end_to_end.model).to(device).eval()
        load_model_weights(model, checkpoint, dict(cfg.end_to_end.model))
        outcomes = []
        rewards = []
        minimum_distances = []
        lengths = []
        started = time.perf_counter()
        for episode in range(args.episodes):
            frames, _, _ = environment.reset()
            episode_reward = 0.0
            for _ in range(int(cfg.gazebo.task.episode_max_steps)):
                # Deployment path calls Actor directly: no Critic or simulator truth argument.
                with torch.no_grad():
                    action = model.actor.act(
                        torch.from_numpy(frames[None]).to(device), deterministic=True
                    )[0][0].cpu().numpy()
                frames, reward, terminated, truncated, _, info = environment.step(action)
                episode_reward += reward
                if terminated or truncated:
                    final = info["final"]
                    outcomes.append(final["outcome"])
                    rewards.append(episode_reward)
                    minimum_distances.append(float(final["minimum_distance"]))
                    lengths.append(int(final["episode_length"]))
                    print(
                        f"episode={episode + 1}/{args.episodes} outcome={final['outcome']} "
                        f"reward={episode_reward:.2f} min={final['minimum_distance']:.2f}",
                        flush=True,
                    )
                    break
            else:
                raise RuntimeError("environment failed to emit its configured timeout")
        elapsed = time.perf_counter() - started
        report = {
            "backend": "gazebo_harmonic_px4_sitl",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "episodes": args.episodes,
            "hit_rate": sum(item == "hit" for item in outcomes) / len(outcomes),
            "fov_lost_rate": sum(item == "fov_lost" for item in outcomes) / len(outcomes),
            "ground_collision_rate": sum(item == "ground" for item in outcomes) / len(outcomes),
            "mean_reward": float(np.mean(rewards)),
            "mean_minimum_distance": float(np.mean(minimum_distances)),
            "mean_episode_length": float(np.mean(lengths)),
            "mean_interception_time_s": float(np.mean(lengths)) / 20.0,
            "camera_fps": environment.camera_frames / elapsed,
            "elapsed_seconds": elapsed,
            "outcomes": outcomes,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("AEROINTERCEPT_GAZEBO_EVAL=" + json.dumps(report), flush=True)
    finally:
        if environment is not None:
            environment.close()
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    main()
