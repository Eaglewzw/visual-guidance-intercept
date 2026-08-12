"""Collect full-frame behavior-cloning episodes with the PNG expert."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..end_to_end.data import DATASET_SCHEMA_VERSION
from ..end_to_end.environment import EndToEndInterceptEnv


def collect_episode(env: EndToEndInterceptEnv):
    observation, info = env.reset()
    frames, actions = [], []
    future, risk, confidence, critic = [], [], [], []

    while True:
        action = info["teacher_action"]
        frames.append(observation["frames"][-1].copy())
        actions.append(action.copy())
        future.append(info["aux_targets"]["future_position"].copy())
        risk.append(info["aux_targets"]["collision_risk"])
        confidence.append(info["aux_targets"]["confidence"])
        critic.append(info["critic_obs"].copy())

        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    arrays = {
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.float32),
        "future_position": np.asarray(future, dtype=np.float32),
        "collision_risk": np.asarray(risk, dtype=np.float32),
        "confidence": np.asarray(confidence, dtype=np.float32),
        "critic_obs": np.asarray(critic, dtype=np.float32),
    }
    summary = {
        "length": len(frames),
        "outcome": info["outcome"],
        "mode": info["mode"],
        "min_dist": float(info["min_dist"]),
    }
    return arrays, summary


def prepare_output(output_dir: Path, overwrite: bool) -> Path:
    episodes_dir = output_dir / "episodes"
    existing = list(episodes_dir.glob("episode_*.npz")) if episodes_dir.exists() else []
    if (existing or (output_dir / "manifest.json").exists()) and not overwrite:
        raise FileExistsError(
            f"dataset already exists at {output_dir}; pass --overwrite to replace it")
    if overwrite:
        for path in existing:
            path.unlink()
        manifest = output_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()
    episodes_dir.mkdir(parents=True, exist_ok=True)
    return episodes_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/e2e_bc")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    episode_count = args.episodes or cfg.end_to_end.bc.episodes
    output_dir = Path(args.out)
    episodes_dir = prepare_output(output_dir, args.overwrite)
    env = EndToEndInterceptEnv(cfg, mode=args.mode, seed=args.seed)

    summaries = []
    total_frames = 0
    for episode in range(episode_count):
        arrays, summary = collect_episode(env)
        final_path = episodes_dir / f"episode_{episode:06d}.npz"
        temporary_path = episodes_dir / f".episode_{episode:06d}.tmp.npz"
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(final_path)
        summaries.append(summary)
        total_frames += summary["length"]

        if (episode + 1) % 25 == 0 or episode + 1 == episode_count:
            hits = sum(item["outcome"] == "hit" for item in summaries)
            print(
                f"[{episode + 1}/{episode_count}] frames={total_frames:,} "
                f"teacher_hit={hits / (episode + 1):.1%}")

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "phase": 3,
        "observation": "full_rgb_frame_history",
        "action_protocol": "body_velocity_yaw_rate_v1",
        "episodes": episode_count,
        "frames": total_frames,
        "image_width": int(cfg.end_to_end.render.image_width),
        "image_height": int(cfg.end_to_end.render.image_height),
        "history_frames": int(cfg.end_to_end.model.history_frames),
        "velocity_max": float(cfg.end_to_end.action.velocity_max),
        "yaw_rate_max": float(cfg.end_to_end.action.yaw_rate_max),
        "seed": args.seed,
        "summaries": summaries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    env.close()
    print(f"saved dataset to {output_dir} ({total_frames:,} frames)")


if __name__ == "__main__":
    main()
