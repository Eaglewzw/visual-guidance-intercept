"""Evaluate PNG or the full-frame policy with common intercept metrics."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import DotDict, load_config
from ..end_to_end.environment import EndToEndInterceptEnv
from ..end_to_end.policy import EndToEndActorCritic


DEFAULT_MODES = ["circle", "sinusoidal", "random_walk", "hover_escape"]


class PolicyRunner:
    def __init__(self, policy_path: str, cfg, device: str):
        self.png = policy_path == "png"
        self.device = torch.device(device)
        if not self.png:
            checkpoint = torch.load(policy_path, map_location=self.device)
            if checkpoint.get("phase") != 3:
                raise ValueError("checkpoint is not an end-to-end image policy")
            model_config = checkpoint.get(
                "model_config", dict(cfg.end_to_end.model))
            if int(model_config["history_frames"]) != int(
                    cfg.end_to_end.model.history_frames):
                raise ValueError(
                    "checkpoint/config history_frames mismatch; pass the "
                    "training config with --config")
            for key in ("image_width", "image_height"):
                stored = checkpoint.get("render_config", {}).get(key)
                if stored is not None and int(stored) != int(
                        cfg.end_to_end.render[key]):
                    raise ValueError(
                        f"checkpoint/config {key} mismatch; pass the training config")
            stored_action = checkpoint.get("action_config")
            if (stored_action is not None
                    and stored_action != dict(cfg.end_to_end.action)):
                raise ValueError(
                    "checkpoint/config action protocol mismatch; pass the "
                    "training config")
            stored_labels = checkpoint.get("label_config")
            if (stored_labels is not None
                    and stored_labels != dict(cfg.end_to_end.labels)):
                raise ValueError(
                    "checkpoint/config auxiliary label contract mismatch; "
                    "pass the training config")
            model = EndToEndActorCritic(DotDict(model_config))
            model.load_state_dict(checkpoint["model"], strict=True)
            self.actor = model.actor.eval().to(self.device)

    def act(self, observation, info):
        if self.png:
            return info["teacher_action"], 1.0, 0.0, None
        frames = torch.from_numpy(observation["frames"]).unsqueeze(0).to(
            self.device)
        with torch.no_grad():
            action, _, risk_logit, confidence_logit, attention = self.actor(frames)
        return (
            action[0].cpu().numpy(),
            float(torch.sigmoid(confidence_logit[0])),
            float(torch.sigmoid(risk_logit[0])),
            attention[0].cpu().numpy(),
        )


def save_attention_overlay(frame_chw, attention, path: Path):
    if attention is None:
        return
    frame = frame_chw.transpose(1, 2, 0)
    heat = cv2.resize(attention, (frame.shape[1], frame.shape[0]))
    heat = heat - heat.min()
    heat = heat / max(float(heat.max()), 1e-8)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(frame, 0.65, heat, 0.35, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def run_evaluation(policy, cfg, episodes, seed, modes, device,
                   attention_dir=None, attention_count=0,
                   apply_fallback=False):
    runner = PolicyRunner(policy, cfg, device)
    rows = []
    overlays_saved = 0
    threshold = float(cfg.end_to_end.safety.confidence_threshold)

    for mode_index, mode in enumerate(modes):
        env = EndToEndInterceptEnv(cfg, mode=mode, seed=seed)
        for episode in range(episodes):
            observation, info = env.reset(
                seed=seed + mode_index * 100_000 + episode)
            confidences, risks = [], []
            unsafe_frames = 0
            while True:
                action, confidence, risk, attention = runner.act(
                    observation, info)
                confidences.append(confidence)
                risks.append(risk)
                below_threshold = confidence < threshold
                unsafe_frames += int(below_threshold)
                if below_threshold and apply_fallback and not runner.png:
                    action = info["teacher_action"]
                if (attention_dir is not None
                        and overlays_saved < attention_count
                        and attention is not None):
                    save_attention_overlay(
                        observation["frames"][-1], attention,
                        Path(attention_dir) / f"attention_{overlays_saved:04d}.png")
                    overlays_saved += 1
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    steps = int(info["episode_steps"])
                    rows.append({
                        "mode": mode,
                        "episode": episode,
                        "outcome": info["outcome"],
                        "hit": int(info["outcome"] == "hit"),
                        "min_dist": round(float(info["min_dist"]), 4),
                        "time_s": round(steps * cfg.dynamics.dt, 3),
                        "mean_confidence": round(float(np.mean(confidences)), 4),
                        "mean_collision_risk": round(float(np.mean(risks)), 4),
                        "fallback_fraction": round(unsafe_frames / steps, 4),
                    })
                    break
        env.close()
    return rows


def print_summary(rows, label):
    print(f"\n===== {label} =====")
    print(
        f"{'mode':14s} {'hit_rate':>9s} {'min_dist':>10s} "
        f"{'hit_time':>10s} {'fallback':>10s}")
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        hits = [row for row in subset if row["hit"]]
        hit_rate = len(hits) / len(subset)
        min_distance = np.mean([row["min_dist"] for row in subset])
        hit_time = np.mean([row["time_s"] for row in hits]) if hits else np.nan
        fallback = np.mean([row["fallback_fraction"] for row in subset])
        print(
            f"{mode:14s} {hit_rate:9.1%} {min_distance:10.3f} "
            f"{hit_time:10.2f} {fallback:10.1%}")
    print(f"{'TOTAL':14s} {np.mean([row['hit'] for row in rows]):9.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--policy", required=True, help="'png' or end-to-end checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--attention-dir", default=None)
    parser.add_argument("--attention-count", type=int, default=0)
    parser.add_argument(
        "--apply-fallback", action="store_true",
        help="apply the PNG teacher when visual confidence is below threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = run_evaluation(
        args.policy, cfg, args.episodes, args.seed, args.modes, args.device,
        args.attention_dir, args.attention_count, args.apply_fallback)
    print_summary(rows, args.policy)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {output}")


if __name__ == "__main__":
    main()
