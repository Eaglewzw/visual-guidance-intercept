"""阶段二 Gym 评估：加载 V2 策略（CNN+GRU）与 PNG/V1 对比

用法:
  python -m guidance_rl.eval.eval_v2 --policy checkpoints/rl_policy_v2.pt \\
      --episodes 100 --out results/rl_v2.csv
  python -m guidance_rl.eval.eval_v2 --policy png --episodes 100 --out results/png_v2_env.csv
"""
import argparse
import csv
import os

import numpy as np
import torch

from ..config import load_config
from ..envs import InterceptEnvV2

MODES = ["circle", "sinusoidal", "random_walk", "hover_escape"]


def run_eval(policy_path: str, cfg, episodes: int, seed: int, modes):
    """policy_path='png' 或 V2 checkpoint 路径"""
    use_png = (policy_path == "png")
    if not use_png:
        ckpt = torch.load(policy_path, map_location="cpu")
        from ..models.policy_v2 import ActorV2
        model = ActorV2(pretrained_cnn=False)
        model_dict = {k.replace("actor.", ""): v for k, v in ckpt["model"].items()
                      if k.startswith("actor.")}
        model.load_state_dict(model_dict, strict=False)
        model.eval()
        h = model.initial_hidden(1)

    rows = []
    for mode in modes:
        env = InterceptEnvV2(cfg, mode=mode, seed=seed)
        for ep in range(episodes):
            obs, info = env.reset()
            if not use_png:
                h = model.initial_hidden(1)
            while True:
                if use_png:
                    a = info["teacher_action"]
                else:
                    with torch.no_grad():
                        img = torch.from_numpy(obs["image"]).unsqueeze(0)
                        ego = torch.from_numpy(obs["ego_state"]).unsqueeze(0)
                        act, _, _, h = model(img, ego, h, return_aux=True)
                    a = act.squeeze(0).numpy()
                obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    rows.append({
                        "mode": mode, "episode": ep,
                        "outcome": info["outcome"],
                        "hit": int(info["outcome"] == "hit"),
                        "min_dist": round(info["min_dist"], 4),
                        "time_s": round(info["episode_steps"] * cfg.dynamics.dt, 2),
                    })
                    break
    return rows


def print_table(rows, label):
    print(f"\n===== {label} =====")
    print(f"{'mode':14s} {'hit_rate':>9s} {'min_dist(m)':>12s} {'hit_time(s)':>12s}")
    for mode in sorted(set(r["mode"] for r in rows)):
        sub = [r for r in rows if r["mode"] == mode]
        hits = [r for r in sub if r["hit"]]
        hr = len(hits) / len(sub)
        md = np.mean([r["min_dist"] for r in sub])
        ht = np.mean([r["time_s"] for r in hits]) if hits else float("nan")
        print(f"{mode:14s} {hr:9.1%} {md:12.3f} {ht:12.2f}")
    print(f"{'TOTAL':14s} {sum(r['hit'] for r in rows)/len(rows):9.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--policy", required=True,
                        help="'png' 或 V2 checkpoint 路径 (.pt)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = run_eval(args.policy, cfg, args.episodes, args.seed, args.modes)
    print_table(rows, args.policy)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV 已保存: {args.out}")


if __name__ == "__main__":
    main()
