"""Gym 内批量评估：PNG / BC / RL × 各运动模式

指标与 vpng_intercept_stats CSV 口径一致：命中率、最近接距离、拦截时间、丢失率。

用法:
  python -m aerointercept.evaluation.eval_gym --policy png
  python -m aerointercept.evaluation.eval_gym --policy checkpoints/bc_policy.pt
  python -m aerointercept.evaluation.eval_gym --policy checkpoints/rl_policy.pt \\
      --episodes 200 --out results/rl_eval.csv --save-traj 5
"""
import argparse
import csv
import os

import numpy as np
import torch

from ..config import load_config
from ..environments import InterceptEnv
from ..models.policy import ActorCritic

MODES = ["circle", "sinusoidal", "random_walk", "hover_escape"]


class PolicyRunner:
    """统一接口：png 老师 或 checkpoint 策略（确定性，取 tanh 均值）"""

    def __init__(self, policy: str, cfg, device="cpu"):
        self.kind = "png" if policy == "png" else "ckpt"
        if self.kind == "ckpt":
            ckpt = torch.load(policy, map_location=device)
            self.model = ActorCritic(cfg.model)
            self.model.load_state_dict(ckpt["model"], strict=False)
            self.model.eval().to(device)
            self.device = device

    def reset(self):
        if self.kind == "ckpt":
            self.h = self.model.actor.initial_hidden(1, self.device)

    def act(self, obs, info):
        if self.kind == "png":
            return info["teacher_action"]
        with torch.no_grad():
            o = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            mean, self.h = self.model.actor(o.unsqueeze(1), self.h)
        return mean.squeeze().cpu().numpy()


def run_eval(policy: str, cfg, episodes: int, seed: int, modes,
             save_traj: int = 0):
    runner = PolicyRunner(policy, cfg)
    rows = []
    trajs = []

    for mode in modes:
        env = InterceptEnv(cfg, mode=mode, seed=seed)
        for ep in range(episodes):
            obs, info = env.reset()
            runner.reset()
            record = ep < save_traj
            traj = {"i": [env.dynamics.pos.copy()],
                    "t": [env.target_pos.copy()]} if record else None
            while True:
                a = runner.act(obs, info)
                obs, r, term, trunc, info = env.step(a)
                if record:
                    traj["i"].append(env.dynamics.pos.copy())
                    traj["t"].append(env.target_pos.copy())
                if term or trunc:
                    rows.append({
                        "mode": mode, "episode": ep,
                        "outcome": info["outcome"],
                        "hit": int(info["outcome"] == "hit"),
                        "min_dist": round(info["min_dist"], 4),
                        "time_s": round(info["episode_steps"] * cfg.dynamics.dt, 2),
                    })
                    if record:
                        traj["outcome"] = info["outcome"]
                        traj["mode"] = mode
                        trajs.append(traj)
                    break
    return rows, trajs


def print_table(rows, label):
    print(f"\n===== {label} =====")
    print(f"{'mode':14s} {'hit_rate':>9s} {'min_dist(m)':>12s} "
          f"{'hit_time(s)':>12s} {'fov_lost':>9s}")
    for mode in sorted(set(r["mode"] for r in rows)):
        sub = [r for r in rows if r["mode"] == mode]
        hits = [r for r in sub if r["hit"]]
        lost = sum(1 for r in sub if r["outcome"] == "fov_lost")
        hr = len(hits) / len(sub)
        md = np.mean([r["min_dist"] for r in sub])
        ht = np.mean([r["time_s"] for r in hits]) if hits else float("nan")
        print(f"{mode:14s} {hr:9.1%} {md:12.3f} {ht:12.2f} {lost:9d}")
    total_hr = sum(r["hit"] for r in rows) / len(rows)
    print(f"{'TOTAL':14s} {total_hr:9.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--policy", required=True,
                        help="'png' 或 checkpoint 路径 (.pt)")
    parser.add_argument("--episodes", type=int, default=100, help="每模式集数")
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default=None, help="结果 CSV 路径")
    parser.add_argument("--save-traj", type=int, default=0,
                        help="每模式保存前 N 集轨迹（plot_results 用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows, trajs = run_eval(args.policy, cfg, args.episodes, args.seed,
                           args.modes, args.save_traj)
    print_table(rows, args.policy)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV 已保存: {args.out}")
        if trajs:
            traj_path = args.out.replace(".csv", "_traj.npz")
            np.savez_compressed(
                traj_path,
                **{f"traj{k}_{key}": np.array(t[key])
                   for k, t in enumerate(trajs) for key in ("i", "t")},
                meta=np.array([(t["mode"], t["outcome"]) for t in trajs]))
            print(f"轨迹已保存: {traj_path}")


if __name__ == "__main__":
    main()
