"""评估结果可视化：多策略指标对比 + 3D 拦截轨迹

用法:
  python -m aerointercept.evaluation.plot_results \\
      --csv results/png_eval.csv results/bc_eval.csv results/rl_eval.csv \\
      --labels PNG BC RL --out results/plots
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_metrics(csv_paths, labels, out_dir):
    all_rows = [load_rows(p) for p in csv_paths]
    modes = sorted(set(r["mode"] for rows in all_rows for r in rows))
    x = np.arange(len(modes))
    width = 0.8 / len(labels)

    metrics = [
        ("hit_rate", "命中率", lambda sub: np.mean([float(r["hit"]) for r in sub])),
        ("min_dist", "平均最近接距离 (m)",
         lambda sub: np.mean([float(r["min_dist"]) for r in sub])),
        ("hit_time", "平均拦截时间 (s)",
         lambda sub: np.mean([float(r["time_s"]) for r in sub
                              if r["hit"] == "1"] or [np.nan])),
    ]

    for key, title, fn in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (rows, label) in enumerate(zip(all_rows, labels)):
            vals = []
            for mode in modes:
                sub = [r for r in rows if r["mode"] == mode]
                vals.append(fn(sub) if sub else np.nan)
            ax.bar(x + i * width, vals, width, label=label)
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(modes)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"compare_{key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"保存: {path}")


def plot_trajectories(traj_path, out_dir, max_plots=6):
    data = np.load(traj_path, allow_pickle=True)
    meta = data["meta"]
    n = min(len(meta), max_plots)
    for k in range(n):
        ti = data[f"traj{k}_i"]   # 拦截机 NED 轨迹
        tt = data[f"traj{k}_t"]   # 目标轨迹
        mode, outcome = meta[k]
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        # NED → 画图用 ENU 习惯（z 取负为高度）
        ax.plot(ti[:, 1], ti[:, 0], -ti[:, 2], "b-", label="Interceptor")
        ax.plot(tt[:, 1], tt[:, 0], -tt[:, 2], "r--", label="Target")
        ax.scatter(*[(ti[0, 1],), (ti[0, 0],), (-ti[0, 2],)], c="b", marker="^", s=60)
        ax.scatter(*[(tt[0, 1],), (tt[0, 0],), (-tt[0, 2],)], c="r", marker="o", s=60)
        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        ax.set_zlabel("Alt (m)")
        ax.set_title(f"{mode} — {outcome}")
        ax.legend()
        path = os.path.join(out_dir, f"traj_{k}_{mode}_{outcome}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"保存: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--traj", default=None, help="*_traj.npz 轨迹文件")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()
    assert len(args.csv) == len(args.labels)

    os.makedirs(args.out, exist_ok=True)
    # 中文字体回退（无中文字体时不报错）
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                       "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_metrics(args.csv, args.labels, args.out)
    if args.traj:
        plot_trajectories(args.traj, args.out)


if __name__ == "__main__":
    main()
