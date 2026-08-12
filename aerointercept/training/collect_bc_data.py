"""BC 数据采集：PNG 老师在 Gym 内闭环 rollout

老师动作经过 encode→decode 往返后施加到环境（保证标签在动作空间内可实现，
即 DAgger 意义上的"可执行专家"）。逐集存储，train_bc 按集采样子序列。

用法:
  python -m aerointercept.training.collect_bc_data --episodes 2000 \\
      --out data/bc_dataset.npz
"""
import argparse
import os

import numpy as np

from ..config import load_config
from ..environments import InterceptEnv


def collect(cfg, episodes: int, seed: int = 0, mode: str = "mixed"):
    env = InterceptEnv(cfg, mode=mode, seed=seed)
    all_obs, all_act, ep_lens, outcomes, modes = [], [], [], [], []

    for ep in range(episodes):
        obs, info = env.reset()
        a = info["teacher_action"]
        ep_obs, ep_act = [obs], [a]
        while True:
            obs, r, term, trunc, info = env.step(a)
            a = info["teacher_action"]
            if term or trunc:
                outcomes.append(info["outcome"])
                modes.append(info["mode"])
                break
            ep_obs.append(obs)
            ep_act.append(a)
        all_obs.append(np.stack(ep_obs))
        all_act.append(np.stack(ep_act))
        ep_lens.append(len(ep_obs))
        if (ep + 1) % 200 == 0:
            hits = sum(1 for o in outcomes if o == "hit")
            print(f"[{ep+1}/{episodes}] teacher hit rate {hits/(ep+1):.1%}, "
                  f"total transitions {sum(ep_lens):,}")

    return all_obs, all_act, ep_lens, outcomes, modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/bc_dataset.npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    episodes = args.episodes or cfg.bc.episodes

    all_obs, all_act, ep_lens, outcomes, modes = collect(
        cfg, episodes, seed=args.seed, mode=args.mode)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.concatenate(all_obs).astype(np.float32),
        act=np.concatenate(all_act).astype(np.float32),
        ep_lens=np.array(ep_lens, dtype=np.int64),
        outcomes=np.array(outcomes),
        modes=np.array(modes),
    )
    hits = sum(1 for o in outcomes if o == "hit")
    print(f"\n保存到 {args.out}")
    print(f"  episodes={episodes}  transitions={sum(ep_lens):,}  "
          f"teacher hit rate={hits/episodes:.1%}")


if __name__ == "__main__":
    main()
