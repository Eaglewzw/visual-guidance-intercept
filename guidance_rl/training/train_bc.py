"""行为克隆训练：PNG 老师 → GRU 策略

按集切子序列（seq_len=64）训练 GRU，损失 = tanh 均值与老师动作的 MSE。
隐藏状态从零开始（子序列首帧前的历史不可见）——与部署时 INTERCEPT
进入瞬间的状态一致，不构成训练/部署偏差。

用法:
  python -m guidance_rl.training.train_bc --data data/bc_dataset.npz \\
      --out checkpoints/bc_policy.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from ..config import load_config
from ..models.policy import ActorCritic


class BCDataset:
    """按集存储，随机采样定长子序列"""

    def __init__(self, path: str, seq_len: int, val_frac: float, seed: int = 0):
        data = np.load(path, allow_pickle=True)
        obs, act, ep_lens = data["obs"], data["act"], data["ep_lens"]
        # 还原各集边界
        bounds = np.concatenate([[0], np.cumsum(ep_lens)])
        episodes = [(obs[a:b], act[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        # 过滤过短的集
        episodes = [e for e in episodes if len(e[0]) >= seq_len // 2]

        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)
        n_val = max(1, int(len(episodes) * val_frac))
        self.val_eps = episodes[:n_val]
        self.train_eps = episodes[n_val:]
        self.seq_len = seq_len
        self.rng = rng
        print(f"dataset: {len(self.train_eps)} train / {len(self.val_eps)} val episodes")

    def _sample_from(self, episodes, batch_size):
        L = self.seq_len
        obs_b, act_b = [], []
        for _ in range(batch_size):
            o, a = episodes[self.rng.integers(len(episodes))]
            if len(o) <= L:
                # 不足 seq_len 的集：前向 padding 重复首帧（mask 简化处理）
                pad = L - len(o)
                o = np.concatenate([np.repeat(o[:1], pad, axis=0), o])
                a = np.concatenate([np.repeat(a[:1], pad, axis=0), a])
            else:
                s = self.rng.integers(len(o) - L + 1)
                o, a = o[s:s + L], a[s:s + L]
            obs_b.append(o)
            act_b.append(a)
        return (torch.from_numpy(np.stack(obs_b)),
                torch.from_numpy(np.stack(act_b)))

    def sample_train(self, batch_size):
        return self._sample_from(self.train_eps, batch_size)

    def sample_val(self, batch_size):
        return self._sample_from(self.val_eps, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="data/bc_dataset.npz")
    parser.add_argument("--out", default="checkpoints/bc_policy.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)

    ds = BCDataset(args.data, cfg.bc.seq_len, cfg.bc.val_frac, seed=args.seed)
    model = ActorCritic(cfg.model).to(args.device)
    actor = model.actor
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.bc.lr)

    steps_per_epoch = max(1, sum(len(o) for o, _ in ds.train_eps)
                          // (cfg.bc.batch_size * cfg.bc.seq_len))
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(cfg.bc.epochs):
        actor.train()
        train_loss = 0.0
        for _ in range(steps_per_epoch):
            obs, act = ds.sample_train(cfg.bc.batch_size)
            obs, act = obs.to(args.device), act.to(args.device)
            h0 = actor.initial_hidden(obs.shape[0], args.device)
            mean, _ = actor(obs, h0)
            loss = nn.functional.mse_loss(mean, act)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= steps_per_epoch

        actor.eval()
        with torch.no_grad():
            obs, act = ds.sample_val(min(512, cfg.bc.batch_size * 2))
            obs, act = obs.to(args.device), act.to(args.device)
            h0 = actor.initial_hidden(obs.shape[0], args.device)
            mean, _ = actor(obs, h0)
            val_loss = nn.functional.mse_loss(mean, act).item()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg_model": dict(cfg.model),
                        "val_loss": val_loss}, args.out)
            tag = "  <- saved"
        print(f"epoch {epoch+1:3d}/{cfg.bc.epochs}  "
              f"train {train_loss:.5f}  val {val_loss:.5f}{tag}")

    print(f"\nBC 完成，最优 val MSE = {best_val:.5f}，已保存 {args.out}")
    print("下一步: python -m guidance_rl.evaluation.eval_gym --policy", args.out)


if __name__ == "__main__":
    main()
