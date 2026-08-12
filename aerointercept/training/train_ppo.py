"""PPO 微调主程序（BC 热启动）

用法:
  python -m aerointercept.training.train_ppo --bc-init checkpoints/bc_policy.pt \\
      --out checkpoints/rl_policy.pt --logdir runs/ppo

监控: tensorboard --logdir runs/
关键曲线: rollout/hit_rate（命中率，核心指标）、rollout/ep_reward
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..config import load_config
from ..environments import VecInterceptEnv
from ..models.policy import ActorCritic
from .ppo import PPO, RolloutBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--bc-init", default=None, help="BC 预训练权重（强烈推荐）")
    parser.add_argument("--out", default="checkpoints/rl_policy.pt")
    parser.add_argument("--logdir", default="runs/ppo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="mixed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    p = cfg.ppo
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    env = VecInterceptEnv(p.num_envs, cfg, mode=args.mode, seed=args.seed)
    model = ActorCritic(cfg.model).to(device)

    if args.bc_init:
        ckpt = torch.load(args.bc_init, map_location=device)
        # BC 只训练了 actor；critic 保持随机初始化由 PPO 自行学习
        missing = model.load_state_dict(ckpt["model"], strict=False)
        print(f"BC 热启动: {args.bc_init} (val_loss={ckpt.get('val_loss'):.5f})")
        if missing.missing_keys:
            print(f"  critic 等未初始化参数: {len(missing.missing_keys)} 个（正常）")

    ppo = PPO(model, p, device)
    writer = SummaryWriter(args.logdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    obs_np, priv_np = env.reset()
    obs = torch.from_numpy(obs_np).to(device)
    priv = torch.from_numpy(priv_np).to(device)
    h = model.actor.initial_hidden(p.num_envs, device)

    ep_rewards = np.zeros(p.num_envs)
    recent_rewards = deque(maxlen=200)
    recent_outcomes = deque(maxlen=200)
    best_hit_rate = -1.0
    global_step = 0
    num_updates = p.total_steps // (p.num_envs * p.rollout_steps)
    t0 = time.time()

    for update in range(1, num_updates + 1):
        buf = RolloutBuffer(p.rollout_steps, p.num_envs,
                            cfg.model.obs_dim, cfg.model.priv_dim,
                            cfg.model.act_dim, cfg.model.gru_hidden, device)
        buf.h_init = h.detach().clone()

        # ---------------- rollout ----------------
        for _ in range(p.rollout_steps):
            with torch.no_grad():
                act, logprob, value, h = model.act(
                    obs, priv, h,
                    log_std_min=p.log_std_min, log_std_max=p.log_std_max)
            obs_np, priv_np, rew_np, done_np, infos = env.step(
                act.cpu().numpy())

            ep_rewards += rew_np
            for i, info in enumerate(infos):
                if "final" in info:
                    recent_rewards.append(ep_rewards[i])
                    recent_outcomes.append(info["final"]["outcome"])
                    ep_rewards[i] = 0.0

            done = torch.from_numpy(done_np).to(device)
            buf.add(obs, priv, act, logprob,
                    torch.from_numpy(rew_np).to(device), done, value)
            # episode 结束的环境清零隐藏状态
            h = h * (1.0 - done).view(1, -1, 1)

            obs = torch.from_numpy(obs_np).to(device)
            priv = torch.from_numpy(priv_np).to(device)
            global_step += p.num_envs

        with torch.no_grad():
            last_value = model.critic(obs, priv)
        buf.compute_gae(last_value, p.gamma, p.gae_lambda)

        # ---------------- update ----------------
        stats = ppo.update(buf)
        h = h.detach()

        # ---------------- logging ----------------
        if recent_outcomes:
            hit_rate = sum(1 for o in recent_outcomes if o == "hit") / len(recent_outcomes)
            mean_rew = float(np.mean(recent_rewards))
        else:
            hit_rate, mean_rew = 0.0, 0.0

        writer.add_scalar("rollout/hit_rate", hit_rate, global_step)
        writer.add_scalar("rollout/ep_reward", mean_rew, global_step)
        for k, v in stats.items():
            writer.add_scalar(f"train/{k}", v, global_step)
        writer.add_scalar("train/log_std",
                          model.actor.log_std.mean().item(), global_step)

        if update % 10 == 0:
            sps = global_step / (time.time() - t0)
            print(f"update {update:4d}/{num_updates}  step {global_step/1e6:.2f}M  "
                  f"hit {hit_rate:.1%}  rew {mean_rew:7.2f}  "
                  f"kl {stats['approx_kl']:.4f}  {sps:,.0f} sps")

        # 按近期命中率保存最优
        if len(recent_outcomes) >= 100 and hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            torch.save({"model": model.state_dict(), "cfg_model": dict(cfg.model),
                        "hit_rate": hit_rate, "global_step": global_step}, args.out)

        if update % 50 == 0:
            torch.save({"model": model.state_dict(), "cfg_model": dict(cfg.model),
                        "hit_rate": hit_rate, "global_step": global_step},
                       args.out.replace(".pt", "_last.pt"))

    writer.close()
    if best_hit_rate < 0:
        # episode 样本不足，未触发按命中率保存，落一个最终权重
        torch.save({"model": model.state_dict(), "cfg_model": dict(cfg.model),
                    "hit_rate": None, "global_step": global_step}, args.out)
        print(f"\nPPO 完成（episode 样本不足，保存最终权重），已保存 {args.out}")
    else:
        print(f"\nPPO 完成，最优近期命中率 {best_hit_rate:.1%}，已保存 {args.out}")
    print("下一步: python -m aerointercept.evaluation.eval_gym --policy", args.out)


if __name__ == "__main__":
    main()
