"""阶段二 PPO 微调（BC 热启动 + 图像输入）

Rollout 时图像存储在 CPU uint8 buffer（不占用 GPU），更新时按 minibatch 搬运。
默认 8 envs × 128 steps = 1024 步/更新，图像总内存 ~127 MB。

用法:
  python -m guidance_rl.train.train_ppo_v2 --bc-init checkpoints/bc_policy_v2.pt \\
      --out checkpoints/rl_policy_v2.pt --logdir runs/ppo_v2
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..config import load_config
from ..envs import VecInterceptEnvV2
from ..models.policy_v2 import ActorCriticV2
from ..train.ppo import PPO   # 复用阶段一 GAE + clip update

CROP_SIZE = 288
EGO_DIM = 8


class RolloutBufferV2:
    """CPU uint8 图像存储 + GPU tensor 其他字段"""

    def __init__(self, T, N, obs_dim, priv_dim, act_dim, device):
        self.T, self.N = T, N
        self.device = device
        # 图像存在 CPU（uint8 省内存），其余在 GPU
        self.images = np.zeros((T, N, 3, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        self.ego = torch.zeros(T, N, EGO_DIM, device=device)
        self.gt_obs = torch.zeros(T, N, obs_dim, device=device)
        self.priv = torch.zeros(T, N, priv_dim, device=device)
        self.act = torch.zeros(T, N, act_dim, device=device)
        self.logprob = torch.zeros(T, N, device=device)
        self.reward = torch.zeros(T, N, device=device)
        self.done = torch.zeros(T, N, device=device)
        self.value = torch.zeros(T, N, device=device)
        self.h_init = None
        self.step = 0

    def add(self, image_np, ego, gt_obs, priv, act, logprob, reward, done, value):
        t = self.step
        self.images[t] = image_np
        self.ego[t] = ego
        self.gt_obs[t] = gt_obs
        self.priv[t] = priv
        self.act[t] = act
        self.logprob[t] = logprob
        self.reward[t] = reward
        self.done[t] = done
        self.value[t] = value
        self.step += 1

    def compute_gae(self, last_value, gamma, lam):
        adv = torch.zeros_like(self.reward)
        gae = torch.zeros(self.N, device=self.reward.device)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.value[t + 1]
            nonterminal = 1.0 - self.done[t]
            delta = self.reward[t] + gamma * next_value * nonterminal - self.value[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
        self.adv = adv
        self.ret = adv + self.value
        self.step = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--bc-init", default=None)
    parser.add_argument("--out", default="checkpoints/rl_policy_v2.pt")
    parser.add_argument("--logdir", default="runs/ppo_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="envs 数（图像渲染用 OpenCV 后已足够快）")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率（覆盖 config）")
    parser.add_argument("--freeze-cnn", action="store_true",
                        help="冻结 MobileNetV3 骨干，只训 GRU+投影头")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["ppo"]["num_envs"] = args.num_envs
    cfg["ppo"]["rollout_steps"] = args.rollout_steps
    if args.lr is not None:
        cfg["ppo"]["lr"] = args.lr
    p = cfg.ppo
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**3
        print(f"GPU 显存: {used:.1f}/{total/1024**3:.1f} GiB 已用, "
              f"{free/1024**3:.1f} GiB 空闲")
        if free < 1.5 * 1024**3:
            print("⚠️  显存紧张，建议先清理其他进程 (nvidia-smi) 或 --num-envs 4")

    env = VecInterceptEnvV2(p.num_envs, cfg, mode=args.mode, seed=args.seed)
    model = ActorCriticV2(cfg.model, pretrained_cnn=True).to(device)

    if args.bc_init:
        ckpt = torch.load(args.bc_init, map_location=device)
        missing = model.load_state_dict(ckpt["model"], strict=False)
        print(f"BC 热启动: {args.bc_init} (loss={ckpt.get('bc_loss'):.5f})")
        if missing.missing_keys:
            print(f"  未初始化: {len(missing.missing_keys)} 键（正常）")
        del ckpt
        torch.cuda.empty_cache()

    if args.freeze_cnn:
        # 冻结 MobileNetV3 骨干，只训 GRU + 投影头
        frozen_params = 0
        for name, param in model.actor.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
                frozen_params += param.numel()
        print(f"冻结 CNN 骨干: {frozen_params/1e6:.1f}M 参数  "
              f"可训: {sum(p.numel() for p in model.actor.parameters() if p.requires_grad)/1e6:.1f}M")

    ppo = PPO(model, p, device)
    print(f"PPO V2配置: lr={p.lr:.1e}  envs={p.num_envs}  rollout={p.rollout_steps}  "
          f"steps={p.total_steps/1e6:.1f}M  gamma={p.gamma}  clip={p.clip_eps}")
    writer = SummaryWriter(args.logdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    images_np, ego_np, gt_obs_np, priv_np = env.reset()
    ego = torch.from_numpy(ego_np).to(device)
    gt_obs = torch.from_numpy(gt_obs_np).to(device)
    priv = torch.from_numpy(priv_np).to(device)
    h = model.actor.initial_hidden(p.num_envs, device)

    # pinned memory 图像缓冲区 (pin 一次, 后续转移用 DMA 零阻塞)
    if device == "cuda":
        pin_size = (p.num_envs, 3, CROP_SIZE, CROP_SIZE)
        img_pinned = torch.empty(pin_size, dtype=torch.uint8, pin_memory=True)
    else:
        img_pinned = None

    ep_rewards = np.zeros(p.num_envs)
    recent_rewards = deque(maxlen=200)
    recent_outcomes = deque(maxlen=200)
    best_hit_rate = -1.0
    global_step = 0
    total_steps = p.total_steps
    num_updates = total_steps // (p.num_envs * p.rollout_steps)
    t0 = time.time()

    for update in range(1, num_updates + 1):
        buf = RolloutBufferV2(p.rollout_steps, p.num_envs,
                              cfg.model.obs_dim, cfg.model.priv_dim,
                              cfg.model.act_dim, device)
        buf.h_init = h.detach().clone()

        # ---------------- rollout ----------------
        for _ in range(p.rollout_steps):
            with torch.no_grad():
                # pinned DMA 传输: copy numpy → pinned tensor → GPU(non_blocking)
                if img_pinned is not None:
                    img_pinned.copy_(torch.from_numpy(images_np), non_blocking=True)
                    img_t = img_pinned.to(device, non_blocking=True)
                else:
                    img_t = torch.from_numpy(images_np).to(device)
                act, logprob, value, h = model.act(img_t, ego, gt_obs, priv, h)
            images_next, ego_next, gt_obs_next, priv_next, rew_np, done_np, infos = env.step(
                act.cpu().numpy())

            ep_rewards += rew_np
            for i, info in enumerate(infos):
                if "final" in info:
                    recent_rewards.append(ep_rewards[i])
                    recent_outcomes.append(info["final"]["outcome"])
                    ep_rewards[i] = 0.0

            done = torch.from_numpy(done_np).to(device)
            buf.add(images_np, ego, gt_obs, priv,
                    act, logprob,
                    torch.from_numpy(rew_np).to(device), done, value)
            h = h * (1.0 - done).view(1, -1, 1)

            images_np, ego_np, gt_obs_np, priv_np = images_next, ego_next, gt_obs_next, priv_next
            ego = torch.from_numpy(ego_np).to(device)
            gt_obs = torch.from_numpy(gt_obs_np).to(device)
            priv = torch.from_numpy(priv_np).to(device)
            global_step += p.num_envs

        with torch.no_grad():
            last_value = model.critic(gt_obs, priv)
        buf.compute_gae(last_value, p.gamma, p.gae_lambda)

        # ---------------- update (minibatch 更新，避免 OOM) ----------------
        N, T = p.num_envs, p.rollout_steps

        # 按 env 维切 minibatch：每批只搬运 mb_envs 个环境的图像到 GPU
        mb_size = max(1, N // p.num_minibatches)
        adv_flat = buf.adv.transpose(0, 1).contiguous()       # [N, T]
        ret_flat = buf.ret.transpose(0, 1).contiguous()
        old_lp_flat = buf.logprob.transpose(0, 1).contiguous()
        adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # 累积 minibatch 统计
        mb_pi_loss = mb_vf_loss = mb_ent = 0.0
        mb_kl = mb_clip = 0.0

        for mb_start in range(0, N, mb_size):
            mb_end = min(mb_start + mb_size, N)
            mb_slice = slice(mb_start, mb_end)

            imgs_mb = torch.from_numpy(
                buf.images[:, mb_slice].transpose(1, 0, 2, 3, 4).copy()
            ).to(device)

            ego_mb = buf.ego[:, mb_slice].transpose(0, 1).contiguous()
            gt_obs_mb = buf.gt_obs[:, mb_slice].transpose(0, 1).contiguous()
            priv_mb = buf.priv[:, mb_slice].transpose(0, 1).contiguous()
            act_mb = buf.act[:, mb_slice].transpose(0, 1).contiguous()
            done_mb = buf.done[:, mb_slice].transpose(0, 1).contiguous()
            old_lp_mb = old_lp_flat[mb_slice]
            adv_mb = adv_norm[mb_slice]
            ret_mb = ret_flat[mb_slice]
            h0_mb = buf.h_init[:, mb_slice]

            mean_mb, _ = model.actor.forward_masked(
                imgs_mb, ego_mb, done_mb, h0_mb)

            log_std = model.actor.log_std.clamp(p.log_std_min, p.log_std_max)
            std = log_std.exp().unsqueeze(0).unsqueeze(0).expand(
                mb_end - mb_start, T, -1)
            dist = torch.distributions.Normal(mean_mb, std)
            logprob_mb = dist.log_prob(act_mb).sum(-1)
            entropy_mb = dist.entropy().sum(-1)

            ratio_mb = (logprob_mb - old_lp_mb).exp()
            surr1 = ratio_mb * adv_mb
            surr2 = torch.clamp(ratio_mb, 1 - p.clip_eps, 1 + p.clip_eps) * adv_mb
            pi_loss = -torch.min(surr1, surr2).mean()

            value_pred = model.critic(gt_obs_mb, priv_mb)
            vf_loss = 0.5 * (value_pred - ret_mb).pow(2).mean()

            ent = entropy_mb.mean()
            loss = pi_loss + p.vf_coef * vf_loss - p.ent_coef * ent

            ppo.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p.max_grad_norm)
            ppo.opt.step()

            with torch.no_grad():
                mb_kl += ((ratio_mb - 1) - ratio_mb.log()).mean().item()
                mb_clip += ((ratio_mb - 1).abs() > p.clip_eps).float().mean().item()
            mb_pi_loss += pi_loss.item()
            mb_vf_loss += vf_loss.item()
            mb_ent += ent.item()

            del imgs_mb

        n_mb = (N + mb_size - 1) // mb_size
        pi_loss_val = mb_pi_loss / n_mb
        vf_loss_val = mb_vf_loss / n_mb
        ent_val = mb_ent / n_mb
        approx_kl = mb_kl / n_mb
        clip_frac = mb_clip / n_mb

        del buf
        if update % 5 == 0:
            torch.cuda.empty_cache()

        h = h.detach()

        # ---------------- logging ----------------
        if recent_outcomes:
            hit_rate = sum(1 for o in recent_outcomes if o == "hit") / len(recent_outcomes)
            mean_rew = float(np.mean(recent_rewards))
        else:
            hit_rate, mean_rew = 0.0, 0.0

        writer.add_scalar("rollout/hit_rate", hit_rate, global_step)
        writer.add_scalar("rollout/ep_reward", mean_rew, global_step)
        writer.add_scalar("train/pi_loss", pi_loss_val, global_step)
        writer.add_scalar("train/vf_loss", vf_loss_val, global_step)
        writer.add_scalar("train/entropy", ent_val, global_step)
        writer.add_scalar("train/approx_kl", approx_kl, global_step)
        writer.add_scalar("train/clip_frac", clip_frac, global_step)
        writer.add_scalar("train/log_std", model.actor.log_std.mean().item(), global_step)

        if update % 10 == 0:
            sps = global_step / (time.time() - t0)
            print(f"update {update:4d}/{num_updates}  "
                  f"hit {hit_rate:.1%}  rew {mean_rew:7.2f}  "
                  f"kl {approx_kl:.4f}  {sps:,.0f} sps")

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
        torch.save({"model": model.state_dict(), "cfg_model": dict(cfg.model),
                    "hit_rate": None, "global_step": global_step}, args.out)
    print(f"\nPPO V2 完成 → {args.out}")


if __name__ == "__main__":
    main()
