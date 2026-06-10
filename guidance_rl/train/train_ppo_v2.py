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
    parser.add_argument("--num-envs", type=int, default=8,
                        help="envs 数（图像大，比阶段一小）")
    parser.add_argument("--rollout-steps", type=int, default=128)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # 覆盖为图像友好的配置
    cfg["ppo"]["num_envs"] = args.num_envs
    cfg["ppo"]["rollout_steps"] = args.rollout_steps
    p = cfg.ppo
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    env = VecInterceptEnvV2(p.num_envs, cfg, mode=args.mode, seed=args.seed)
    model = ActorCriticV2(cfg.model, pretrained_cnn=True).to(device)

    if args.bc_init:
        ckpt = torch.load(args.bc_init, map_location=device)
        missing = model.load_state_dict(ckpt["model"], strict=False)
        print(f"BC 热启动: {args.bc_init} (loss={ckpt.get('bc_loss'):.5f})")
        if missing.missing_keys:
            print(f"  未初始化: {len(missing.missing_keys)} 键（正常）")

    ppo = PPO(model, p, device)
    writer = SummaryWriter(args.logdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    images_np, ego_np, gt_obs_np, priv_np = env.reset()
    ego = torch.from_numpy(ego_np).to(device)
    gt_obs = torch.from_numpy(gt_obs_np).to(device)
    priv = torch.from_numpy(priv_np).to(device)
    h = model.actor.initial_hidden(p.num_envs, device)

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

        # ---------------- update (简化版：整段 BPTT，不做 minibatch) ----------------
        # 图像量大，整批更新避免额外 CPU-GPU 搬运
        images_t = torch.from_numpy(buf.images[:, :].reshape(-1, 3, CROP_SIZE, CROP_SIZE)).to(device)
        ego_t = buf.ego.view(-1, EGO_DIM)
        gt_obs_t = buf.gt_obs.view(-1, buf.gt_obs.shape[-1])
        priv_t = buf.priv.view(-1, buf.priv.shape[-1])

        # 分配每步：action/adv/ret → [T,N,...] → [N,T,...]（Actor 序列）
        N, T = p.num_envs, p.rollout_steps
        images_seq = buf.images.transpose(1, 0, 2, 3, 4)  # [N, T, 3, 288, 288]
        images_seq = torch.from_numpy(images_seq.copy()).to(device)
        ego_seq = buf.ego.transpose(0, 1).contiguous()
        gt_obs_seq = buf.gt_obs.transpose(0, 1).contiguous()
        priv_seq = buf.priv.transpose(0, 1).contiguous()
        act_seq = buf.act.transpose(0, 1).contiguous()
        done_seq = buf.done.transpose(0, 1).contiguous()
        old_logprob_seq = buf.logprob.transpose(0, 1).contiguous()
        adv_seq = buf.adv.transpose(0, 1).contiguous()
        ret_seq = buf.ret.transpose(0, 1).contiguous()

        # 优势归一化
        adv = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

        # Actor 更新
        h0_seq = buf.h_init
        mean, _ = model.actor.forward_masked(
            images_seq, ego_seq, done_seq, h0_seq)
        log_std = model.actor.log_std.clamp(p.log_std_min, p.log_std_max)
        std = log_std.exp().unsqueeze(0).unsqueeze(0).expand(N, T, -1)
        dist = torch.distributions.Normal(mean, std)
        logprob = dist.log_prob(act_seq).sum(-1)
        entropy = dist.entropy().sum(-1)

        ratio = (logprob - old_logprob_seq).exp()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - p.clip_eps, 1 + p.clip_eps) * adv
        pi_loss = -torch.min(surr1, surr2).mean()

        # Critic 更新
        value_pred = model.critic(gt_obs_seq, priv_seq)
        vf_loss = 0.5 * (value_pred - ret_seq).pow(2).mean()

        ent = entropy.mean()
        loss = pi_loss + p.vf_coef * vf_loss - p.ent_coef * ent

        ppo.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p.max_grad_norm)
        ppo.opt.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - ratio.log()).mean().item()
            clip_frac = ((ratio - 1).abs() > p.clip_eps).float().mean().item()

        h = h.detach()

        # ---------------- logging ----------------
        if recent_outcomes:
            hit_rate = sum(1 for o in recent_outcomes if o == "hit") / len(recent_outcomes)
            mean_rew = float(np.mean(recent_rewards))
        else:
            hit_rate, mean_rew = 0.0, 0.0

        writer.add_scalar("rollout/hit_rate", hit_rate, global_step)
        writer.add_scalar("rollout/ep_reward", mean_rew, global_step)
        writer.add_scalar("train/pi_loss", pi_loss.item(), global_step)
        writer.add_scalar("train/vf_loss", vf_loss.item(), global_step)
        writer.add_scalar("train/entropy", ent.item(), global_step)
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
