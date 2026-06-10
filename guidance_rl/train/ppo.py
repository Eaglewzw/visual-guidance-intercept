"""紧凑版 Recurrent PPO（非对称 Actor-Critic）

要点：
  - Actor 为 GRU，rollout 中逐步携带隐藏状态，episode 结束自动清零
  - Critic 为前馈网络，输入特权观测（仿真真值），无需循环
  - 更新时按环境维度切 minibatch，从 rollout 起始隐藏状态整段 BPTT 重放，
    forward_masked 在 done 处重置隐藏状态
  - GAE + clip 目标 + 熵正则 + KL 提前停止
"""
import numpy as np
import torch


class RolloutBuffer:
    """[T, N, ...] 布局"""

    def __init__(self, T, N, obs_dim, priv_dim, act_dim, gru_hidden, device):
        self.T, self.N = T, N
        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.priv = torch.zeros(T, N, priv_dim, device=device)
        self.act = torch.zeros(T, N, act_dim, device=device)
        self.logprob = torch.zeros(T, N, device=device)
        self.reward = torch.zeros(T, N, device=device)
        self.done = torch.zeros(T, N, device=device)     # 该步执行后 episode 是否结束
        self.value = torch.zeros(T, N, device=device)
        self.h_init = torch.zeros(1, N, gru_hidden, device=device)  # rollout 起始隐藏状态
        self.step = 0

    def add(self, obs, priv, act, logprob, reward, done, value):
        t = self.step
        self.obs[t] = obs
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


class PPO:
    def __init__(self, model, cfg_ppo, device):
        """model: ActorCritic; cfg_ppo: DotDict 的 ppo 节"""
        self.model = model
        self.cfg = cfg_ppo
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg_ppo.lr)

    def update(self, buf: RolloutBuffer):
        cfg = self.cfg
        # 优势归一化
        adv = (buf.adv - buf.adv.mean()) / (buf.adv.std() + 1e-8)

        N = buf.N
        idx_all = np.arange(N)
        mb_size = max(1, N // cfg.num_minibatches)
        stats = {"pi_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}

        early_stop = False
        for _ in range(cfg.epochs):
            if early_stop:
                break
            np.random.shuffle(idx_all)
            for s in range(0, N, mb_size):
                mb = idx_all[s:s + mb_size]
                # [T, mb, ...] → [mb, T, ...]
                obs = buf.obs[:, mb].transpose(0, 1)
                priv = buf.priv[:, mb].transpose(0, 1)
                act = buf.act[:, mb].transpose(0, 1)
                done = buf.done[:, mb].transpose(0, 1)
                old_logprob = buf.logprob[:, mb].transpose(0, 1)
                mb_adv = adv[:, mb].transpose(0, 1)
                mb_ret = buf.ret[:, mb].transpose(0, 1)
                h0 = buf.h_init[:, mb]

                logprob, entropy, value = self.model.evaluate(
                    obs, priv, act, done, h0,
                    log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max)

                ratio = (logprob - old_logprob).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
                pi_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * (value - mb_ret).pow(2).mean()
                ent = entropy.mean()

                loss = pi_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean().item()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
                stats["pi_loss"] += pi_loss.item()
                stats["vf_loss"] += vf_loss.item()
                stats["entropy"] += ent.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["n_updates"] += 1

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    early_stop = True
                    break

        n = max(1, stats.pop("n_updates"))
        return {k: v / n for k, v in stats.items()}
