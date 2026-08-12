"""策略网络与非对称 Critic

RecurrentActor : GRU(15→128) + MLP 头 → tanh 均值（4 维动作）
                 GRU 记忆承担时序信息提取（接近速度、目标机动模式），
                 替代 PNG 中的 LOS 差分。部署时只用 Actor（CPU 足够）。
PrivilegedCritic: 前馈 MLP，输入 = 观测 + 特权状态（相对位置/速度真值）。
                 特权信息使状态近似 Markov，无需循环结构。
                 仅训练用 —— 对应真实系统"统计专用"GPS 数据。
"""
import torch
import torch.nn as nn


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim=15, act_dim=4, gru_hidden=128, head_hidden=64,
                 init_log_std=-1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gru_hidden = gru_hidden

        self.gru = nn.GRU(obs_dim, gru_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

        # 末层小初始化：初始策略接近零动作（≈纯追踪），BC/RL 都从温和行为起步
        nn.init.uniform_(self.head[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.head[-1].bias)

    def initial_hidden(self, batch_size: int, device=None):
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def forward(self, obs_seq: torch.Tensor, h0: torch.Tensor):
        """obs_seq: [B,T,obs_dim], h0: [1,B,H] → (mean [B,T,act], hT)"""
        feat, hT = self.gru(obs_seq, h0)
        mean = torch.tanh(self.head(feat))
        return mean, hT

    def forward_masked(self, obs_seq, h0, done_seq):
        """带 episode 边界的序列前向：done 处重置隐藏状态（PPO/BC 训练用）

        obs_seq: [B,T,obs], done_seq: [B,T]（该步之后是否换了新集）
        逐步展开，速度可接受（B*T ~ 8k steps/update）。
        """
        B, T, _ = obs_seq.shape
        h = h0
        means = []
        for t in range(T):
            feat, h = self.gru(obs_seq[:, t:t + 1, :], h)
            means.append(torch.tanh(self.head(feat)))
            # done 后清零隐藏状态（下一步属于新 episode）
            mask = (1.0 - done_seq[:, t]).view(1, B, 1)
            h = h * mask
        return torch.cat(means, dim=1), h


class PrivilegedCritic(nn.Module):
    def __init__(self, obs_dim=15, priv_dim=9, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + priv_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, priv):
        return self.net(torch.cat([obs, priv], dim=-1)).squeeze(-1)


class ActorCritic(nn.Module):
    """训练容器；rollout 时逐步调用 act()，更新时调用 evaluate()"""

    def __init__(self, cfg_model):
        super().__init__()
        self.actor = RecurrentActor(
            obs_dim=cfg_model.obs_dim, act_dim=cfg_model.act_dim,
            gru_hidden=cfg_model.gru_hidden, head_hidden=cfg_model.head_hidden,
            init_log_std=cfg_model.init_log_std)
        self.critic = PrivilegedCritic(
            obs_dim=cfg_model.obs_dim, priv_dim=cfg_model.priv_dim,
            hidden=cfg_model.critic_hidden)

    @torch.no_grad()
    def act(self, obs, priv, h, deterministic=False,
            log_std_min=-2.5, log_std_max=0.0):
        """obs/priv: [B,dim], h: [1,B,H] → (action, logprob, value, h')"""
        mean, h_new = self.actor(obs.unsqueeze(1), h)
        mean = mean.squeeze(1)
        log_std = self.actor.log_std.clamp(log_std_min, log_std_max)
        std = log_std.exp().expand_as(mean)
        if deterministic:
            action = mean
            logprob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(-1)
        value = self.critic(obs, priv)
        return action, logprob, value, h_new

    def evaluate(self, obs_seq, priv_seq, act_seq, done_seq, h0,
                 log_std_min=-2.5, log_std_max=0.0):
        """序列重放：返回 (logprob, entropy, value) 各 [B,T]"""
        mean, _ = self.actor.forward_masked(obs_seq, h0, done_seq)
        log_std = self.actor.log_std.clamp(log_std_min, log_std_max)
        std = log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        logprob = dist.log_prob(act_seq).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs_seq, priv_seq)
        return logprob, entropy, value
