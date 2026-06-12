"""阶段二策略网络：MobileNetV3-Small + GRU + 辅助 bbox 头

架构:
  搜索区域 (288×288×3) ──→ MobileNetV3-Small ──→ 576维特征
       │                                              │
       │                           ┌───────────────────┤
       │                           │                   │
       │                    Linear(576→128)    AuxHead(576→5)
       │                    visual_feat(128)   [bbox_cx,cy,w,h, conf]
       │                           │
       └── concat ────────────────┤
          ego_state (8维)          │
       [vx,vy,vz,|V|,roll,pitch,yaw,local_z]
                                   │
                              GRU(136→128)
                                   │
                              ActionHead → 4维 tanh

辅助头监督（训练期）:
  - bbox: MSE(预测, 真值归一化坐标)，权值 aux_bbox_coef=1.0
  - conf: BCE(预测, 1=目标在裁剪区内)，权值 aux_conf_coef=0.5
  梯度从辅助头反向传播到 CNN 骨干，稳定视觉表征学习。

训练期: forward() 返回 (action_seq, bbox_pred, conf_pred, hT)
推理期: forward() 只取 action 部分，部署时剪枝辅助头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from .policy import RecurrentActor  # 阶段一的 GRU 头（复用其结构模式）

CROP_SIZE = 288


# ============================================================
#  CNN 编码器
# ============================================================
class MobileNetEncoder(nn.Module):
    """MobileNetV3-Small → GAP → 576 维特征"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # 输出 [B, 576, H/32, W/32]
        self.avgpool = backbone.avgpool    # → [B, 576, 1, 1]

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 288, 288] 归一化到 ImageNet 统计量
        feat = self.features(x)         # [B, 576, 9, 9]
        feat = self.avgpool(feat)       # [B, 576, 1, 1]
        return feat.view(feat.size(0), -1)  # [B, 576]


# ============================================================
#  辅助头
# ============================================================
class AuxHead(nn.Module):
    """bbox 回归 + 置信度预测（可选）"""

    def __init__(self, in_dim: int = 576, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.bbox = nn.Linear(hidden, 4)   # cx, cy, w, h ∈ [0,1]
        self.conf = nn.Linear(hidden, 1)   # logit → sigmoid 即置信度

    def forward(self, feat):
        h = self.net(feat)
        box = torch.sigmoid(self.bbox(h))         # sigmoid 约束 [0,1]
        conf = self.conf(h)                       # 原始 logit（BCEWithLogits 用）
        return box, conf


# ============================================================
#  阶段二 Actor（完整网络）
# ============================================================
class ActorV2(nn.Module):
    """CNN 编码器 + 自身状态编码 + GRU + 动作头 + 辅助头

    输入:
      images    [B, 3, 288, 288]  uint8 [0-255] 搜索区域裁剪
      ego_state [B, 8]           自身状态：[vx,vy,vz,|V|,roll,pitch,yaw,local_z]
      h0        [1, B, gru_hidden]

    输出:
      action    [B, 4]           tanh 动作（同阶段一）
      bbox_pred [B, 4]           sigmoid bbox 预测（辅助头，训练用）
      conf_pred [B, 1]           置信度 logit（辅助头，训练用）
      hT        [1, B, gru_hidden]
    """

    def __init__(self, obs_dim=15, act_dim=4, gru_hidden=128, head_hidden=64,
                 init_log_std=-1.0, pretrained_cnn=True):
        super().__init__()
        self.act_dim = act_dim
        self.gru_hidden = gru_hidden
        self.ego_dim = 8

        # CNN
        self.encoder = MobileNetEncoder(pretrained=pretrained_cnn)
        self.visual_proj = nn.Linear(576, 128)

        # 辅助头（训练期有监督，部署时剪枝）
        self.aux_head = AuxHead(in_dim=576, hidden=128)

        # 自身状态编码
        self.ego_proj = nn.Sequential(
            nn.Linear(self.ego_dim, 32),
            nn.ReLU(inplace=True),
        )

        # GRU：输入 = visual_128 + ego_32
        gru_input = 128 + 32
        self.gru = nn.GRU(gru_input, gru_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

        # 末层小初始化
        nn.init.uniform_(self.head[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.head[-1].bias)

    # ---- forward ----
    def forward(self, images: torch.Tensor, ego_state: torch.Tensor,
                h0: torch.Tensor, return_aux: bool = True):
        """单步或多步前向（batch_first=True）。

        images:   [B, T, 3, 288, 288] 或 [B, 3, 288, 288]
        ego_state: [B, T, 8] 或 [B, 8]
        h0:        [1, B, H]
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)
            ego_flat = ego_state.view(B * T, -1)
        else:
            B = images.shape[0]
            T = 1
            images_flat = images
            ego_flat = ego_state

        # 标准化 uint8 → ImageNet norm
        images_norm = (images_flat.float() / 255.0
                       - torch.tensor([0.485, 0.456, 0.406],
                                      device=images.device).view(1, 3, 1, 1)) \
                      / torch.tensor([0.229, 0.224, 0.225],
                                     device=images.device).view(1, 3, 1, 1)

        feat_576 = self.encoder(images_norm)           # [B*T, 576]
        visual = self.visual_proj(feat_576)             # [B*T, 128]
        ego_code = self.ego_proj(ego_flat)              # [B*T, 32]
        combined = torch.cat([visual, ego_code], dim=-1)  # [B*T, 136]

        combined_seq = combined.view(B, T, -1)
        gru_out, hT = self.gru(combined_seq, h0)        # [B, T, H], [1, B, H]
        action = torch.tanh(self.head(gru_out))          # [B, T, 4]

        if not return_aux:
            return action.squeeze(1) if T == 1 else action, hT

        bbox, conf = self.aux_head(feat_576)
        bbox = bbox.view(B, T, 4) if T > 1 else bbox
        conf = conf.view(B, T, 1) if T > 1 else conf

        if T == 1:
            return action.squeeze(1), bbox.squeeze(0), conf.squeeze(0), hT
        return action, bbox, conf, hT

    # ---- 初始化 ----
    def initial_hidden(self, batch_size: int, device=None):
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    # ---- 批量编码 + GRU 逐步展开（PPO 更新用，关键优化）----
    def forward_masked(self, images, ego_state, done_seq, h0):
        """将 CNN 编码一次跑完所有 timestep，然后 GRU 逐步展开。

        images:    [B, T, 3, 288, 288]
        ego_state: [B, T, 8]
        done_seq:  [B, T]
        h0:        [1, B, H]
        返回 mean [B, T, 4]
        """
        B, T = images.shape[0], images.shape[1]

        # 一次性 CNN 编码所有帧：B×T → 一次 MobileNetV3 前向
        imgs_flat = images.reshape(B * T, 3, CROP_SIZE, CROP_SIZE)
        ego_flat = ego_state.reshape(B * T, -1)
        images_norm = (imgs_flat.float() / 255.0
                       - torch.tensor([0.485, 0.456, 0.406],
                                      device=images.device).view(1, 3, 1, 1)) \
                      / torch.tensor([0.229, 0.224, 0.225],
                                     device=images.device).view(1, 3, 1, 1)
        feat_576 = self.encoder(images_norm)          # [B*T, 576]
        visual = self.visual_proj(feat_576)            # [B*T, 128]
        ego_code = self.ego_proj(ego_flat)              # [B*T, 32]
        combined = torch.cat([visual, ego_code], dim=-1).view(B, T, -1)  # [B, T, 136]

        # GRU 逐步展开（轻量，无 CNN）
        h = h0
        means = []
        for t in range(T):
            out, h = self.gru(combined[:, t:t + 1, :], h)
            means.append(torch.tanh(self.head(out)))
            mask = (1.0 - done_seq[:, t]).view(1, B, 1)
            h = h * mask
        return torch.cat(means, dim=1), h

    # ---- 推理（部署 + rollout）----
    @torch.no_grad()
    def act(self, image, ego_state, h, deterministic=False,
            log_std_min=-2.5, log_std_max=0.0):
        """image: [B,3,288,288], ego_state: [B,8], h: [1,B,H]"""
        action, _, _, h_new = self.forward(image, ego_state, h, return_aux=True)
        if deterministic:
            return action, h_new
        log_std = self.log_std.clamp(log_std_min, log_std_max)
        std = log_std.exp().unsqueeze(0).expand_as(action)
        dist = torch.distributions.Normal(action, std)
        sampled = dist.sample()
        logprob = dist.log_prob(sampled).sum(-1)
        return sampled, logprob, h_new


# ============================================================
#  训练容器（非对称 Critic 复用阶段一：输入=几何特征+真值）
# ============================================================
class ActorCriticV2(nn.Module):
    """Actor: CNN+GRU 看图像+ego；Critic: 前馈 MLP 看阶段一几何特征+特权真值

    设计理由：Critic 只看 gt_obs（从仿真真值构造的阶段一 15 维特征
    + 9 维特权信息），不需要图像——仿真中目标位置已知时这些可直接算。
    训练结束 Critic 丢弃，部署时纯视觉原则不变。
    """

    def __init__(self, cfg_model, pretrained_cnn=True):
        super().__init__()
        self.actor = ActorV2(
            obs_dim=cfg_model.obs_dim, act_dim=cfg_model.act_dim,
            gru_hidden=cfg_model.gru_hidden, head_hidden=cfg_model.head_hidden,
            init_log_std=cfg_model.init_log_std, pretrained_cnn=pretrained_cnn)
        from .policy import PrivilegedCritic
        self.critic = PrivilegedCritic(
            obs_dim=cfg_model.obs_dim, priv_dim=cfg_model.priv_dim,
            hidden=cfg_model.critic_hidden)

    @torch.no_grad()
    def act(self, image, ego_state, gt_obs, priv, h, deterministic=False):
        """gt_obs: 阶段一 15 维几何观测（从仿真真值构造，仅供 Critic）"""
        action, logprob, h_new = self.actor.act(image, ego_state, h, deterministic)
        value = self.critic(gt_obs, priv)
        return action, logprob, value, h_new
