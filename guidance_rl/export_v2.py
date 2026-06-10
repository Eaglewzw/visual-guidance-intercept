"""导出阶段二部署模型：CNN+GRU → TorchScript

输出:
  policy_v2.pt       : TorchScript Actor (MobileNetV3+GRU，裁剪辅助头)
  policy_v2_meta.json : 动作解码常数 + 特征版本 + 图像预处理常数

部署时：
  - policy_runtime_v2.py 负责 288×288 裁剪、ImageNet 归一化、GRU 隐状态管理
  - 搜索区域裁剪复用 LightTrack get_search_bbox 逻辑

用法:
  python -m guidance_rl.export_v2 --ckpt checkpoints/rl_policy_v2.pt \\
      --out /home/verser/ros2_ws/src/uav_rl_guidance/models/policy_v2.pt
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from .config import load_config
from .features import FEATURE_VERSION
from .models.policy_v2 import ActorV2

ACT_DIM = 4
GRU_HIDDEN = 128
CROP_SIZE = 288


class PolicyTSV2(nn.Module):
    """部署封装：图像归一化 + CNN + GRU 单步推理"""

    def __init__(self, actor: ActorV2):
        super().__init__()
        self.encoder = actor.encoder
        self.visual_proj = actor.visual_proj
        self.ego_proj = actor.ego_proj
        self.gru = actor.gru
        self.head = actor.head
        # 注册归一化常数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.gru_hidden = actor.gru_hidden

    def forward(self, image: torch.Tensor, ego_state: torch.Tensor,
                h: torch.Tensor):
        # image: [1, 3, 288, 288] uint8 → float32 → normalize
        x = image.float() / 255.0
        x = (x - self.mean) / self.std
        feat = self.encoder(x)                  # [1, 576]
        visual = self.visual_proj(feat)          # [1, 128]
        ego = self.ego_proj(ego_state)            # [1, 32]
        combined = torch.cat([visual, ego], dim=-1).unsqueeze(1)  # [1, 1, 160]
        gru_out, h_new = self.gru(combined, h)   # [1, 1, H]
        action = torch.tanh(self.head(gru_out)).squeeze(1)  # [1, 4]
        return action, h_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="export/policy_v2.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ActorCriticV2(cfg.model, pretrained_cnn=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    wrapper = PolicyTSV2(model.actor)
    scripted = torch.jit.script(wrapper)

    # 冒烟
    img = torch.randint(0, 256, (1, 3, CROP_SIZE, CROP_SIZE), dtype=torch.uint8)
    ego = torch.zeros(1, 8)
    h = torch.zeros(1, 1, GRU_HIDDEN)
    act, h2 = scripted(img, ego, h)
    assert act.shape == (1, ACT_DIM) and h2.shape == h.shape

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scripted.save(args.out)

    meta = {
        "phase": 2,
        "feature_version": FEATURE_VERSION,
        "act_dim": ACT_DIM,
        "gru_hidden": GRU_HIDDEN,
        "crop_size": CROP_SIZE,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "action_decode": {
            "dv_angle_max": cfg.action.dv_angle_max,
            "speed_min": cfg.png.speed_min,
            "speed_cmd": cfg.png.speed_cmd,
            "yaw_rate_max": cfg.png.yaw_rate_max,
            "elev_clamp": cfg.png.elev_clamp,
        },
        "source_ckpt": os.path.abspath(args.ckpt),
        "ckpt_hit_rate": ckpt.get("hit_rate"),
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"TorchScript V2 已导出: {args.out}")
    print(f"元数据:               {meta_path}")


# 为 export 导入作兼容
from .models.policy_v2 import ActorCriticV2  # noqa: E402 (after class def)

if __name__ == "__main__":
    main()
