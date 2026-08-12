"""导出部署模型：checkpoint → TorchScript + 元数据

输出两个文件：
  policy.pt        : TorchScript Actor，接口 forward(obs[1,15], h[1,1,128]) → (act, h')
  policy_meta.json : 动作解码常数 + 特征版本（policy_runtime 加载时校验）

用法:
  python -m aerointercept.export --ckpt checkpoints/rl_policy.pt \\
      --out /home/verser/ros2_ws/src/uav_rl_guidance/models/policy.pt
"""
import argparse
import json
import os

import torch
import torch.nn as nn

from .config import load_config
from .features import FEATURE_VERSION, OBS_DIM, ACT_DIM
from .models.policy import ActorCritic


class PolicyTS(nn.Module):
    """部署封装：单步推理，确定性（tanh 均值）"""

    def __init__(self, actor):
        super().__init__()
        self.gru = actor.gru
        self.head = actor.head

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        # obs: [1, obs_dim], h: [1, 1, hidden]
        feat, h_new = self.gru(obs.unsqueeze(1), h)
        action = torch.tanh(self.head(feat)).squeeze(1)
        return action, h_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="export/policy.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ActorCritic(cfg.model)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    wrapper = PolicyTS(model.actor)
    scripted = torch.jit.script(wrapper)

    # 冒烟验证
    obs = torch.zeros(1, OBS_DIM)
    h = torch.zeros(1, 1, cfg.model.gru_hidden)
    act, h2 = scripted(obs, h)
    assert act.shape == (1, ACT_DIM) and h2.shape == h.shape

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scripted.save(args.out)

    meta = {
        "feature_version": FEATURE_VERSION,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "gru_hidden": cfg.model.gru_hidden,
        "action_decode": {
            "dv_angle_max": cfg.action.dv_angle_max,
            "speed_min": cfg.png.speed_min,
            "speed_cmd": cfg.png.speed_cmd,
            "yaw_rate_max": cfg.png.yaw_rate_max,
            "elev_clamp": cfg.png.elev_clamp,
        },
        "camera": {
            "focal_length": cfg.camera.focal_length,
            "image_width": cfg.camera.image_width,
            "image_height": cfg.camera.image_height,
        },
        "source_ckpt": os.path.abspath(args.ckpt),
        "ckpt_hit_rate": ckpt.get("hit_rate"),
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"TorchScript 已导出: {args.out}")
    print(f"元数据已导出:      {meta_path}")


if __name__ == "__main__":
    main()
