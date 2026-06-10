"""阶段二 BC 训练：图像→GRU→动作 + 辅助 bbox/conf 监督

损失 = MSE(action, teacher_action) + aux_bbox_coef·MSE(bbox_pred, gt)
       + aux_conf_coef·BCE(conf_pred, gt_conf)

辅助头梯度反向传播到 CNN 骨干，稳定视觉表征。

用法:
  python -m guidance_rl.train.train_bc_v2 --data data/bc_v2 \\
      --out checkpoints/bc_policy_v2.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ..config import load_config
from ..models.policy_v2 import ActorCriticV2

CROP_SIZE = 288


class BCImageDataset(Dataset):
    """从 JPEG 目录 + metadata.npz 按需加载图像"""

    def __init__(self, data_dir: str):
        meta = np.load(os.path.join(data_dir, "metadata.npz"), allow_pickle=True)
        self.img_dir = os.path.join(data_dir, "images")
        self.num_frames = int(meta["num_frames"])
        self.ego = meta["ego"]
        self.action = meta["action"]
        self.bbox_label = meta["bbox_label"]
        self.conf_label = meta["conf_label"]
        self.ep_ends = meta["ep_ends"]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # 加载 JPEG 并转为 tensor
        img_path = os.path.join(self.img_dir, f"{idx:07d}.jpg")
        img = np.array(Image.open(img_path))  # [H, W, 3] uint8
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1)).copy()
        return {
            "image": img,
            "ego_state": self.ego[idx].copy(),
            "teacher_action": self.action[idx].copy(),
            "bbox_label": self.bbox_label[idx].copy(),
            "conf_label": self.conf_label[idx].copy(),
        }


def collate_batch(batch):
    """合并 batch → tensor，图像直接堆叠"""
    images = torch.from_numpy(np.stack([b["image"] for b in batch]))
    ego = torch.from_numpy(np.stack([b["ego_state"] for b in batch]))
    act = torch.from_numpy(np.stack([b["teacher_action"] for b in batch]))
    bbox = torch.from_numpy(np.stack([b["bbox_label"] for b in batch]))
    conf = torch.from_numpy(np.stack([b["conf_label"] for b in batch]))
    return images, ego, act, bbox, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="data/bc_v2")
    parser.add_argument("--out", default="checkpoints/bc_policy_v2.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    # 辅助损失系数
    parser.add_argument("--aux-bbox-coef", type=float, default=2.0)
    parser.add_argument("--aux-conf-coef", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)

    epochs = args.epochs or cfg.bc.epochs
    batch_size = args.batch_size or cfg.bc.batch_size
    lr = args.lr or cfg.bc.lr

    ds = BCImageDataset(args.data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_batch, num_workers=2,
                        pin_memory=(args.device == "cuda"))
    print(f"数据集: {ds.num_frames:,} 帧  batch_size={batch_size}  epochs={epochs}")

    model = ActorCriticV2(cfg.model, pretrained_cnn=True).to(args.device)
    actor = model.actor
    opt = torch.optim.Adam(actor.parameters(), lr=lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        actor.train()
        total_loss = total_act = total_bbox = total_conf = 0.0
        n_batches = 0

        for images, ego, act_gt, bbox_gt, conf_gt in loader:
            images = images.to(args.device)
            ego = ego.to(args.device)
            act_gt = act_gt.to(args.device)
            bbox_gt = bbox_gt.to(args.device)
            conf_gt = conf_gt.to(args.device)

            h0 = actor.initial_hidden(images.shape[0], args.device)
            act_pred, bbox_pred, conf_pred, _ = actor(images, ego, h0)

            loss_act = nn.functional.mse_loss(act_pred, act_gt)
            loss_bbox = nn.functional.mse_loss(bbox_pred, bbox_gt)
            loss_conf = nn.functional.binary_cross_entropy_with_logits(
                conf_pred, conf_gt)

            loss = loss_act + args.aux_bbox_coef * loss_bbox + args.aux_conf_coef * loss_conf

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_act += loss_act.item()
            total_bbox += loss_bbox.item()
            total_conf += loss_conf.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        tag = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "cfg_model": dict(cfg.model),
                        "bc_loss": avg_loss}, args.out)
            tag = "  <- saved"

        print(f"epoch {epoch+1:3d}/{epochs}  "
              f"total={avg_loss:.4f}  act={total_act/n_batches:.4f}  "
              f"bbox={total_bbox/n_batches:.4f}  conf={total_conf/n_batches:.4f}{tag}")

    print(f"\nBC V2 完成，最优 loss={best_loss:.4f} → {args.out}")


if __name__ == "__main__":
    main()
