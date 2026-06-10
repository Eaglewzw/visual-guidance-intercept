"""阶段二 BC 数据采集：PNG 老师闭环 → (图像, ego_state, 动作, bbox标签)

输出格式:
  data/bc_v2/
    images/           JPEG 文件（每帧一个，文件名 = 全局步序号 .jpg）
    metadata.npz      {ego, action, bbox_label, conf_label, ep_ends, modes, outcomes}

训练时用 ImageFolderDataset 按需加载图像（不会一次性加载 200K 张图到内存）。

用法:
  python -m guidance_rl.train.collect_bc_data_v2 --episodes 500 --out data/bc_v2
"""
import argparse
import os

import numpy as np
from PIL import Image

from ..config import load_config
from ..envs import InterceptEnvV2


def collect(cfg, episodes: int, out_dir: str, seed: int = 0, mode: str = "mixed"):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    env = InterceptEnvV2(cfg, mode=mode, seed=seed)
    all_ego, all_act, all_bbox, all_conf, ep_ends = [], [], [], [], []
    outcomes, modes = [], []
    global_idx = 0

    for ep in range(episodes):
        obs, info = env.reset()
        a = info["teacher_action"]
        # 存首帧
        Image.fromarray(obs["image"].transpose(1, 2, 0)).save(
            os.path.join(img_dir, f"{global_idx:07d}.jpg"), quality=85)
        all_ego.append(obs["ego_state"])
        all_act.append(a)
        all_bbox.append(info["bbox_label"])
        all_conf.append(info["conf_label"])
        global_idx += 1

        while True:
            obs, r, term, trunc, info = env.step(a)
            a = info["teacher_action"]
            if term or trunc:
                outcomes.append(info["outcome"])
                modes.append(info["mode"])
                ep_ends.append(global_idx)
                break
            Image.fromarray(obs["image"].transpose(1, 2, 0)).save(
                os.path.join(img_dir, f"{global_idx:07d}.jpg"), quality=85)
            all_ego.append(obs["ego_state"])
            all_act.append(a)
            all_bbox.append(info["bbox_label"])
            all_conf.append(info["conf_label"])
            global_idx += 1

        if (ep + 1) % 100 == 0:
            hits = sum(1 for o in outcomes if o == "hit")
            print(f"[{ep+1}/{episodes}] teacher hit rate {hits/(ep+1):.1%}, "
                  f"total frames {global_idx:,}")

    np.savez_compressed(
        os.path.join(out_dir, "metadata.npz"),
        ego=np.array(all_ego, dtype=np.float32),
        action=np.array(all_act, dtype=np.float32),
        bbox_label=np.array(all_bbox, dtype=np.float32),
        conf_label=np.array(all_conf, dtype=np.float32),
        ep_ends=np.array(ep_ends, dtype=np.int64),
        outcomes=np.array(outcomes),
        modes=np.array(modes),
        num_frames=global_idx,
    )
    hits = sum(1 for o in outcomes if o == "hit")
    print(f"\n保存到 {out_dir}/  frames={global_idx:,}  teacher hit={hits/episodes:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/bc_v2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    episodes = args.episodes or cfg.bc.episodes
    collect(cfg, episodes, args.out, seed=args.seed, mode=args.mode)


if __name__ == "__main__":
    main()
