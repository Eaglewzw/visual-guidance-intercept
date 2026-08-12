"""相机与检测模型 —— 模拟 uav_vision_dectect（YOLO+LightTrack）的 bbox 输出

针孔投影（focal=1397.2, 1920x1080，与 SDF 一致）+ 检测退化模型：
  - bbox 尺寸 w = focal * 目标物理尺寸 / 距离（每集尺寸 ±20% 随机）
  - 中心噪声 σ = max(1px, 0.06*w)，尺寸噪声 10%
  - 漏检：基础 2% + 小目标附加 0.25*exp(-w/6px)
  - 延迟：每集随机 0~2 帧（检测耗时 + 话题传输）
  - 出 FOV / 在相机后方 → 丢失（width=-1 的语义）

输出与 /camera_detect_result (RectMsg) 相同：(x, y, w, h) 左上角+宽高 或 None。
"""
import math
from collections import deque

import numpy as np

from ..geometry import project_to_pixel


class CameraModel:
    def __init__(self, cfg, rng: np.random.Generator):
        """cfg: DotDict 的 camera 节"""
        self.cfg = cfg
        self.rng = rng
        self.cx = cfg.image_width / 2.0
        self.cy = cfg.image_height / 2.0
        self.reset()

    def reset(self):
        cfg, rng = self.cfg, self.rng
        # 每集随机：目标可视尺寸、延迟帧数
        j = cfg.target_size_jitter
        self.size_w = cfg.target_size_w * (1.0 + rng.uniform(-j, j))
        self.size_h = cfg.target_size_h * (1.0 + rng.uniform(-j, j))
        self.latency = int(rng.integers(0, cfg.latency_frames_max + 1))
        self.queue = deque(maxlen=self.latency + 1)

    def observe(self, rel_ned: np.ndarray, roll: float, pitch: float, yaw: float):
        """rel_ned: 目标相对拦截机的 NED 向量。返回 (x,y,w,h) 或 None"""
        raw = self._detect_raw(rel_ned, roll, pitch, yaw)
        # 延迟队列：本帧入队，取 latency 帧前的结果
        self.queue.append(raw)
        if len(self.queue) <= self.latency:
            return None
        return self.queue[0]

    # ------------------------------------------------------------------
    def _detect_raw(self, rel_ned, roll, pitch, yaw):
        cfg, rng = self.cfg, self.rng

        res = project_to_pixel(rel_ned, roll, pitch, yaw,
                               cfg.focal_length, self.cx, self.cy)
        if res is None:
            return None
        u, v, rng_m = res

        # 理想 bbox 尺寸
        w = max(cfg.focal_length * self.size_w / rng_m, cfg.min_bbox_px)
        h = max(cfg.focal_length * self.size_h / rng_m, cfg.min_bbox_px)

        # FOV 判定（允许 bbox 中心略出边缘即丢失，与检测器行为一致）
        if not (0 <= u < cfg.image_width and 0 <= v < cfg.image_height):
            return None

        # 漏检概率（小目标更易丢）
        p_miss = cfg.miss_base + cfg.miss_small_scale * math.exp(-w / cfg.miss_small_px)
        if rng.random() < p_miss:
            return None

        # 噪声
        sigma_c = max(1.0, cfg.pixel_noise_frac * w)
        u += rng.normal(0, sigma_c)
        v += rng.normal(0, sigma_c)
        w *= max(0.3, 1.0 + rng.normal(0, cfg.size_noise_frac))
        h *= max(0.3, 1.0 + rng.normal(0, cfg.size_noise_frac))

        # RectMsg 语义：左上角 + 宽高，整数像素
        x = int(round(u - w / 2.0))
        y = int(round(v - h / 2.0))
        return (x, y, max(int(round(w)), 1), max(int(round(h)), 1))
