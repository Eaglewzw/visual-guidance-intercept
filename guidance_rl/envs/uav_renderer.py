"""搜索区域 2D 精灵渲染器 —— 阶段二图像观测生成

在现有的轻量 Gym 运动学引擎上叠加视觉渲染，免去 Gazebo/Isaac Lab：
  1) 针孔投影：目标 NED 相对位置 → 全帧像素 (u, v)
  2) 裁剪：以 (u,v) 为中心取 288×288 搜索区域（模拟 LightTrack get_search_bbox）
  3) 渲染 UAV 精灵：在裁剪区内绘十字/菱形无人机形状，大小 ∝ focal·target_size/range
  4) 域随机化：背景(纯色/梯度/噪声/纹理)、UAV颜色/大小、光照/对比度、运动模糊、传感器噪声

输出: np.ndarray (288, 288, 3) uint8 RGB，裁剪 bbox 真值 (cx,cy,w,h 归一化到 [0,1])

设计原则：
  - 纯 numpy + PIL，无 GPU 渲染依赖，~0.1ms/帧
  - 域随机化在每集 reset() 时采样一次，集内保持不变（一致性），集间充分变化（泛化）
  - 与 LightTrack 搜索区域定义对齐（main.py get_search_bbox），
    部署时可直接用 uav_vision_dectect 的 crop 逻辑替换
"""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from ..geometry import project_to_pixel


class UAVRenderer:
    """搜索区域 2D 精灵渲染器。支持纹理模式（真实无人机图片）和几何模式。

    sprite_path: 无人机俯视纹理（PNG 透明背景最佳，JPG 也行）。
                 设为 None 则仅用几何图形。
    sprite_prob: 每集使用纹理模式的概率（0-1），剩余概率用几何图形。
                 设为 1.0 则 100% 纹理，0.5 则混合。
    """

    def __init__(self, cfg, rng: np.random.Generator,
                 sprite_path: str = None, sprite_prob: float = 0.8):
        self.cfg = cfg
        self.rng = rng
        self.crop_size = cfg.camera.get("crop_size", 288)
        self.half = self.crop_size // 2

        # 加载无人机纹理
        self.sprite = None
        self.sprite_prob = sprite_prob
        if sprite_path and os.path.exists(sprite_path):
            raw = Image.open(sprite_path)
            if raw.mode == "RGBA":
                self.sprite = raw
                self.sprite_has_alpha = True
            else:
                # JPG 白底 → 把接近白色的像素变透明
                self.sprite = self._remove_white_bg(raw.convert("RGBA"))
                self.sprite_has_alpha = True

    @staticmethod
    def _remove_white_bg(rgba: Image.Image, threshold: int = 230):
        """JPG 白底转透明：亮度 > threshold 的像素 alpha 设为 0"""
        arr = np.array(rgba)
        gray = arr[:, :, :3].mean(axis=2)
        mask = gray > threshold
        arr[mask, 3] = 0
        return Image.fromarray(arr, "RGBA")

    # ------------------------------------------------------------------
    def reset(self):
        """每集开始时调用：重新采样所有外观参数"""
        rng = self.rng
        c = self.cfg

        # -- 背景 --
        self.bg_type = rng.choice(["solid", "gradient", "noise", "texture"],
                                  p=[0.25, 0.25, 0.25, 0.25])
        self.bg_color = tuple(rng.integers(0, 256, 3).tolist())
        self.bg_color2 = tuple(rng.integers(0, 256, 3).tolist())
        self.bg_noise_std = rng.uniform(5, 40)

        # -- 无人机外观 --
        self.drone_color = tuple(rng.integers(30, 256, 3).tolist())
        # 纹理模式优先（如果有 sprite + 概率命中）
        if self.sprite is not None and rng.random() < self.sprite_prob:
            self.drone_shape = "sprite"
            # 纹理颜色抖动范围
            self.sprite_hue_shift = rng.uniform(-0.08, 0.08)
            self.sprite_saturation = rng.uniform(0.6, 1.4)
            self.sprite_brightness = rng.uniform(0.7, 1.3)
            self.sprite_contrast = rng.uniform(0.7, 1.3)
            self.sprite_rotation = rng.uniform(-35, 35)  # 度
        else:
            self.drone_shape = rng.choice(["cross", "diamond", "quad"],
                                          p=[0.4, 0.4, 0.2])
        self.drone_scale_jitter = rng.uniform(0.85, 1.15)

        # -- 传感器退化 --
        self.sensor_noise_std = rng.uniform(0.5, 6.0)
        self.motion_blur_kernel = rng.integers(0, 4)  # 0,1,2,3 px 一维核
        self.brightness_jitter = rng.uniform(0.7, 1.3)
        self.contrast_jitter = rng.uniform(0.7, 1.3)

        # -- 光照方向（梯度背景和无人机高光用）--
        self.light_angle = rng.uniform(0, 2 * math.pi)
        self.light_strength = rng.uniform(0.0, 0.3)

        # -- 预缓存背景 + 旋转精灵（加速 render() 热路径）--
        self._bg_cache = self._render_background_cv2()
        if self.drone_shape == "sprite" and self.sprite is not None:
            self._sprite_cache = self._prep_sprite_cv2()

    # ------------------------------------------------------------------
    def render(self, rel_ned: np.ndarray, roll: float, pitch: float, yaw: float,
               target_vel_ned: np.ndarray = None):
        """返回 (image [288,288,3] uint8, bbox_label [cx,cy,w,h] 归一化)"""
        if not hasattr(self, "bg_type"):
            self.reset()  # 首次调用自动重置
        # ---- 1) 针孔投影 ----
        res = project_to_pixel(rel_ned, roll, pitch, yaw,
                               self.cfg.camera.focal_length,
                               self.cfg.camera.image_width / 2.0,
                               self.cfg.camera.image_height / 2.0)
        if res is None:
            return (np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8),
                    np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))
        u_true, v_true, rng_m = res

        # ---- 2) 裁剪区域 ----
        cx_crop = int(round(u_true))
        cy_crop = int(round(v_true))
        x1 = cx_crop - self.half
        y1 = cy_crop - self.half

        # UAV 在裁剪区内的中心（亚像素）
        u_crop = u_true - x1
        v_crop = v_true - y1

        # ---- 3) bbox 尺寸 ----
        size_w = self.cfg.camera.focal_length * self.cfg.camera.target_size_w / max(rng_m, 0.5)
        size_h = self.cfg.camera.focal_length * self.cfg.camera.target_size_h / max(rng_m, 0.5)
        size_w *= self.drone_scale_jitter
        size_h *= self.drone_scale_jitter
        bbox_w = max(float(size_w), 2.0)
        bbox_h = max(float(size_h), 2.0)

        # ---- 4) 渲染 ----
        if self.drone_shape == "sprite" and hasattr(self, "_sprite_cache"):
            arr = self._composite_sprite_cv2(u_crop, v_crop, bbox_w, bbox_h)
            if self.motion_blur_kernel > 0 and target_vel_ned is not None:
                v_n = float(np.linalg.norm(target_vel_ned))
                if v_n > 0.1:
                    arr = cv2.blur(arr, (max(1, int(v_n * 1.5)), max(1, int(v_n * 1.5))))
            arr = arr.astype(np.float32)
        else:
            img = self._render_background()
            img = self._render_drone(img, u_crop, v_crop, bbox_w, bbox_h)
            if self.motion_blur_kernel > 0 and target_vel_ned is not None:
                v_norm = float(np.linalg.norm(target_vel_ned))
                if v_norm > 0.1:
                    img = img.filter(
                        ImageFilter.BoxBlur(radius=max(0.3, min(2.0, v_norm * 0.4))))
            arr = np.array(img).astype(np.float32)

        # ---- 5) 传感器噪声 + 亮度/对比度 ----
        arr += self.rng.normal(0, self.sensor_noise_std, arr.shape).astype(np.float32)
        arr = arr * self.contrast_jitter + (self.brightness_jitter - 1.0) * 128
        arr += (self.brightness_jitter - 1.0) * 128
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # ---- 7) bbox 标签（归一化到 [0,1]）----
        bbox_label = np.array([
            (u_crop - bbox_w / 2) / self.crop_size,   # cx (归一化)
            (v_crop - bbox_h / 2) / self.crop_size,   # cy (归一化)
            bbox_w / self.crop_size,                    # w (归一化)
            bbox_h / self.crop_size,                    # h (归一化)
        ], dtype=np.float32)

        return arr, bbox_label

    # ------------------------------------------------------------------
    #  内部：背景渲染
    # ------------------------------------------------------------------
    def _render_background(self):
        size = (self.crop_size, self.crop_size)
        if self.bg_type == "solid":
            return Image.new("RGB", size, self.bg_color)
        elif self.bg_type == "gradient":
            return self._gradient_bg(size)
        elif self.bg_type == "noise":
            arr = (self.rng.normal(128, self.bg_noise_std, (self.crop_size, self.crop_size, 3))
                   .clip(0, 255).astype(np.uint8))
            return Image.fromarray(arr)
        else:  # texture —— 几个随机色块模拟地面/天空纹理
            img = Image.new("RGB", size, self.bg_color)
            draw = ImageDraw.Draw(img)
            n_patches = self.rng.integers(3, 8)
            for _ in range(n_patches):
                c = tuple(self.rng.integers(0, 256, 3).tolist())
                x0 = int(self.rng.integers(0, self.crop_size))
                y0 = int(self.rng.integers(0, self.crop_size))
                r = int(self.rng.integers(20, 80))
                draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=c)
            return img

    def _gradient_bg(self, size):
        """方向渐变 + 少量噪声"""
        theta = self.light_angle
        gx = math.cos(theta)
        gy = math.sin(theta)
        xs = np.linspace(-1, 1, size[0])
        ys = np.linspace(-1, 1, size[1])
        xx, yy = np.meshgrid(xs, ys)
        grad = (xx * gx + yy * gy) * self.light_strength  # [-0.3, 0.3]
        grad = (grad + 0.5) * 255
        arr = np.stack([
            np.clip(grad * self.bg_color[0] / 128, 0, 255),
            np.clip(grad * self.bg_color[1] / 128, 0, 255),
            np.clip(grad * self.bg_color[2] / 128, 0, 255),
        ], axis=-1).astype(np.uint8)
        arr = (arr.astype(np.float32)
               + self.rng.normal(0, self.bg_noise_std * 0.3, arr.shape))
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    # ------------------------------------------------------------------
    #  内部：无人机精灵渲染
    # ------------------------------------------------------------------
    def _render_drone(self, img: Image.Image, cx: float, cy: float,
                      w: float, h: float) -> Image.Image:
        """在 img 上绘制 UAV 形状，支持亚像素中心。sprite 模式使用真实纹理"""
        if self.drone_shape == "sprite" and self.sprite is not None:
            return self._render_sprite(img, cx, cy, w, h)

        draw = ImageDraw.Draw(img)
        s = max(w, h)  # 特征尺度
        cx_i, cy_i = int(round(cx)), int(round(cy))

        # 光照调制：假设光照从左上方来，无人机左侧比右侧亮
        light_mod = 1.0 + self.light_strength
        color_light = tuple(min(255, int(c * light_mod)) for c in self.drone_color)
        color_dark = tuple(max(0, int(c * (1.0 - self.light_strength))) for c in self.drone_color)

        if self.drone_shape == "cross":
            # 十字形：机身 + 四个旋翼臂
            arm_len = int(s * 0.55)
            arm_w = max(1, int(s * 0.06))
            # 中心圆（机身）
            r_body = int(s * 0.08)
            draw.ellipse([cx_i - r_body, cy_i - r_body,
                          cx_i + r_body, cy_i + r_body], fill=self.drone_color)
            # 四个臂
            for dx, dy in [(arm_len, 0), (-arm_len, 0), (0, arm_len), (0, -arm_len)]:
                x0 = cx_i + dx - arm_w // 2
                y0 = cy_i + dy - arm_w // 2
                draw.rectangle([x0, y0, x0 + arm_w, y0 + arm_w], fill=self.drone_color)
            # 旋翼圆盘
            r_rotor = int(s * 0.12)
            for dx, dy in [(arm_len, 0), (-arm_len, 0), (0, arm_len), (0, -arm_len)]:
                draw.ellipse([cx_i + dx - r_rotor, cy_i + dy - r_rotor,
                              cx_i + dx + r_rotor, cy_i + dy + r_rotor],
                             outline=color_light, width=max(1, int(s * 0.03)))
        elif self.drone_shape == "diamond":
            # 菱形机身
            d = int(s * 0.4)
            pts = [(cx_i, cy_i - d), (cx_i + d, cy_i), (cx_i, cy_i + d), (cx_i - d, cy_i)]
            draw.polygon(pts, fill=self.drone_color, outline=color_dark)
            # 旋翼标记
            r_r = int(s * 0.1)
            for dx, dy in [(d, 0), (-d, 0)]:
                draw.ellipse([cx_i + dx - r_r, cy_i + dy - r_r,
                              cx_i + dx + r_r, cy_i + dy + r_r], fill=color_light)
        else:  # quad
            # 四旋翼俯视图：中心方框 + 四个旋翼
            body_half = int(s * 0.1)
            draw.rectangle([cx_i - body_half, cy_i - body_half,
                            cx_i + body_half, cy_i + body_half], fill=color_dark)
            r_r = int(s * 0.15)
            arm_r = int(s * 0.3)
            for dx, dy in [(arm_r, arm_r), (-arm_r, arm_r),
                           (arm_r, -arm_r), (-arm_r, -arm_r)]:
                draw.ellipse([cx_i + dx - r_r, cy_i + dy - r_r,
                              cx_i + dx + r_r, cy_i + dy + r_r],
                             fill=color_light, outline=self.drone_color,
                             width=max(1, int(s * 0.02)))

        return img

    # ------------------------------------------------------------------
    #  纹理精灵渲染（真实无人机图片）
    # ------------------------------------------------------------------
    def _render_sprite(self, img: Image.Image, cx: float, cy: float,
                       w: float, h: float) -> Image.Image:
        """将无人机纹理缩放/旋转/颜色抖动后贴到 img 上"""
        sprite = self.sprite

        # 1) 缩放
        size = int(max(w, h) * 1.3)
        size = max(size, 6)
        ratio = size / max(sprite.width, sprite.height)
        new_w = max(1, int(sprite.width * ratio))
        new_h = max(1, int(sprite.height * ratio))
        scaled = sprite.resize((new_w, new_h), Image.BILINEAR)

        # 2) 旋转
        if abs(self.sprite_rotation) > 0.5:
            scaled = scaled.rotate(self.sprite_rotation, resample=Image.BILINEAR,
                                   expand=True, fillcolor=(0, 0, 0, 0))

        # ---- 保存 alpha mask（HSV 转换会丢失 alpha）----
        if self.sprite_has_alpha and scaled.mode == "RGBA":
            alpha_mask = np.array(scaled)[:, :, 3].copy()
        else:
            alpha_mask = None

        # 3) 颜色抖动 (HSV) —— 在 RGB 副本上进行，保留原 RGBA
        rgb_copy = scaled.convert("RGB")
        need_color = (abs(self.sprite_hue_shift) > 0.001
                      or abs(self.sprite_saturation - 1.0) > 0.01)
        if need_color:
            hsv = rgb_copy.convert("HSV")
            arr = np.array(hsv, dtype=np.float32)
            arr[:, :, 0] = (arr[:, :, 0] / 255.0 + self.sprite_hue_shift) % 1.0
            arr[:, :, 0] *= 255.0
            arr[:, :, 1] = np.clip(arr[:, :, 1] * self.sprite_saturation, 0, 255)
            rgb_copy = Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGB")

        # 4) 亮度/对比度（在 RGB 上操作）
        rgb_copy = ImageEnhance.Brightness(rgb_copy).enhance(self.sprite_brightness)
        rgb_copy = ImageEnhance.Contrast(rgb_copy).enhance(self.sprite_contrast)

        # 5) 恢复 alpha 通道
        if alpha_mask is not None:
            arr_rgb = np.array(rgb_copy)
            rgba = np.dstack([arr_rgb, alpha_mask])
            scaled = Image.fromarray(rgba.astype(np.uint8), "RGBA")
        else:
            scaled = rgb_copy

        # 6) 贴到背景
        px = int(cx - scaled.width // 2)
        py = int(cy - scaled.height // 2)
        if alpha_mask is not None:
            img_rgba = img.convert("RGBA")
            img_rgba.paste(scaled, (px, py), scaled)
            return img_rgba.convert("RGB")
        else:
            img.paste(scaled.convert("RGB"), (px, py))
            return img

    # ==================================================================
    #  OpenCV 加速路径（纹理模式，比 PIL 快 10-20×）
    # ==================================================================
    def _render_background_cv2(self) -> np.ndarray:
        """生成背景 numpy 数组 [H, W, 3] uint8 (RGB)"""
        rng = self.rng
        H = W = self.crop_size
        if self.bg_type == "solid":
            return np.full((H, W, 3), self.bg_color, dtype=np.uint8)
        elif self.bg_type == "gradient":
            xs = np.linspace(-1, 1, W, dtype=np.float32)
            ys = np.linspace(-1, 1, H, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            grad = (xx * math.cos(self.light_angle)
                    + yy * math.sin(self.light_angle)) * self.light_strength
            grad = (grad + 0.5) * 255
            arr = np.stack([
                np.clip(grad * c / 128, 0, 255) for c in self.bg_color
            ], axis=-1).astype(np.uint8)
            noise = rng.normal(0, self.bg_noise_std * 0.3, arr.shape)
            return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:  # noise / texture
            return np.clip(
                rng.normal(128, self.bg_noise_std, (H, W, 3)), 0, 255
            ).astype(np.uint8)

    def _prep_sprite_cv2(self):
        """预计算旋转后的无人机精灵 (numpy RGBA)"""
        sprite_arr = np.array(self.sprite)  # RGBA
        # 转为 OpenCV 格式: RGBA → BGRA
        sprite_bgra = cv2.cvtColor(sprite_arr, cv2.COLOR_RGBA2BGRA)
        return sprite_bgra

    def _composite_sprite_cv2(self, u_crop: float, v_crop: float,
                               bbox_w: float, bbox_h: float) -> np.ndarray:
        """OpenCV 合成：背景上贴缩放+旋转+颜色抖动的无人机纹理"""
        rng = self.rng
        bg = self._bg_cache.copy()  # [H,W,3] uint8 RGB

        # 1) 缩放
        size = int(max(bbox_w, bbox_h) * 1.3)
        size = max(size, 6)
        sprite_h, sprite_w = self._sprite_cache.shape[:2]
        ratio = size / max(sprite_w, sprite_h)
        new_w = max(1, int(sprite_w * ratio))
        new_h = max(1, int(sprite_h * ratio))
        scaled = cv2.resize(self._sprite_cache, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)

        # 2) 旋转
        if abs(self.sprite_rotation) > 0.5:
            M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2),
                                        self.sprite_rotation, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            rw = int(new_h * sin + new_w * cos)
            rh = int(new_h * cos + new_w * sin)
            M[0, 2] += rw / 2 - new_w / 2
            M[1, 2] += rh / 2 - new_h / 2
            scaled = cv2.warpAffine(scaled, M, (rw, rh),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        # 3) 分离 alpha
        bgra = scaled
        alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        rgb = bgra[:, :, :3].astype(np.float32)  # BGR

        # 4) 颜色抖动 (HSV in OpenCV: 8-bit BGR→HSV)
        need_color = (abs(self.sprite_hue_shift) > 0.001
                      or abs(self.sprite_saturation - 1.0) > 0.01)
        if need_color:
            hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + self.sprite_hue_shift * 180) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.sprite_saturation, 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # 5) 亮度/对比度
        rgb = rgb * self.sprite_contrast + (self.sprite_brightness - 1.0) * 128
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # 6) Alpha 合成到背景 (BGR → RGB)
        sh, sw = rgb.shape[:2]
        px = int(u_crop - sw // 2)
        py = int(v_crop - sh // 2)
        # 计算有效区域
        x1, y1 = max(0, px), max(0, py)
        x2, y2 = min(bg.shape[1], px + sw), min(bg.shape[0], py + sh)
        sx1, sy1 = x1 - px, y1 - py
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        if x2 <= x1 or y2 <= y1:
            return bg
        alpha_roi = alpha[sy1:sy2, sx1:sx2, np.newaxis]
        # 合成: bg*(1-alpha) + rgb*alpha, BGR→RGB
        bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
        fg_roi = cv2.cvtColor(rgb[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB).astype(np.float32)
        blended = bg_roi * (1 - alpha_roi) + fg_roi * alpha_roi
        bg[y1:y2, x1:x2] = blended.astype(np.uint8)
        return bg
