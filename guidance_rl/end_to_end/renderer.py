"""Fast domain-randomized full-frame renderer.

This module never crops around a known target. It renders the complete camera
field of view at the policy resolution;
target localization is therefore part of the actor's job.  Simulator labels are
returned separately and are never included in the observation.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from ..geometry import project_to_pixel


@dataclass(frozen=True)
class RenderResult:
    """Rendered RGB frame plus training-only geometric labels."""

    image: np.ndarray
    visible: bool
    center_normalized: np.ndarray
    bbox_normalized: np.ndarray


class FullFrameRenderer:
    """Procedural full-camera renderer with episode-level randomization."""

    def __init__(self, camera_cfg, render_cfg, rng: np.random.Generator):
        self.camera_cfg = camera_cfg
        self.cfg = render_cfg
        self.rng = rng
        self.width = int(render_cfg.image_width)
        self.height = int(render_cfg.image_height)
        self._sprite = self._load_sprite(render_cfg.get("sprite_path", ""))
        self._initialized = False

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def reset(self) -> None:
        """Resample appearance while keeping it coherent within an episode."""
        rng = self.rng
        self.focal_scale = rng.uniform(
            1.0 - self.cfg.focal_jitter, 1.0 + self.cfg.focal_jitter)
        self.target_scale = rng.uniform(
            1.0 - self.cfg.target_scale_jitter,
            1.0 + self.cfg.target_scale_jitter,
        )
        self.brightness = rng.uniform(*self.cfg.brightness_range)
        self.contrast = rng.uniform(*self.cfg.contrast_range)
        self.noise_std = rng.uniform(*self.cfg.noise_std_range)
        self.blur_kernel = int(rng.choice(self.cfg.blur_kernels))
        self.use_sprite = (
            self._sprite is not None
            and rng.random() < self.cfg.sprite_probability
        )
        self.target_color = tuple(int(v) for v in rng.integers(20, 236, 3))
        self.target_rotation_offset = rng.uniform(-25.0, 25.0)
        self._panorama = self._make_panorama()
        self._initialized = True

    def render(self, rel_ned: np.ndarray, roll: float, pitch: float,
               yaw: float, target_velocity_ned=None) -> RenderResult:
        """Render one full RGB frame and return simulator-only labels."""
        if not self._initialized:
            self.reset()
        image = self._background_view(roll, pitch, yaw)
        projection = project_to_pixel(
            rel_ned, roll, pitch, yaw,
            self.camera_cfg.focal_length * self.focal_scale,
            self.camera_cfg.image_width / 2.0,
            self.camera_cfg.image_height / 2.0,
        )

        if projection is None:
            return self._finish(image, False)

        u_full, v_full, distance = projection
        scale_x = self.width / float(self.camera_cfg.image_width)
        scale_y = self.height / float(self.camera_cfg.image_height)
        center_x = u_full * scale_x
        center_y = v_full * scale_y
        target_w = (
            self.camera_cfg.focal_length * self.focal_scale
            * self.camera_cfg.target_size_w / max(distance, 0.5)
            * scale_x * self.target_scale
        )
        target_h = (
            self.camera_cfg.focal_length * self.focal_scale
            * self.camera_cfg.target_size_h / max(distance, 0.5)
            * scale_y * self.target_scale
        )
        target_w = max(float(self.cfg.min_target_pixels), float(target_w))
        target_h = max(float(self.cfg.min_target_pixels), float(target_h))

        visible = (
            center_x + target_w / 2.0 >= 0.0
            and center_x - target_w / 2.0 < self.width
            and center_y + target_h / 2.0 >= 0.0
            and center_y - target_h / 2.0 < self.height
        )
        center = np.array([
            2.0 * center_x / self.width - 1.0,
            2.0 * center_y / self.height - 1.0,
        ], dtype=np.float32)
        bbox = np.array([
            center_x / self.width,
            center_y / self.height,
            target_w / self.width,
            target_h / self.height,
        ], dtype=np.float32)

        if visible:
            rotation = self.target_rotation_offset
            if target_velocity_ned is not None:
                velocity = np.asarray(target_velocity_ned)
                if np.linalg.norm(velocity[:2]) > 0.05:
                    heading = math.atan2(velocity[1], velocity[0])
                    rotation += math.degrees(heading - yaw)
            if self.use_sprite:
                self._draw_sprite(
                    image, center_x, center_y, target_w, target_h, rotation)
            else:
                self._draw_geometric_target(
                    image, center_x, center_y, target_w, target_h, rotation)

        return self._finish(image, visible, center, bbox)

    def _finish(self, image, visible, center=None, bbox=None) -> RenderResult:
        image = image.astype(np.float32)
        image = (image - 127.5) * self.contrast + 127.5
        image *= self.brightness
        if self.noise_std > 0.0:
            image += self.rng.normal(0.0, self.noise_std, image.shape)
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        if self.blur_kernel > 1:
            kernel = self.blur_kernel if self.blur_kernel % 2 else self.blur_kernel + 1
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
        if center is None:
            center = np.zeros(2, dtype=np.float32)
        if bbox is None:
            bbox = np.zeros(4, dtype=np.float32)
        return RenderResult(
            image=np.ascontiguousarray(image),
            visible=bool(visible),
            center_normalized=center,
            bbox_normalized=bbox,
        )

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------
    def _make_panorama(self) -> np.ndarray:
        """Create a wide cyclic scene so yaw produces coherent optic motion."""
        height = self.height * 3
        width = self.width * 4
        top = self.rng.integers(70, 220, 3).astype(np.float32)
        bottom = self.rng.integers(20, 190, 3).astype(np.float32)
        blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        panorama = top[None, None, :] * (1.0 - blend) + bottom[None, None, :] * blend
        panorama = np.repeat(panorama, width, axis=1)

        low_frequency = self.rng.normal(
            0.0, 18.0, (max(2, height // 16), max(2, width // 16), 3))
        low_frequency = cv2.resize(
            low_frequency.astype(np.float32), (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        panorama += low_frequency
        panorama = np.clip(panorama, 0.0, 255.0).astype(np.uint8)

        for _ in range(int(self.rng.integers(12, 30))):
            color = tuple(int(v) for v in self.rng.integers(0, 256, 3))
            center = (
                int(self.rng.integers(0, width)),
                int(self.rng.integers(height // 3, height)),
            )
            axes = (
                int(self.rng.integers(5, max(6, width // 12))),
                int(self.rng.integers(3, max(4, height // 10))),
            )
            cv2.ellipse(panorama, center, axes, 0, 0, 360, color, -1)
        return panorama

    def _background_view(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        panorama = self._panorama
        panorama_h, panorama_w = panorama.shape[:2]
        x0 = int((yaw % (2.0 * math.pi)) / (2.0 * math.pi) * panorama_w)
        focal_policy = (
            self.camera_cfg.focal_length * self.focal_scale
            * self.width / float(self.camera_cfg.image_width)
        )
        y_center = panorama_h // 2 + int(math.tan(pitch) * focal_policy)
        y0 = int(np.clip(y_center - self.height // 2, 0, panorama_h - self.height))
        x_indices = (np.arange(self.width) + x0) % panorama_w
        view = panorama[y0:y0 + self.height][:, x_indices].copy()
        if abs(roll) > 1e-3:
            transform = cv2.getRotationMatrix2D(
                (self.width / 2.0, self.height / 2.0),
                -math.degrees(roll), 1.0)
            view = cv2.warpAffine(
                view, transform, (self.width, self.height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return view

    # ------------------------------------------------------------------
    # Target drawing
    # ------------------------------------------------------------------
    def _load_sprite(self, configured_path: str):
        if not configured_path:
            return None
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            return None
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            near_white = np.all(image > 238, axis=2).astype(np.uint8)
            count, labels = cv2.connectedComponents(near_white)
            background = np.zeros_like(near_white, dtype=bool)
            border_labels = np.unique(np.concatenate([
                labels[0], labels[-1], labels[:, 0], labels[:, -1],
            ]))
            for label in border_labels:
                if 0 < label < count:
                    background |= labels == label
            alpha = np.where(background, 0, 255).astype(np.uint8)
            image = np.dstack([image, alpha])
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    def _draw_sprite(self, image, center_x, center_y, width, height, rotation):
        sprite = self._sprite
        scale = max(width, height) * 1.45 / max(sprite.shape[:2])
        new_size = (
            max(2, int(round(sprite.shape[1] * scale))),
            max(2, int(round(sprite.shape[0] * scale))),
        )
        resized = cv2.resize(sprite, new_size, interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation, 1.0)
        cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
        out_w = max(2, int(h * sin_a + w * cos_a))
        out_h = max(2, int(h * cos_a + w * sin_a))
        matrix[0, 2] += out_w / 2.0 - w / 2.0
        matrix[1, 2] += out_h / 2.0 - h / 2.0
        rotated = cv2.warpAffine(
            resized, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        self._alpha_composite(image, rotated, center_x, center_y)

    @staticmethod
    def _alpha_composite(background, foreground, center_x, center_y):
        h, w = foreground.shape[:2]
        x0, y0 = int(round(center_x - w / 2)), int(round(center_y - h / 2))
        bx0, by0 = max(0, x0), max(0, y0)
        bx1 = min(background.shape[1], x0 + w)
        by1 = min(background.shape[0], y0 + h)
        if bx1 <= bx0 or by1 <= by0:
            return
        fx0, fy0 = bx0 - x0, by0 - y0
        fg = foreground[fy0:fy0 + by1 - by0, fx0:fx0 + bx1 - bx0]
        alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
        roi = background[by0:by1, bx0:bx1].astype(np.float32)
        background[by0:by1, bx0:bx1] = np.clip(
            fg[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha),
            0, 255).astype(np.uint8)

    def _draw_geometric_target(self, image, center_x, center_y,
                               width, height, rotation):
        radius = max(2, int(round(max(width, height) / 2.0)))
        points = np.array([
            [-radius, 0], [0, -max(1, radius // 3)],
            [radius, 0], [0, max(1, radius // 3)],
        ], dtype=np.float32)
        angle = math.radians(rotation)
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ], dtype=np.float32)
        points = points @ rotation_matrix.T
        points += np.array([center_x, center_y], dtype=np.float32)
        points = np.round(points).astype(np.int32)
        cv2.fillConvexPoly(image, points, self.target_color)
        center = (int(round(center_x)), int(round(center_y)))
        rotor_radius = max(1, radius // 5)
        for point in points[[0, 2]]:
            cv2.circle(image, tuple(point), rotor_radius,
                       tuple(min(255, c + 35) for c in self.target_color), 1)
        cv2.circle(image, center, max(1, radius // 6), (15, 15, 15), -1)
