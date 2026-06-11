"""阶段二拦截 Gym 环境 —— 图像观测 + 几何真值（Critic 用）

与阶段一 InterceptEnv 的区别：
  - obs 从 15 维几何特征变为 (288×288×3 图像, 8维自身状态)
  - info 额外提供 bbox_label（辅助头真值）和 gt_obs（Critic 输入的阶段一特征）
  - 使用 uav_renderer 替代 camera_model 的纯 bbox 输出
  - 动力学/目标运动/PNG 老师完全复用阶段一

观测空间 (gym.spaces.Dict):
  "image":    Box(0,255, (3,288,288), uint8)  搜索区域裁剪
  "ego_state": Box(-inf,inf, (8,), float32)    自身速度+姿态+高度

info 字典（训练用，部署不可得）:
  "teacher_action": [4]        BC 标签
  "critic_obs":      [15]      阶段一几何特征（Critic 输入）
  "priv_obs":        [9]       特权真值（Critic 输入）
  "bbox_label":      [4]       辅助头真值 (cx,cy,w,h 归一化)
  "conf_label":      1         辅助头真值 (1=可见, 0=丢失)
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import load_config
from ..features import (
    FeatureBuilder, decode_action, encode_action_from_velocity, OBS_DIM, ACT_DIM,
)
from ..geometry import segment_min_distance
from ..png_teacher import PNGTeacher
from .target_motion import TargetMotion
from .interceptor_dynamics import InterceptorDynamics
from .uav_renderer import UAVRenderer

PRIV_DIM = 9
CROP_SIZE = 288
EGO_DIM = 8
# ego 归一化（与阶段一一致）
_V_NORM = 8.0
_Z_NORM = 20.0


class InterceptEnvV2(gym.Env):
    """阶段二环境：图像观测 + 几何真值（Critic）。mode: mixed / 具体运动模式"""

    metadata = {"render_modes": []}

    def __init__(self, cfg=None, mode: str = "mixed", seed=None):
        super().__init__()
        self.cfg = cfg or load_config()
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        c = self.cfg
        self.dt = c.dynamics.dt
        self.hit_radius = c.env.hit_radius
        self.max_steps = c.env.episode_max_steps
        self.lost_steps = c.png.lost_steps
        self.rw = c.env.reward

        self.dynamics = InterceptorDynamics(c.dynamics)
        spr = c.get("v2", {}).get("sprite_path", "")
        spr_prob = c.get("v2", {}).get("sprite_prob", 0.8)
        self.renderer = UAVRenderer(c, self.rng,
                                     sprite_path=spr or None,
                                     sprite_prob=spr_prob)
        self.fb = FeatureBuilder(c.camera.focal_length,
                                 c.camera.image_width, c.camera.image_height)
        self.teacher = PNGTeacher.from_config(c)

        self._decode_kw = dict(
            dv_angle_max=c.action.dv_angle_max,
            speed_min=c.png.speed_min, speed_cmd=c.png.speed_cmd,
            yaw_rate_max=c.png.yaw_rate_max, elev_clamp=c.png.elev_clamp,
        )
        self._encode_kw = dict(
            dv_angle_max=c.action.dv_angle_max,
            speed_min=c.png.speed_min, speed_cmd=c.png.speed_cmd,
            yaw_rate_max=c.png.yaw_rate_max,
        )

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (3, CROP_SIZE, CROP_SIZE), dtype=np.uint8),
            "ego_state": spaces.Box(-np.inf, np.inf, (EGO_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    # ==================================================================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.renderer.rng = self.rng
        c = self.cfg

        # ---- 拦截机 ----
        yaw0 = self.rng.uniform(-math.pi, math.pi)
        self.dynamics.reset(np.array([0.0, 0.0, c.env.standby_alt]), yaw0)

        # ---- 目标 ----
        mode = self.mode
        if mode == "mixed":
            mode = str(self.rng.choice(c.target.modes, p=c.target.mode_probs))
        self._episode_mode = mode

        r0 = self.rng.uniform(*c.env.spawn_range)
        az = yaw0 + self.rng.uniform(-c.env.spawn_az_jitter, c.env.spawn_az_jitter)
        alt = c.target.alt + self.rng.uniform(-c.target.alt_jitter, c.target.alt_jitter)
        center = self.dynamics.pos + np.array([
            r0 * math.cos(az), r0 * math.sin(az), 0.0])
        center[2] = alt

        self.target = TargetMotion(mode, c.target, self.rng)
        self.target_pos = self.target.reset(center)
        self.target_vel = np.zeros(3)

        self.renderer.reset()
        self.fb.reset()
        self.teacher.reset()

        self.steps = 0
        self.lost_count = 0
        rel = self.target_pos - self.dynamics.pos
        self.min_dist = float(np.linalg.norm(rel))
        self.prev_dist = self.min_dist
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        # 首帧观测
        image, bbox_label, conf_label, gt_obs, det = self._observe(rel)
        self._gt_obs = gt_obs
        self._det = det
        obs = self._make_obs(image)
        info = self._make_info(rel, bbox_label, conf_label, hit=False)
        return obs, info

    # ==================================================================
    def step(self, action):
        c = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        vx, vy, vz, yaw_rate = decode_action(
            action, self.fb.los_v, self.fb.los_z, **self._decode_kw)

        # ---- 推进 ----
        i_prev, i_now = self.dynamics.step(
            np.array([vx, vy, vz]), yaw_rate, self.dt)
        t_prev = self.target_pos.copy()
        self.target_pos, self.target_vel = self.target.step(self.dt, i_now)
        rel = self.target_pos - i_now

        step_min = segment_min_distance(i_prev, i_now, t_prev, self.target_pos)
        dist = float(np.linalg.norm(rel))
        self.min_dist = min(self.min_dist, step_min)
        hit = step_min < self.hit_radius

        # ---- 观测 ----
        image, bbox_label, conf_label, gt_obs, det = self._observe(rel)
        self._gt_obs = gt_obs
        self._det = det
        self.lost_count = 0 if conf_label > 0.5 else self.lost_count + 1
        obs = self._make_obs(image)

        # ---- 奖励（同阶段一）----
        rw = self.rw
        reward = rw.w_close * (self.prev_dist - dist) / self.dt / c.png.speed_cmd
        reward -= rw.time_penalty
        reward -= rw.w_smooth * float(np.sum((action - self.prev_action) ** 2))
        if conf_label > 0.5:
            exn = self.fb.ex / (c.camera.image_width / 2.0)
            eyn = self.fb.ey / (c.camera.image_height / 2.0)
            reward -= rw.w_fov * (exn * exn + eyn * eyn)
        else:
            reward -= rw.invalid_penalty

        self.steps += 1
        terminated = False
        truncated = False
        outcome = ""
        if hit:
            reward += rw.hit_bonus
            terminated = True
            outcome = "hit"
        elif self.lost_count >= self.lost_steps:
            reward -= rw.fov_lost_penalty
            terminated = True
            outcome = "fov_lost"
        elif i_now[2] > c.env.ground_z:
            reward -= rw.ground_penalty
            terminated = True
            outcome = "ground"
        elif self.steps >= self.max_steps:
            truncated = True
            outcome = "timeout"

        self.prev_dist = dist
        self.prev_action = action

        info = self._make_info(rel, bbox_label, conf_label, hit=hit)
        if outcome:
            info["outcome"] = outcome
            info["min_dist"] = self.min_dist
            info["episode_steps"] = self.steps
            info["mode"] = self._episode_mode
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    def _observe(self, rel_ned: np.ndarray):
        """渲染图像 + 阶段一几何特征（用于 Critic + 老师）"""
        d = self.dynamics
        image, bbox_label = self.renderer.render(rel_ned, d.roll, d.pitch, d.yaw,
                                                  target_vel_ned=self.target_vel)

        # 检测模拟：出 FOV / 在后方 → 丢失
        res = True  # renderer 总会产出图像，这里判断几何 FOV
        from ..geometry import project_to_pixel
        pix = project_to_pixel(rel_ned, d.roll, d.pitch, d.yaw,
                               self.cfg.camera.focal_length,
                               self.cfg.camera.image_width / 2.0,
                               self.cfg.camera.image_height / 2.0)
        if pix is None:
            det = None
            conf_label = 0.0
        else:
            u, v, _ = pix
            in_fov = (0 <= u < self.cfg.camera.image_width
                      and 0 <= v < self.cfg.camera.image_height)
            # 模拟漏检（与阶段一 camera_model 同分布）
            w = self.cfg.camera.focal_length * self.cfg.camera.target_size_w / max(
                float(np.linalg.norm(rel_ned)), 0.5)
            p_miss = (self.cfg.camera.miss_base
                      + self.cfg.camera.miss_small_scale
                      * math.exp(-w / self.cfg.camera.miss_small_px))
            if in_fov and self.rng.random() >= p_miss:
                # 模拟 bbox（中心噪声）
                sigma_c = max(1.0, self.cfg.camera.pixel_noise_frac * w)
                u_n = u + self.rng.normal(0, sigma_c)
                v_n = v + self.rng.normal(0, sigma_c)
                w_n = max(1, int(round(w * max(0.3, 1.0 + self.rng.normal(
                    0, self.cfg.camera.size_noise_frac)))))
                h_n = max(1, int(round(
                    w * self.cfg.camera.target_size_h / self.cfg.camera.target_size_w
                    * max(0.3, 1.0 + self.rng.normal(0, self.cfg.camera.size_noise_frac)))))
                det = (int(round(u_n - w_n / 2)), int(round(v_n - h_n / 2)), w_n, h_n)
                conf_label = 1.0
            else:
                det = None
                conf_label = 0.0

        # 构造阶段一特征（供 Critic + 老师）。fb.build() 返回完整 15 维，捕获备用
        gt_obs = self.fb.build(det, d.roll, d.pitch, d.yaw,
                               d.vel[0], d.vel[1], d.vel[2], d.pos[2], self.dt)

        return image, bbox_label, conf_label, gt_obs, det

    def _make_obs(self, image):
        """actor 观测: (image, ego_state)"""
        d = self.dynamics
        v_norm = math.sqrt(d.vel[0] ** 2 + d.vel[1] ** 2 + d.vel[2] ** 2)
        ego = np.array([
            d.vel[0] / _V_NORM, d.vel[1] / _V_NORM, d.vel[2] / _V_NORM,
            v_norm / _V_NORM, d.roll, d.pitch, d.yaw / math.pi,
            -d.pos[2] / _Z_NORM,
        ], dtype=np.float32)
        # HWC → CHW (torch 格式)
        image_chw = np.transpose(image, (2, 0, 1)).copy()
        return {"image": image_chw, "ego_state": ego}

    def _make_info(self, rel_ned, bbox_label, conf_label, hit):
        d = self.dynamics
        relv = self.target_vel - d.vel
        priv = np.concatenate([
            rel_ned / 30.0, relv / 10.0, self.target_vel / 5.0,
        ]).astype(np.float32)

        # Critic 输入 = 缓存的最新阶段一几何观测（_observe 中 fb.build 的返回值）
        gt_obs = self._gt_obs

        # 老师动作：使用缓存的最新检测帧
        cmd = self.teacher.step(
            self._det, d.roll, d.pitch, d.yaw,
            d.vel[0], d.vel[1], d.vel[2])
        teacher_act = encode_action_from_velocity(
            cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate,
            self.fb.los_v, self.fb.los_z, **self._encode_kw)

        return {
            "critic_obs": gt_obs,
            "priv_obs": priv,
            "teacher_action": teacher_act,
            "bbox_label": bbox_label,
            "conf_label": np.array([conf_label], dtype=np.float32),
            "dist": float(np.linalg.norm(rel_ned)),
            "valid": conf_label > 0.5,
        }


# ======================================================================
class VecInterceptEnvV2:
    """N 个 InterceptEnvV2 串行步进（图像量大，少量并行即可）"""

    def __init__(self, num_envs: int, cfg=None, mode: str = "mixed", seed: int = 0):
        self.envs = [InterceptEnvV2(cfg, mode, seed=seed + i)
                     for i in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        images, ego_states, critic_obs, priv_obs = [], [], [], []
        for e in self.envs:
            o, info = e.reset()
            images.append(o["image"])
            ego_states.append(o["ego_state"])
            critic_obs.append(info["critic_obs"])
            priv_obs.append(info["priv_obs"])
        return (np.stack(images), np.stack(ego_states),
                np.stack(critic_obs), np.stack(priv_obs))

    def step(self, actions):
        images, ego, critic, priv = [], [], [], []
        rews, dones, infos = [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(a)
            done = term or trunc
            if done:
                final_info = info
                o, info = e.reset()
                info["final"] = final_info
            images.append(o["image"])
            ego.append(o["ego_state"])
            critic.append(info["critic_obs"])
            priv.append(info["priv_obs"])
            rews.append(r)
            dones.append(done)
            infos.append(info)
        return (np.stack(images), np.stack(ego),
                np.stack(critic), np.stack(priv),
                np.array(rews, dtype=np.float32),
                np.array(dones, dtype=np.float32), infos)
