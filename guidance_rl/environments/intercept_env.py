"""学习制导的拦截 Gym 环境

只建模 INTERCEPT 阶段（起飞/搜索由部署节点的状态机处理，不学习）：
reset 时目标已在视场内，episode 在命中/长时间丢失/超时/触地时结束。

观测  : features.FeatureBuilder 的 15 维向量（与部署完全一致）
动作  : features.decode_action 的 4 维 [-1,1]（相对 LOS 偏移参数化）
特权观测: info["critic_obs"] 9 维（相对位置/相对速度/目标速度，仅训练 Critic 用，
         对应真实系统中"统计专用"的 GPS 数据 —— 部署时不可用）
teacher : info["teacher_action"] PNG 老师对当前帧的动作标签（BC 采样/对比用）
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import load_config
from ..features import (
    FeatureBuilder, decode_action, encode_action_from_velocity, OBS_DIM, ACT_DIM,
)
from ..png_teacher import PNGTeacher
from .camera_model import CameraModel
from .interception_core import InterceptionCore

PRIV_DIM = 9
# 特权观测归一化
_PRIV_POS_NORM = 30.0
_PRIV_VEL_NORM = 10.0
_PRIV_TVEL_NORM = 5.0


class InterceptEnv(gym.Env):
    """单环境。mode: circle/sinusoidal/random_walk/hover_escape/mixed"""

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

        self.core = InterceptionCore(c, mode=mode, seed=seed)
        self.dynamics = self.core.dynamics
        self.camera = CameraModel(c.camera, self.rng)
        self.fb = FeatureBuilder(c.camera.focal_length,
                                 c.camera.image_width, c.camera.image_height)
        self.teacher = PNGTeacher.from_config(c)

        # 动作解码常数
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

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    # ==================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.camera.rng = self.rng
        self.core.rng = self.rng
        self.core.reset()
        self._episode_mode = self.core.episode_mode

        self.camera.reset()
        self.fb.reset()
        self.teacher.reset()

        self.lost_count = 0
        self.prev_dist = self.core.previous_distance
        self.prev_action = np.zeros(ACT_DIM, dtype=np.float32)

        # 首帧观测（动力学未推进，先观测一次）
        det = self._observe()
        obs = self._build_obs(det)
        info = self._make_info(det, hit=False)
        return obs, info

    # ==================================================================
    def step(self, action):
        c = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # ---- 动作解码（用 last-known LOS）→ 速度指令 ----
        vx, vy, vz, yaw_rate = decode_action(
            action, self.fb.los_v, self.fb.los_z, **self._decode_kw)

        transition = self.core.step(np.array([vx, vy, vz]), yaw_rate)
        hit = transition.hit

        # ---- 观测 ----
        det = self._observe()
        if det is None:
            self.lost_count += 1
        else:
            self.lost_count = 0
        obs = self._build_obs(det)

        # ---- 奖励 ----
        rw = self.rw
        reward = (rw.w_close
                  * (transition.previous_distance - transition.distance)
                  / self.dt / c.png.speed_cmd)
        reward -= rw.time_penalty
        reward -= rw.w_smooth * float(np.sum((action - self.prev_action) ** 2))
        if det is not None:
            # FOV 偏离惩罚（归一化像素误差二次型）
            exn = self.fb.ex / (c.camera.image_width / 2.0)
            eyn = self.fb.ey / (c.camera.image_height / 2.0)
            reward -= rw.w_fov * (exn * exn + eyn * eyn)
        else:
            reward -= rw.invalid_penalty

        # ---- 终止判定 ----
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
        elif transition.touched_ground:
            reward -= rw.ground_penalty
            terminated = True
            outcome = "ground"
        elif transition.timed_out:
            reward -= rw.timeout_penalty
            truncated = True
            outcome = "timeout"

        self.prev_action = action
        self.prev_dist = self.core.previous_distance

        info = self._make_info(det, hit=hit)
        if outcome:
            info["outcome"] = outcome
            info["min_dist"] = self.core.min_dist
            info["episode_steps"] = self.core.steps
            info["mode"] = self.core.episode_mode
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    #  内部
    # ==================================================================
    def _observe(self):
        rel = self.core.target_pos - self.dynamics.pos
        return self.camera.observe(rel, self.dynamics.roll,
                                   self.dynamics.pitch, self.dynamics.yaw)

    def _build_obs(self, det):
        d = self.dynamics
        return self.fb.build(det, d.roll, d.pitch, d.yaw,
                             d.vel[0], d.vel[1], d.vel[2], d.pos[2], self.dt)

    def _make_info(self, det, hit: bool):
        d = self.dynamics
        rel = self.core.target_pos - d.pos
        critic_obs = np.concatenate([
            rel / _PRIV_POS_NORM,
            (self.core.target_vel - d.vel) / _PRIV_VEL_NORM,
            self.core.target_vel / _PRIV_TVEL_NORM,
        ]).astype(np.float32)

        # PNG 老师动作标签（老师状态每步推进一次，与策略共享同一检测流）
        cmd = self.teacher.step(det, d.roll, d.pitch, d.yaw,
                                d.vel[0], d.vel[1], d.vel[2])
        teacher_action = encode_action_from_velocity(
            cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate,
            self.fb.los_v, self.fb.los_z, **self._encode_kw)

        return {
            "critic_obs": critic_obs,
            "teacher_action": teacher_action,
            "teacher_phase": cmd.phase,
            "dist": float(np.linalg.norm(rel)),
            "valid": det is not None,
        }

    # Compatibility properties used by the existing evaluators.
    @property
    def target_pos(self):
        return self.core.target_pos

    @property
    def target_vel(self):
        return self.core.target_vel

    @property
    def target(self):
        return self.core.target

    @property
    def steps(self):
        return self.core.steps

    @property
    def min_dist(self):
        return self.core.min_dist


# ======================================================================
#  简易同步向量环境（自动 reset）
# ======================================================================
class VecInterceptEnv:
    """N 个 InterceptEnv 串行步进；环境很轻，串行已够 PPO 用"""

    def __init__(self, num_envs: int, cfg=None, mode: str = "mixed", seed: int = 0):
        self.envs = [InterceptEnv(cfg, mode, seed=seed + i) for i in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        obs, priv = [], []
        for e in self.envs:
            o, info = e.reset()
            obs.append(o)
            priv.append(info["critic_obs"])
        return np.stack(obs), np.stack(priv)

    def step(self, actions):
        obs, priv, rews, dones, infos = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(a)
            done = term or trunc
            if done:
                # 记录终局信息后自动 reset
                final_info = info
                o, info = e.reset()
                info["final"] = final_info
            obs.append(o)
            priv.append(info["critic_obs"])
            rews.append(r)
            dones.append(done)
            infos.append(info)
        return (np.stack(obs), np.stack(priv),
                np.array(rews, dtype=np.float32),
                np.array(dones, dtype=np.float32), infos)
