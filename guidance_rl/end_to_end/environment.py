"""Full-frame Gym environment: RGB frames in, direct velocity action out."""
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..config import load_config
from ..environments.camera_model import CameraModel
from ..environments.interception_core import InterceptionCore
from ..png_teacher import PNGTeacher
from .actions import ACTION_DIM, decode_action, encode_velocity_command, ned_to_body
from .renderer import FullFrameRenderer, RenderResult


CRITIC_DIM = 15


class EndToEndInterceptEnv(gym.Env):
    """Detector-free actor environment with privileged training labels.

    Actor observation:
        ``frames``: ``[history, 3, H, W]`` uint8 full RGB frames.

    Training-only ``info`` values:
        ``critic_obs``: simulator truth for the asymmetric critic.
        ``teacher_action``: PNG command converted to the direct body protocol.
        ``aux_targets``: future relative position, collision risk, visibility.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg=None, mode: str = "mixed", seed=None,
                 render_mode=None):
        super().__init__()
        self.cfg = cfg or load_config()
        self.mode = mode
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.render_rng = np.random.default_rng(self._render_seed(seed))

        c = self.cfg
        e2e = c.end_to_end
        self.dt = float(c.dynamics.dt)
        self.lost_steps = int(c.png.lost_steps)
        self.history_frames = int(e2e.model.history_frames)
        self.velocity_max = float(e2e.action.velocity_max)
        self.yaw_rate_max = float(e2e.action.yaw_rate_max)

        self.core = InterceptionCore(c, mode=mode, seed=seed)
        self.dynamics = self.core.dynamics
        self.camera = CameraModel(c.camera, self.rng)
        self.renderer = FullFrameRenderer(c.camera, e2e.render, self.render_rng)
        self.teacher = PNGTeacher.from_config(c)
        self._frame_history = deque(maxlen=self.history_frames)
        self._latest_render = None

        height = int(e2e.render.image_height)
        width = int(e2e.render.image_width)
        self.observation_space = spaces.Dict({
            "frames": spaces.Box(
                0, 255, (self.history_frames, 3, height, width), np.uint8),
        })
        self.action_space = spaces.Box(
            -1.0, 1.0, (ACTION_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.render_rng = np.random.default_rng(self._render_seed(seed))
        self.core.rng = self.rng
        self.camera.rng = self.rng
        self.renderer.set_rng(self.render_rng)

        self.core.reset()
        self.camera.reset()
        self.renderer.reset()
        self.teacher.reset()
        self._frame_history.clear()
        self.lost_count = 0
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)

        render_result, detection = self._observe()
        frame = self._to_chw(render_result.image)
        for _ in range(self.history_frames):
            self._frame_history.append(frame.copy())
        observation = self._observation()
        info = self._make_info(render_result, detection)
        return observation, info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0)
        command = decode_action(
            action, self.dynamics.yaw,
            velocity_max=self.velocity_max,
            yaw_rate_max=self.yaw_rate_max,
        )
        transition = self.core.step(command.ned_velocity, command.yaw_rate)

        render_result, detection = self._observe()
        self._frame_history.append(self._to_chw(render_result.image))
        self.lost_count = 0 if render_result.visible else self.lost_count + 1

        reward = self._reward(action, transition, render_result)
        terminated, truncated, outcome_name, reward = self._termination(
            transition, reward)
        self.previous_action = action.copy()

        info = self._make_info(render_result, detection)
        if outcome_name:
            info.update({
                "outcome": outcome_name,
                "min_dist": self.core.min_dist,
                "episode_steps": self.core.steps,
                "mode": self.core.episode_mode,
            })
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self._latest_render is None:
            return None
        return self._latest_render.image.copy()

    def close(self):
        self._frame_history.clear()

    # ------------------------------------------------------------------
    # Observation and labels
    # ------------------------------------------------------------------
    def _observe(self):
        dynamics = self.dynamics
        relative = self.core.target_pos - dynamics.pos
        rendered = self.renderer.render(
            relative, dynamics.roll, dynamics.pitch, dynamics.yaw,
            self.core.target_vel,
        )
        detection = self.camera.observe(
            relative, dynamics.roll, dynamics.pitch, dynamics.yaw)
        self._latest_render = rendered
        return rendered, detection

    @staticmethod
    def _to_chw(image):
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    @staticmethod
    def _render_seed(seed):
        if seed is None:
            return None
        # Keep visual randomization independent from target/detector RNG draws.
        return int(seed) ^ 0x5EED5EED

    def _observation(self):
        return {"frames": np.stack(tuple(self._frame_history), axis=0)}

    def _make_info(self, rendered: RenderResult, detection):
        dynamics = self.dynamics
        relative = self.core.target_pos - dynamics.pos
        relative_velocity = self.core.target_vel - dynamics.vel

        png_command = self.teacher.step(
            detection,
            dynamics.roll, dynamics.pitch, dynamics.yaw,
            dynamics.vel[0], dynamics.vel[1], dynamics.vel[2],
        )
        teacher_action = encode_velocity_command(
            [png_command.vx, png_command.vy, png_command.vz],
            png_command.yaw_rate,
            dynamics.yaw,
            velocity_max=self.velocity_max,
            yaw_rate_max=self.yaw_rate_max,
        )
        return {
            "critic_obs": self._critic_observation(
                relative, relative_velocity, rendered.visible),
            "teacher_action": teacher_action,
            "teacher_phase": png_command.phase,
            "aux_targets": self._auxiliary_targets(
                relative, relative_velocity, rendered.visible),
            "dist": float(np.linalg.norm(relative)),
            "visible": rendered.visible,
            # Useful for renderer/unit diagnostics; never fed to the actor.
            "bbox_label": rendered.bbox_normalized.copy(),
        }

    def _critic_observation(self, relative, relative_velocity, visible):
        e2e = self.cfg.end_to_end
        yaw = self.dynamics.yaw
        rel_body = ned_to_body(relative, yaw) / e2e.labels.position_norm
        rel_vel_body = ned_to_body(
            relative_velocity, yaw) / e2e.labels.velocity_norm
        target_vel_body = ned_to_body(
            self.core.target_vel, yaw) / e2e.labels.target_velocity_norm
        ego_vel_body = ned_to_body(
            self.dynamics.vel, yaw) / self.velocity_max
        observation = np.concatenate([
            rel_body,
            rel_vel_body,
            target_vel_body,
            ego_vel_body,
            np.array([
                -self.dynamics.pos[2] / e2e.labels.altitude_norm,
                np.linalg.norm(relative) / e2e.labels.position_norm,
                float(visible),
            ]),
        ]).astype(np.float32)
        if observation.shape != (CRITIC_DIM,):
            raise RuntimeError(f"critic observation has shape {observation.shape}")
        return observation

    def _auxiliary_targets(self, relative, relative_velocity, visible):
        labels = self.cfg.end_to_end.labels
        future_relative = (
            relative + relative_velocity * float(labels.future_horizon_s))
        future_body = ned_to_body(
            future_relative, self.dynamics.yaw) / labels.position_norm
        future_body = np.clip(future_body, -1.0, 1.0).astype(np.float32)

        velocity_squared = float(relative_velocity @ relative_velocity)
        if velocity_squared < 1e-8:
            time_to_closest = 0.0
        else:
            time_to_closest = float(np.clip(
                -(relative @ relative_velocity) / velocity_squared,
                0.0, labels.risk_horizon_s,
            ))
        closest = relative + relative_velocity * time_to_closest
        closest_distance = float(np.linalg.norm(closest))
        closing = float(relative @ relative_velocity) < 0.0
        risk = (
            np.exp(-closest_distance / float(labels.risk_radius))
            if closing else 0.0
        )
        return {
            "future_position": future_body,
            "collision_risk": np.float32(np.clip(risk, 0.0, 1.0)),
            "confidence": np.float32(float(visible)),
        }

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------
    def _reward(self, action, transition, rendered):
        cfg = self.cfg
        rw = cfg.env.reward
        reward = (
            rw.w_close
            * (transition.previous_distance - transition.distance)
            / self.dt / cfg.png.speed_cmd
        )
        reward -= rw.time_penalty
        reward -= rw.w_smooth * float(np.sum(
            (action - self.previous_action) ** 2))
        if rendered.visible:
            reward -= rw.w_fov * float(
                rendered.center_normalized @ rendered.center_normalized)
        else:
            reward -= rw.invalid_penalty
        return float(reward)

    def _termination(self, transition, reward):
        rw = self.cfg.env.reward
        terminated = truncated = False
        outcome = ""
        if transition.hit:
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
        return terminated, truncated, outcome, float(reward)

    # Compatibility with the learned-guidance evaluator's trajectory interface.
    @property
    def target_pos(self):
        return self.core.target_pos


def stack_training_info(infos):
    """Convert environment infos into dense arrays used by BC/PPO."""
    return {
        "critic_obs": np.stack([info["critic_obs"] for info in infos]),
        "future_position": np.stack([
            info["aux_targets"]["future_position"] for info in infos]),
        "collision_risk": np.asarray([
            info["aux_targets"]["collision_risk"] for info in infos],
            dtype=np.float32),
        "confidence": np.asarray([
            info["aux_targets"]["confidence"] for info in infos],
            dtype=np.float32),
    }


class VecEndToEndInterceptEnv:
    """Small synchronous vector wrapper with automatic episode reset."""

    def __init__(self, num_envs: int, cfg=None, mode: str = "mixed",
                 seed: int = 0):
        self.envs = [
            EndToEndInterceptEnv(cfg, mode=mode, seed=seed + index)
            for index in range(num_envs)
        ]
        self.num_envs = num_envs

    def reset(self):
        observations, infos = [], []
        for env in self.envs:
            observation, info = env.reset()
            observations.append(observation["frames"])
            infos.append(info)
        return np.stack(observations), stack_training_info(infos), infos

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                final_info = info
                observation, info = env.reset()
                info["final"] = final_info
            observations.append(observation["frames"])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            stack_training_info(infos),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()
