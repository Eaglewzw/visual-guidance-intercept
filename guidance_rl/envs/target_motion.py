"""目标机运动模型 —— 移植 uav_target_sim/src/uav_target_sim.cpp

C++ 以 0.1s 步进发位置 setpoint；此处换算为连续时间在任意 dt 下积分：
  circle      : R=5, ω=0.5 rad/s（cpp 构造函数 angular_speed_=0.05/0.1s 覆盖 hpp 默认值）
  sinusoidal  : 前向 1 m/s + 侧向 a=0.5*sin(0.5t)
  random_walk : v += N(0,σ=0.2) 每 0.1s 一步 → 按 sqrt(dt/0.1) 缩放保持同等扩散
  boundary    : 距中心超 80%*max_range 起施加 3*overshoot² 回复加速度（line 129-143）
额外增加 hover_escape 模式（对应已有实验 vpng_intercept_stats_hover_escape.csv）。

每集参数随机化（半径/角速度/速度/σ 抖动）由 cfg.target 控制，提升策略泛化。
"""
import math
import numpy as np


class TargetMotion:
    def __init__(self, mode: str, cfg, rng: np.random.Generator):
        """cfg: DotDict 的 target 节；center: 活动范围中心（reset 传入）"""
        self.mode = mode
        self.cfg = cfg
        self.rng = rng

    # ------------------------------------------------------------------
    def reset(self, center: np.ndarray):
        """center: 边界约束中心（NED，z 即目标飞行高度）"""
        cfg, rng = self.cfg, self.rng
        u = lambda j: 1.0 + rng.uniform(-j, j)   # 抖动因子

        self.center = center.astype(np.float64).copy()
        self.max_range = cfg.max_range + rng.uniform(-cfg.max_range_jitter,
                                                     cfg.max_range_jitter)
        self.t = 0.0
        self.vel = np.zeros(3)

        if self.mode == "circle":
            c = cfg.circle
            self.radius = c.radius * u(c.radius_jitter)
            self.omega = c.omega * u(c.omega_jitter) * rng.choice([-1.0, 1.0])
            self.theta = rng.uniform(0, 2 * math.pi)
            self.pos = self._circle_pos()
        elif self.mode == "sinusoidal":
            c = cfg.sinusoidal
            self.base_speed = c.base_speed * u(c.base_speed_jitter)
            self.amplitude = c.amplitude * u(c.amplitude_jitter)
            self.omega = c.omega * u(c.omega_jitter)
            heading = rng.uniform(0, 2 * math.pi)
            self.vel = np.array([self.base_speed * math.cos(heading),
                                 self.base_speed * math.sin(heading), 0.0])
            self.pos = self.center.copy()
        elif self.mode == "random_walk":
            c = cfg.random_walk
            self.sigma_v = c.sigma_v * u(c.sigma_v_jitter)
            self.max_speed = c.max_speed
            self.pos = self.center.copy()
        elif self.mode == "hover_escape":
            c = cfg.hover_escape
            self.trigger_range = c.trigger_range
            self.accel = c.accel
            self.max_speed = c.max_speed
            self.escaping = False
            self.pos = self.center.copy()
        else:
            raise ValueError(f"unknown target motion mode: {self.mode}")
        return self.pos.copy()

    # ------------------------------------------------------------------
    def step(self, dt: float, interceptor_pos: np.ndarray):
        """推进一步，返回 (pos, vel)"""
        self.t += dt
        if self.mode == "circle":
            self.theta = (self.theta + self.omega * dt) % (2 * math.pi)
            new_pos = self._circle_pos()
            self.vel = (new_pos - self.pos) / dt
            self.pos = new_pos
        elif self.mode == "sinusoidal":
            a_y = self.amplitude * math.sin(self.omega * self.t)
            # 侧向加速度作用在与初始前向垂直方向；C++ 中固定世界 y 轴，等价处理
            self.vel[1] += a_y * dt
            self._apply_boundary(dt)
            self.pos += self.vel * dt
        elif self.mode == "random_walk":
            scale = self.sigma_v * math.sqrt(dt / 0.1)   # 等效 0.1s 步进扩散
            self.vel[0] += scale * self.rng.standard_normal()
            self.vel[1] += scale * self.rng.standard_normal()
            speed = math.hypot(self.vel[0], self.vel[1])
            if speed > self.max_speed:
                self.vel[:2] *= self.max_speed / speed
            self._apply_boundary(dt)
            self.pos += self.vel * dt
        elif self.mode == "hover_escape":
            rel = interceptor_pos - self.pos
            dist = float(np.linalg.norm(rel))
            if not self.escaping and dist < self.trigger_range:
                self.escaping = True
            if self.escaping:
                # 水平方向背离拦截机加速，加少量切向扰动
                away = -rel.copy()
                away[2] = 0.0
                n = np.linalg.norm(away)
                if n > 1e-6:
                    away /= n
                    tangent = np.array([-away[1], away[0], 0.0])
                    jitter = 0.5 * self.rng.standard_normal()
                    acc = self.accel * (away + jitter * tangent)
                    self.vel[:2] += acc[:2] * dt
                speed = math.hypot(self.vel[0], self.vel[1])
                if speed > self.max_speed:
                    self.vel[:2] *= self.max_speed / speed
            self._apply_boundary(dt)
            self.pos += self.vel * dt
        return self.pos.copy(), self.vel.copy()

    # ------------------------------------------------------------------
    def _circle_pos(self):
        return self.center + np.array([
            self.radius * math.cos(self.theta),
            self.radius * math.sin(self.theta),
            0.0,
        ])

    def _apply_boundary(self, dt: float):
        """uav_target_sim.cpp apply_boundary() line 129-143"""
        d = self.pos[:2] - self.center[:2]
        dist = float(np.linalg.norm(d))
        if dist > self.max_range * 0.8 and dist > 0.01:
            overshoot = (dist - self.max_range * 0.8) / (self.max_range * 0.2)
            restore = 3.0 * overshoot * overshoot
            n = d / dist
            self.vel[0] -= restore * n[0] * dt
            self.vel[1] -= restore * n[1] * dt
