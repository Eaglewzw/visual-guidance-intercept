"""拦截机简化动力学 —— 速度指令一阶响应质点模型

对应真实链路：策略输出 NED 速度指令 → PX4 速度环跟踪。
PX4 速度环近似为一阶系统（τ_v ≈ 0.4s，可按 Gazebo 录制数据校准，
见 eval/record_gazebo_episode.py）。

前倾姿态模型：四旋翼通过倾斜产生水平加速度，
  pitch ≈ -atan(a_fwd/g)（前飞低头，pitch<0）
  roll  ≈  atan(a_side/g)
带 τ_att 一阶滞后。这决定相机指向 —— 正是 vision_png_control 中
k_ey 垂直视场补偿存在的原因，Gym 必须复现该效应。
"""
import math
import numpy as np

G = 9.81


class InterceptorDynamics:
    def __init__(self, cfg):
        """cfg: DotDict 的 dynamics 节"""
        self.tau_v = cfg.tau_v
        self.tau_yaw_rate = cfg.tau_yaw_rate
        self.tau_att = cfg.tau_att
        self.a_max = cfg.a_max
        self.v_max = cfg.v_max
        self.tilt_max = cfg.tilt_max
        self.reset(np.zeros(3), 0.0)

    def reset(self, pos: np.ndarray, yaw: float):
        self.pos = pos.astype(np.float64).copy()
        self.vel = np.zeros(3)
        self.yaw = float(yaw)
        self.yaw_rate = 0.0
        self.roll = 0.0
        self.pitch = 0.0

    def step(self, v_cmd: np.ndarray, yaw_rate_cmd: float, dt: float):
        """推进一步，返回 (pos_prev, pos)（供步内线段命中判定）"""
        pos_prev = self.pos.copy()

        # ---- 速度一阶响应 + 加速度限幅 ----
        acc = (np.asarray(v_cmd, dtype=np.float64) - self.vel) / self.tau_v
        a_norm = float(np.linalg.norm(acc))
        if a_norm > self.a_max:
            acc *= self.a_max / a_norm
        self.vel += acc * dt
        v_norm = float(np.linalg.norm(self.vel))
        if v_norm > self.v_max:
            self.vel *= self.v_max / v_norm
        self.pos += self.vel * dt

        # ---- 偏航角速率一阶响应 ----
        self.yaw_rate += (yaw_rate_cmd - self.yaw_rate) * dt / self.tau_yaw_rate
        self.yaw = math.atan2(math.sin(self.yaw + self.yaw_rate * dt),
                              math.cos(self.yaw + self.yaw_rate * dt))

        # ---- 前倾姿态（决定相机指向）----
        # 水平加速度投影到机头系
        a_fwd = acc[0] * math.cos(self.yaw) + acc[1] * math.sin(self.yaw)
        a_side = -acc[0] * math.sin(self.yaw) + acc[1] * math.cos(self.yaw)
        pitch_des = max(-self.tilt_max, min(self.tilt_max, -math.atan2(a_fwd, G)))
        roll_des = max(-self.tilt_max, min(self.tilt_max, math.atan2(a_side, G)))
        self.pitch += (pitch_des - self.pitch) * dt / self.tau_att
        self.roll += (roll_des - self.roll) * dt / self.tau_att

        return pos_prev, self.pos.copy()
