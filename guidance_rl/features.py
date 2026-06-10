"""观测特征构造与动作编解码 —— Gym 训练与 ROS2 部署共用的唯一实现

特征向量（15 维，FEATURE_VERSION 标识，导出模型时写入 policy_meta.json）：
   0  los_v / (π/4)              仰角归一化
   1  sin(los_z)                 方位角（sin/cos 避免 ±π 不连续）
   2  cos(los_z)
   3  clip(dlos_v/dt / 2, ±2)    LOS 仰角速率（PNG 的核心输入量）
   4  clip(dlos_z/dt / 2, ±2)    LOS 方位角速率
   5  (ln w - ln 20) / 2         bbox 宽（对数尺度 ≈ 距离的代理）
   6  (ln h - ln 20) / 2
   7  clip(dlnw/dt, ±2)          bbox 尺寸变化率（接近速度代理）
   8  valid                      本帧检测有效标志
   9  min(age, 2)/2              距上一有效帧的时间（s，归一化）
  10  vx / 8                     自身 NED 速度
  11  vy / 8
  12  vz / 8
  13  |V| / 8
  14  -local_z / 20              高度（NED z 取负归一化）

丢失帧处理：保持 last-known LOS/尺寸（0-2、5-6 维不变），速率项清零，
valid=0、age 递增 —— 策略由此学习惯性续飞行为。

动作（4 维，tanh 输出 ∈ [-1,1]，带 PNG 归纳偏置：零动作 ≈ 纯追踪）：
  a0 → v_angle_v = clamp(los_v + a0*dv_angle_max, ±elev_clamp)
  a1 → v_angle_z = los_z + a1*dv_angle_max          （相对 LOS 的前置偏移）
  a2 → speed     = speed_min + (speed_cmd-speed_min)*(a2+1)/2
  a3 → yaw_rate  = a3 * yaw_rate_max
速度合成与 handle_intercept() line 478-480 完全一致（angles_to_velocity）。
"""
import math
import numpy as np

from .geometry import pixel_to_los, angles_to_velocity, wrap_pi

FEATURE_VERSION = 1
OBS_DIM = 15
ACT_DIM = 4

# 归一化常数
_V_NORM = 8.0          # 速度归一化 (m/s)
_LOS_RATE_NORM = 2.0   # LOS 角速率归一化 (rad/s)
_LOGW_REF = math.log(20.0)
_AGE_MAX = 2.0         # 丢失时间饱和 (s)
_Z_NORM = 20.0


class FeatureBuilder:
    """有状态特征构造器，每个控制周期（20Hz）调用 build() 一次"""

    def __init__(self, focal: float, image_width: int, image_height: int):
        self.focal = focal
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0
        self.reset()

    def reset(self):
        self.has_los = False
        self.los_v = 0.0
        self.los_z = 0.0
        self.log_w = _LOGW_REF
        self.log_h = _LOGW_REF
        self.age = 0.0
        self.ex = 0.0
        self.ey = 0.0

    def build(self, det, roll, pitch, yaw, vx, vy, vz, local_z, dt) -> np.ndarray:
        """det: (x, y, w, h) 或 None（丢失帧）"""
        valid = det is not None
        dlos_v = dlos_z = dlog_w = 0.0

        if valid:
            x, y, w, h = det
            w = max(float(w), 1.0)
            h = max(float(h), 1.0)
            self.ex = (x + w / 2.0) - self.cx
            self.ey = (y + h / 2.0) - self.cy
            los_v, los_z, _ = pixel_to_los(self.ex, self.ey, self.focal,
                                           roll, pitch, yaw)
            log_w, log_h = math.log(w), math.log(h)
            if self.has_los:
                dlos_v = wrap_pi(los_v - self.los_v) / dt
                dlos_z = wrap_pi(los_z - self.los_z) / dt
                dlog_w = (log_w - self.log_w) / dt
            self.los_v, self.los_z = los_v, los_z
            self.log_w, self.log_h = log_w, log_h
            self.has_los = True
            self.age = 0.0
        else:
            self.age += dt

        v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        clip = lambda v, lim: max(-lim, min(lim, v))

        return np.array([
            self.los_v / (math.pi / 4.0),
            math.sin(self.los_z),
            math.cos(self.los_z),
            clip(dlos_v / _LOS_RATE_NORM, 2.0),
            clip(dlos_z / _LOS_RATE_NORM, 2.0),
            (self.log_w - _LOGW_REF) / 2.0,
            (self.log_h - _LOGW_REF) / 2.0,
            clip(dlog_w, 2.0),
            1.0 if valid else 0.0,
            min(self.age, _AGE_MAX) / _AGE_MAX,
            vx / _V_NORM,
            vy / _V_NORM,
            vz / _V_NORM,
            v_norm / _V_NORM,
            -local_z / _Z_NORM,
        ], dtype=np.float32)


# ============================================================
#  动作编解码
# ============================================================

def decode_action(a, los_v, los_z, *, dv_angle_max=0.8,
                  speed_min=2.0, speed_cmd=5.0, yaw_rate_max=1.0,
                  elev_clamp=math.pi / 4.0):
    """动作 [-1,1]^4 → (vx, vy, vz, yaw_rate) NED 速度指令

    los_v/los_z 为 FeatureBuilder 维护的 last-known LOS 角。
    """
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    v_angle_v = max(-elev_clamp, min(elev_clamp, los_v + a[0] * dv_angle_max))
    v_angle_z = los_z + a[1] * dv_angle_max
    speed = speed_min + (speed_cmd - speed_min) * (a[2] + 1.0) / 2.0
    yaw_rate = a[3] * yaw_rate_max
    vx, vy, vz = angles_to_velocity(v_angle_v, v_angle_z, speed)
    return vx, vy, vz, yaw_rate


def encode_action(v_angle_v, v_angle_z, speed, yaw_rate, los_v, los_z, *,
                  dv_angle_max=0.8, speed_min=2.0, speed_cmd=5.0,
                  yaw_rate_max=1.0):
    """期望速度角 → 动作标签（decode_action 的逆）"""
    a0 = np.clip((v_angle_v - los_v) / dv_angle_max, -1.0, 1.0)
    a1 = np.clip(wrap_pi(v_angle_z - los_z) / dv_angle_max, -1.0, 1.0)
    a2 = np.clip(2.0 * (speed - speed_min) / (speed_cmd - speed_min) - 1.0, -1.0, 1.0)
    a3 = np.clip(yaw_rate / yaw_rate_max, -1.0, 1.0)
    return np.array([a0, a1, a2, a3], dtype=np.float32)


def encode_action_from_velocity(vx, vy, vz, yaw_rate, los_v, los_z, *,
                                dv_angle_max=0.8, speed_min=2.0, speed_cmd=5.0,
                                yaw_rate_max=1.0):
    """最终 NED 速度指令 → 动作标签（BC 监督用）

    把 PNG 的 vz 垂直补偿（k_ey*ey）折算进等效仰角后再编码，
    保证 decode(encode(cmd)) ≈ cmd（限幅边界处有截断误差）。
    """
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-6:
        return np.array([0.0, 0.0, -1.0, np.clip(yaw_rate / yaw_rate_max, -1, 1)],
                        dtype=np.float32)
    v_angle_v = math.atan2(vz, math.hypot(vx, vy))
    v_angle_z = math.atan2(vx, vy)
    return encode_action(v_angle_v, v_angle_z, speed, yaw_rate, los_v, los_z,
                         dv_angle_max=dv_angle_max, speed_min=speed_min,
                         speed_cmd=speed_cmd, yaw_rate_max=yaw_rate_max)
