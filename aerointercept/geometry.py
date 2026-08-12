"""像素↔LOS 几何变换

与 ros2_ws/src/uav_vision_png/src/vision_png_control.cpp 逐行对齐：
  - R_C2B          : 构造函数 line 74-76
  - euler_to_r_b2n : png_calculate() Step 3, line 285-290（ZYX: Rz*Ry*Rx）
  - pixel_to_los   : png_calculate() Step 1-4, line 272-303
  - angles_to_velocity : handle_intercept() line 478-480

约定：
  NED 系，x=北 y=东 z=地；LOS_z 方位角 atan2(N, E)（北=0 的非常规约定，
  但与速度合成公式 vx=cos(v)sin(z)V / vy=cos(v)cos(z)V 自洽，照搬 C++ 不改动）。
"""
import math
import numpy as np

# 相机→机体旋转矩阵：Cx→By, Cy→Bz, Cz→Bx（vision_png_control.cpp line 74-76）
R_C2B = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def wrap_pi(angle: float) -> float:
    """归一化到 (-pi, pi]（C++ 中 while 循环的等价实现）"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def euler_to_r_b2n(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """机体→NED 旋转矩阵，ZYX 欧拉角：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def quat_to_euler(q0: float, q1: float, q2: float, q3: float):
    """PX4 四元数 (w,x,y,z) → ZYX 欧拉角，与 odometry 回调 line 130-134 一致"""
    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q0 * q2 - q3 * q1))))
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return roll, pitch, yaw


def pixel_to_los(ex: float, ey: float, focal: float,
                 roll: float, pitch: float, yaw: float):
    """像素误差 → NED 系 LOS 角

    输入 ex/ey 为 bbox 中心相对图像中心的像素误差。
    返回 (los_v, los_z, los_vec_ned)：
      los_v = atan2(z, sqrt(x^2+y^2))   仰角（NED Down 为正）
      los_z = atan2(N, E)               方位角
    """
    nt_cam = np.array([ex, ey, focal])
    r_b2n = euler_to_r_b2n(roll, pitch, yaw)
    nt_ned = r_b2n @ R_C2B @ nt_cam
    rxy = math.hypot(nt_ned[0], nt_ned[1])
    los_v = math.atan2(nt_ned[2], rxy)
    los_z = math.atan2(nt_ned[0], nt_ned[1])
    return los_v, los_z, nt_ned


def angles_to_velocity(v_angle_v: float, v_angle_z: float, speed: float):
    """期望速度角 → NED 速度指令（handle_intercept line 478-480）"""
    vx = math.cos(v_angle_v) * math.sin(v_angle_z) * speed  # N
    vy = math.cos(v_angle_v) * math.cos(v_angle_z) * speed  # E
    vz = math.sin(v_angle_v) * speed                        # D
    return vx, vy, vz


def project_to_pixel(rel_ned: np.ndarray, roll: float, pitch: float, yaw: float,
                     focal: float, cx: float, cy: float):
    """NED 相对位置 → 像素坐标（pixel_to_los 的逆过程，Gym 相机模型用）

    返回 (u, v, range) 或 None（目标在相机后方）。
    """
    r_b2n = euler_to_r_b2n(roll, pitch, yaw)
    p_body = r_b2n.T @ rel_ned
    p_cam = R_C2B.T @ p_body
    if p_cam[2] <= 0.1:  # 目标在相机平面后方/过近
        return None
    u = focal * p_cam[0] / p_cam[2] + cx
    v = focal * p_cam[1] / p_cam[2] + cy
    rng = float(np.linalg.norm(rel_ned))
    return u, v, rng


def segment_min_distance(p1a: np.ndarray, p1b: np.ndarray,
                         p2a: np.ndarray, p2b: np.ndarray) -> float:
    """一个步长内两条运动线段间的最小距离（高速接近时的连续命中判定）

    p1a→p1b 拦截机步内位移，p2a→p2b 目标步内位移。
    相对运动 d(t) = (p1a-p2a) + t*[(p1b-p1a)-(p2b-p2a)], t∈[0,1]
    """
    d0 = p1a - p2a
    dv = (p1b - p1a) - (p2b - p2a)
    dv2 = float(dv @ dv)
    if dv2 < 1e-12:
        return float(np.linalg.norm(d0))
    t = -float(d0 @ dv) / dv2
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(d0 + t * dv))
