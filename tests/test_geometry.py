"""geometry / png_teacher / features 与 C++ 实现的数值一致性测试

手算样例对照 vision_png_control.cpp 的公式，确保移植无符号/约定错误。
"""
import math
import numpy as np
import pytest

from guidance_rl.geometry import (
    R_C2B, wrap_pi, euler_to_r_b2n, pixel_to_los,
    angles_to_velocity, project_to_pixel, segment_min_distance, quat_to_euler,
)
from guidance_rl.png_teacher import PNGTeacher
from guidance_rl.features import (
    FeatureBuilder, decode_action, encode_action_from_velocity, OBS_DIM,
)

FOCAL = 1397.2
W, H = 1920, 1080


# ============================================================
#  基础几何
# ============================================================

def test_wrap_pi():
    assert wrap_pi(0.0) == 0.0
    assert abs(wrap_pi(3 * math.pi) - math.pi) < 1e-12
    assert abs(wrap_pi(-3 * math.pi) + math.pi) < 1e-12
    assert abs(wrap_pi(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-12


def test_r_c2b_mapping():
    """Cx→By, Cy→Bz, Cz→Bx（C++ line 73-76 注释）"""
    cam_x = np.array([1, 0, 0])
    cam_y = np.array([0, 1, 0])
    cam_z = np.array([0, 0, 1])
    assert np.allclose(R_C2B @ cam_x, [0, 1, 0])   # Cx → By
    assert np.allclose(R_C2B @ cam_y, [0, 0, 1])   # Cy → Bz
    assert np.allclose(R_C2B @ cam_z, [1, 0, 0])   # Cz → Bx


def test_euler_identity():
    assert np.allclose(euler_to_r_b2n(0, 0, 0), np.eye(3))


def test_euler_yaw_90():
    """yaw=+90°：机体 x（前）指向东"""
    r = euler_to_r_b2n(0, 0, math.pi / 2)
    body_x_in_ned = r @ np.array([1, 0, 0])
    assert np.allclose(body_x_in_ned, [0, 1, 0], atol=1e-12)


def test_quat_roundtrip():
    roll, pitch, yaw = 0.1, -0.2, 0.7
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    r2, p2, y2 = quat_to_euler(q0, q1, q2, q3)
    assert abs(r2 - roll) < 1e-9 and abs(p2 - pitch) < 1e-9 and abs(y2 - yaw) < 1e-9


# ============================================================
#  像素 → LOS（对照 png_calculate Step 1-4）
# ============================================================

def test_los_center_level_north():
    """目标在图像正中、机体水平朝北 → LOS 水平指北：los_v=0, los_z=atan2(N,E)=π/2"""
    los_v, los_z, vec = pixel_to_los(0.0, 0.0, FOCAL, 0, 0, 0)
    # 相机光轴 Cz → Bx（前）→ NED 北
    assert np.allclose(vec / np.linalg.norm(vec), [1, 0, 0], atol=1e-12)
    assert abs(los_v) < 1e-12
    assert abs(los_z - math.pi / 2) < 1e-12   # atan2(1, 0)


def test_los_target_right():
    """目标在画面右侧（ex>0）→ LOS 偏东 → los_z 减小（atan2(N,E) 约定）"""
    los_v0, los_z0, _ = pixel_to_los(0, 0, FOCAL, 0, 0, 0)
    _, los_z1, vec = pixel_to_los(200.0, 0, FOCAL, 0, 0, 0)
    assert vec[1] > 0                  # E 分量为正（偏右=偏东）
    assert los_z1 < los_z0


def test_los_target_below():
    """目标在画面下方（ey>0，图像 y 向下）→ LOS 偏向 NED Down → los_v > 0"""
    los_v, _, vec = pixel_to_los(0, 200.0, FOCAL, 0, 0, 0)
    assert vec[2] > 0
    assert los_v > 0


def test_los_hand_computed():
    """手算样例：ex=100, ey=-50, 水平姿态 yaw=0
    Nt_cam=[100,-50,1397.2] → body=[1397.2,100,-50] → NED 同 body
    los_v = atan2(-50, sqrt(1397.2²+100²)), los_z = atan2(1397.2, 100)
    """
    los_v, los_z, _ = pixel_to_los(100.0, -50.0, FOCAL, 0, 0, 0)
    exp_v = math.atan2(-50.0, math.hypot(1397.2, 100.0))
    exp_z = math.atan2(1397.2, 100.0)
    assert abs(los_v - exp_v) < 1e-12
    assert abs(los_z - exp_z) < 1e-12


def test_velocity_synthesis():
    """对照 handle_intercept line 478-480"""
    vv, vz_ang, V = 0.2, 1.1, 5.0
    vx, vy, vz = angles_to_velocity(vv, vz_ang, V)
    assert abs(vx - math.cos(vv) * math.sin(vz_ang) * V) < 1e-12
    assert abs(vy - math.cos(vv) * math.cos(vz_ang) * V) < 1e-12
    assert abs(vz - math.sin(vv) * V) < 1e-12
    # 合成速度模长 = V
    assert abs(math.sqrt(vx**2 + vy**2 + vz**2) - V) < 1e-9


def test_velocity_los_roundtrip():
    """angles_to_velocity 与 LOS 角定义自洽：合成速度方向的 LOS 角 = 输入角"""
    vv, vz_ang = -0.3, 2.4
    vx, vy, vz = angles_to_velocity(vv, vz_ang, 5.0)
    assert abs(math.atan2(vz, math.hypot(vx, vy)) - vv) < 1e-9
    assert abs(wrap_pi(math.atan2(vx, vy) - vz_ang)) < 1e-9


# ============================================================
#  投影与 LOS 的互逆性（相机模型正确性的关键）
# ============================================================

@pytest.mark.parametrize("roll,pitch,yaw", [
    (0, 0, 0), (0.1, -0.2, 0.7), (-0.05, 0.3, 0.4),
])
def test_project_pixel_los_roundtrip(roll, pitch, yaw):
    """已知相对位置 → 投影像素 → pixel_to_los → 方向应与相对位置一致"""
    rel = np.array([12.0, 5.0, -2.0])
    res = project_to_pixel(rel, roll, pitch, yaw, FOCAL, W / 2, H / 2)
    assert res is not None
    u, v, rng = res
    ex, ey = u - W / 2, v - H / 2
    _, _, vec = pixel_to_los(ex, ey, FOCAL, roll, pitch, yaw)
    dir_los = vec / np.linalg.norm(vec)
    dir_true = rel / np.linalg.norm(rel)
    assert np.allclose(dir_los, dir_true, atol=1e-9)
    assert abs(rng - np.linalg.norm(rel)) < 1e-9


def test_project_behind_camera():
    rel = np.array([-5.0, 0.0, 0.0])   # 正后方
    assert project_to_pixel(rel, 0, 0, 0, FOCAL, W / 2, H / 2) is None


def test_segment_min_distance():
    # 静止情形退化为端点距离
    p = np.zeros(3)
    assert abs(segment_min_distance(p, p, np.array([3., 0, 0]), np.array([3., 0, 0])) - 3.0) < 1e-12
    # 对穿：两质点交换位置，中途距离为 0
    d = segment_min_distance(np.array([0., 0, 0]), np.array([10., 0, 0]),
                             np.array([10., 0, 0]), np.array([0., 0, 0]))
    assert d < 1e-9
    # 端点处最近（t 限幅在 [0,1]）：拦截机 0→1 追静止目标(5,0,0)
    d = segment_min_distance(np.array([0., 0, 0]), np.array([1., 0, 0]),
                             np.array([5., 0, 0]), np.array([5., 0, 0]))
    assert abs(d - 4.0) < 1e-12


# ============================================================
#  PNG 老师（对照 png_calculate / handle_intercept 状态机行为）
# ============================================================

def make_teacher():
    return PNGTeacher(coast_steps=3, lost_steps=10)


def test_png_init_frame():
    """首帧：d_v_angle = LOS，输出 INIT，速度方向≈LOS 方向（含 ey 补偿）"""
    t = make_teacher()
    t.reset()
    # 目标在图像中心 → LOS 指北
    det = (W / 2 - 25, H / 2 - 25, 50, 50)
    cmd = t.step(det, 0, 0, 0, 0, 0, 0)
    assert cmd.phase == "INIT"
    assert abs(t.d_v_angle_v - t.los_v) < 1e-12
    assert abs(t.d_v_angle_z - t.los_z) < 1e-12
    # 静止时 speed = max(0+d_gain, speed_min) = speed_min
    assert abs(cmd.speed - t.speed_min) < 1e-12
    # 中心目标 ey=0 → 无垂直补偿，速度指北
    assert cmd.vx > 0 and abs(cmd.vy) < 1e-9 and abs(cmd.vz) < 1e-9


def test_png_update_rule():
    """第二帧 LOS 变化 > 阈值时：d_v_angle = k*dLOS + last_v_angle"""
    t = make_teacher()
    t.reset()
    det0 = (W / 2 - 25, H / 2 - 25, 50, 50)
    t.step(det0, 0, 0, 0, 0, 0, 0)
    v_angle_v0, v_angle_z0 = t.d_v_angle_v, t.d_v_angle_z
    los_v0, los_z0 = t.los_v, t.los_z

    # 目标右移 100px → LOS 变化超过 0.02 rad
    det1 = (W / 2 - 25 + 100, H / 2 - 25, 50, 50)
    t.step(det1, 0, 0, 0, 3, 0, 0)
    diff_v = wrap_pi(t.los_v - los_v0)
    diff_z = wrap_pi(t.los_z - los_z0)
    assert abs(diff_z) > t.los_diff_thresh
    assert abs(t.d_v_angle_v - (t.kv * diff_v + v_angle_v0)) < 1e-12
    assert abs(t.d_v_angle_z - (t.kz * diff_z + v_angle_z0)) < 1e-12


def test_png_small_change_no_update():
    """LOS 变化 < 0.02 rad：角度不更新（C++ line 333）"""
    t = make_teacher()
    t.reset()
    det0 = (W / 2 - 25, H / 2 - 25, 50, 50)
    t.step(det0, 0, 0, 0, 0, 0, 0)
    v_angle_z0 = t.d_v_angle_z
    det1 = (W / 2 - 25 + 5, H / 2 - 25, 50, 50)   # 5px ≈ 0.0036 rad
    t.step(det1, 0, 0, 0, 0, 0, 0)
    assert abs(t.d_v_angle_z - v_angle_z0) < 1e-12


def test_png_yaw_pd_and_clamp():
    """偏航 PD：d_yaw = k1*ex + k2*dex，限幅 ±yaw_rate_max"""
    t = make_teacher()
    t.reset()
    t.step((W / 2 - 25, H / 2 - 25, 50, 50), 0, 0, 0, 0, 0, 0)
    ex = 400.0
    cmd = t.step((W / 2 - 25 + ex, H / 2 - 25, 50, 50), 0, 0, 0, 0, 0, 0)
    expected = t.k1_yaw * ex + t.k2_yaw * (ex - 0.0)
    assert abs(cmd.yaw_rate - expected) < 1e-12
    # 大误差限幅
    t2 = PNGTeacher(k1_yaw=0.01)
    t2.reset()
    t2.step((W / 2, H / 2, 50, 50), 0, 0, 0, 0, 0, 0)
    cmd2 = t2.step((W - 100, H / 2, 50, 50), 0, 0, 0, 0, 0, 0)
    assert abs(cmd2.yaw_rate) <= t2.yaw_rate_max + 1e-12


def test_png_ey_compensation():
    """目标在画面上方（ey<0）→ vz 补偿为负（上升）"""
    t = make_teacher()
    t.reset()
    det = (W / 2 - 25, H / 2 - 25 - 300, 50, 50)   # ey = -300
    cmd = t.step(det, 0, 0, 0, 0, 0, 0)
    vz_no_comp = math.sin(t.d_v_angle_v) * cmd.speed
    assert cmd.vz < vz_no_comp   # 补偿使 vz 更小（上升）


def test_png_speed_ramp():
    """速度爬升：d_v = clamp(|V|+d_gain, [speed_min, speed_cmd])"""
    t = make_teacher()
    t.reset()
    det = (W / 2 - 25, H / 2 - 25, 50, 50)
    cmd = t.step(det, 0, 0, 0, 4.5, 0, 0)   # |V|=4.5, +1.0 → 5.5 → clamp 5.0
    assert abs(cmd.speed - t.speed_cmd) < 1e-12


def test_png_lost_coast_then_search():
    """丢失分级：< coast_steps 惯性续飞；>= coast_steps 减速旋转"""
    t = make_teacher()   # coast_steps=3
    t.reset()
    det = (W / 2 - 25, H / 2 - 25, 50, 50)
    cmd0 = t.step(det, 0, 0, 0, 0, 0, 0)
    c1 = t.step(None, 0, 0, 0, 0, 0, 0)
    c2 = t.step(None, 0, 0, 0, 0, 0, 0)
    assert c1.phase == "COAST" and c2.phase == "COAST"
    assert abs(c1.vx - cmd0.vx) < 1e-12   # 续飞保持缓存速度
    c3 = t.step(None, 0, 0, 0, 0, 0, 0)
    assert c3.phase == "LOST"
    assert abs(c3.vx) < 1e-12 and abs(c3.yaw_rate) > 0   # 制动+旋转搜索
    # 目标重现 → 恢复
    c4 = t.step(det, 0, 0, 0, 0, 0, 0)
    assert c4.phase == "INTERCEPT"


# ============================================================
#  特征与动作编解码
# ============================================================

def test_feature_dim_and_valid_flag():
    fb = FeatureBuilder(FOCAL, W, H)
    obs = fb.build((900, 500, 40, 30), 0, 0, 0, 1, 2, -1, -6.0, 0.05)
    assert obs.shape == (OBS_DIM,)
    assert obs[8] == 1.0 and obs[9] == 0.0
    obs2 = fb.build(None, 0, 0, 0, 1, 2, -1, -6.0, 0.05)
    assert obs2[8] == 0.0 and obs2[9] > 0.0
    # 丢失帧保持 last-known LOS
    assert abs(obs2[0] - obs[0]) < 1e-9
    # 速率项清零
    assert obs2[3] == 0.0 and obs2[4] == 0.0 and obs2[7] == 0.0


def test_action_roundtrip():
    """decode(encode(cmd)) ≈ cmd（非限幅区域）"""
    los_v, los_z = 0.05, math.pi / 2
    vx, vy, vz = angles_to_velocity(0.3, los_z - 0.4, 4.0)
    a = encode_action_from_velocity(vx, vy, vz, 0.5, los_v, los_z)
    assert np.all(np.abs(a) <= 1.0)
    vx2, vy2, vz2, yr2 = decode_action(a, los_v, los_z)
    assert abs(vx2 - vx) < 1e-6 and abs(vy2 - vy) < 1e-6 and abs(vz2 - vz) < 1e-6
    assert abs(yr2 - 0.5) < 1e-9


def test_zero_action_is_pure_pursuit():
    """零动作 = 沿 LOS 方向以中等速度飞行（PNG 归纳偏置）"""
    los_v, los_z = 0.1, 1.2
    vx, vy, vz, yr = decode_action(np.zeros(4), los_v, los_z,
                                   speed_min=2.0, speed_cmd=5.0)
    speed = math.sqrt(vx**2 + vy**2 + vz**2)
    assert abs(speed - 3.5) < 1e-9           # (2+5)/2
    assert abs(math.atan2(vz, math.hypot(vx, vy)) - los_v) < 1e-9
    assert abs(yr) < 1e-12
