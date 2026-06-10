"""PNG 老师：vision_png_control.cpp 制导算法的 Python 移植

三处用途共用同一实现：
  1. Gym 内 BC 数据采集的专家策略
  2. Gym 评估的 PNG 基线
  3. ROS2 部署节点的 watchdog 回退控制器

移植对照（vision_png_control.cpp）：
  png_calculate()      line 266-393：LOS 计算 + PNG 角度更新 + 偏航 PD
  handle_intercept()   line 447-509：速度合成 + 垂直补偿 + 丢失分级
  handle_track_lost()  line 515-527：减速旋转搜索
"""
import math
from dataclasses import dataclass, field

from .geometry import pixel_to_los, angles_to_velocity, wrap_pi


@dataclass
class PNGCommand:
    """一次制导输出"""
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    phase: str = "INTERCEPT"      # INTERCEPT / COAST / LOST / INIT
    # 调试量（与 /vpng_data 对应）
    v_angle_v: float = 0.0
    v_angle_z: float = 0.0
    speed: float = 0.0


@dataclass
class PNGTeacher:
    """有状态 PNG 控制器，20Hz 调用 step() 一次"""
    # ---- 参数（默认值 = uav_vision_png/config/params.yaml 调好的那组）----
    focal: float = 1397.2
    image_width: int = 1920
    image_height: int = 1080
    kv: float = 4.0
    kz: float = 4.0
    speed_cmd: float = 5.0
    speed_min: float = 2.0
    d_gain: float = 1.0
    k1_yaw: float = 0.0005
    k2_yaw: float = 0.0002
    yaw_rate_max: float = 1.0
    k_ey: float = 0.012
    vz_ey_max: float = 3.0
    los_diff_thresh: float = 0.02
    elev_clamp: float = math.pi / 4.0
    coast_steps: int = 10
    lost_steps: int = 30
    search_yaw_rate: float = 0.2

    # ---- 内部状态 ----
    initialized: bool = field(default=False, init=False)
    los_v: float = field(default=0.0, init=False)
    los_z: float = field(default=0.0, init=False)
    last_los_v: float = field(default=0.0, init=False)
    last_los_z: float = field(default=0.0, init=False)
    d_v_angle_v: float = field(default=0.0, init=False)
    d_v_angle_z: float = field(default=0.0, init=False)
    last_v_angle_v: float = field(default=0.0, init=False)
    last_v_angle_z: float = field(default=0.0, init=False)
    d_yaw: float = field(default=0.0, init=False)
    last_ex: float = field(default=0.0, init=False)
    lost_frames: int = field(default=0, init=False)
    coast_cmd: PNGCommand = field(default_factory=PNGCommand, init=False)

    @classmethod
    def from_config(cls, cfg) -> "PNGTeacher":
        """从 DotDict 配置构造（configs/default.yaml 的 camera + png 节）"""
        return cls(
            focal=cfg.camera.focal_length,
            image_width=cfg.camera.image_width,
            image_height=cfg.camera.image_height,
            kv=cfg.png.kv, kz=cfg.png.kz,
            speed_cmd=cfg.png.speed_cmd, speed_min=cfg.png.speed_min,
            d_gain=cfg.png.d_gain,
            k1_yaw=cfg.png.k1_yaw, k2_yaw=cfg.png.k2_yaw,
            yaw_rate_max=cfg.png.yaw_rate_max,
            k_ey=cfg.png.k_ey, vz_ey_max=cfg.png.vz_ey_max,
            los_diff_thresh=cfg.png.los_diff_thresh,
            elev_clamp=cfg.png.elev_clamp,
            coast_steps=cfg.png.coast_steps, lost_steps=cfg.png.lost_steps,
            search_yaw_rate=cfg.png.search_yaw_rate,
        )

    def reset(self):
        self.initialized = False
        self.los_v = self.los_z = 0.0
        self.last_los_v = self.last_los_z = 0.0
        self.d_v_angle_v = self.d_v_angle_z = 0.0
        self.last_v_angle_v = self.last_v_angle_z = 0.0
        self.d_yaw = 0.0
        self.last_ex = 0.0
        self.lost_frames = 0
        self.coast_cmd = PNGCommand()

    def step(self, det, roll, pitch, yaw, vx, vy, vz) -> PNGCommand:
        """单步制导

        det: (x, y, w, h) bbox 左上角+宽高（像素），None 表示本帧丢失
        roll/pitch/yaw: 自身姿态 (rad)；vx/vy/vz: 自身 NED 速度 (m/s)
        """
        # ---------- 丢失分级（handle_intercept line 453-469）----------
        if det is None:
            self.lost_frames += 1
            if self.lost_frames < self.coast_steps:
                cmd = PNGCommand(**vars(self.coast_cmd))
                cmd.phase = "COAST"
                return cmd
            # 长时间丢失：制动 + 旋转搜索（handle_track_lost line 519-521）
            search_yaw = self.d_yaw if abs(self.d_yaw) > 0.01 else self.search_yaw_rate
            return PNGCommand(yaw_rate=search_yaw, phase="LOST")
        self.lost_frames = 0

        # ---------- Step 1: 像素误差（png_calculate line 272-277）----------
        x, y, w, h = det
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        ex = (x + w / 2.0) - cx
        ey = (y + h / 2.0) - cy

        # ---------- Step 2-4: LOS 角（line 279-303）----------
        los_v, los_z, _ = pixel_to_los(ex, ey, self.focal, roll, pitch, yaw)
        self.los_v, self.los_z = los_v, los_z

        # ---------- Step 5: PNG 角度更新（line 305-356）----------
        if not self.initialized:
            # 首帧：用 LOS 方向初始化期望速度方向后直接返回（C++ line 309-321）
            self.d_v_angle_v = los_v
            self.d_v_angle_z = los_z
            self.last_los_v, self.last_los_z = los_v, los_z
            self.last_v_angle_v, self.last_v_angle_z = los_v, los_z
            self.initialized = True
            cmd = self._intercept_cmd(vx, vy, vz, ey)
            cmd.phase = "INIT"
            self.coast_cmd = cmd
            return cmd

        diff_v = wrap_pi(los_v - self.last_los_v)
        diff_z = wrap_pi(los_z - self.last_los_z)

        if abs(diff_v) > self.los_diff_thresh or abs(diff_z) > self.los_diff_thresh:
            self.d_v_angle_v = self.kv * diff_v + self.last_v_angle_v
            self.d_v_angle_z = self.kz * diff_z + self.last_v_angle_z
            # 仰角限幅 ±π/4（line 341）
            self.d_v_angle_v = max(-self.elev_clamp, min(self.elev_clamp, self.d_v_angle_v))
            self.last_v_angle_v = self.d_v_angle_v
            self.last_v_angle_z = self.d_v_angle_z
            self.last_los_v, self.last_los_z = los_v, los_z

        # ---------- Step 6: 偏航 PD（line 358-368，每个有效帧都执行）----------
        d_ex = ex - self.last_ex
        self.d_yaw = self.k1_yaw * ex + self.k2_yaw * d_ex
        self.d_yaw = max(-self.yaw_rate_max, min(self.yaw_rate_max, self.d_yaw))
        self.last_ex = ex

        # ---------- 速度合成（handle_intercept line 471-501）----------
        cmd = self._intercept_cmd(vx, vy, vz, ey)
        self.coast_cmd = cmd
        return cmd

    def _intercept_cmd(self, vx, vy, vz, ey) -> PNGCommand:
        """速度爬升 + 角度合成 + 垂直视场补偿"""
        v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        d_v = min(v_norm + self.d_gain, self.speed_cmd)
        d_v = max(d_v, self.speed_min)

        cvx, cvy, cvz = angles_to_velocity(self.d_v_angle_v, self.d_v_angle_z, d_v)

        # ey 垂直视场补偿（line 482-492）
        vz_ey = self.k_ey * ey
        vz_ey = max(-self.vz_ey_max, min(self.vz_ey_max, vz_ey))
        cvz += vz_ey

        return PNGCommand(
            vx=cvx, vy=cvy, vz=cvz, yaw_rate=self.d_yaw,
            phase="INTERCEPT",
            v_angle_v=self.d_v_angle_v, v_angle_z=self.d_v_angle_z, speed=d_v,
        )
