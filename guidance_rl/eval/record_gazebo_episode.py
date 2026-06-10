"""Gazebo 数据录制节点 —— 校核 Gym 与 Gazebo 的 sim-to-sim 差距

录制真实闭环数据（任一制导节点运行时均可），用于：
  1. 校准 Gym 动力学参数（tau_v：对比速度指令与实际速度的阶跃响应）
  2. 校核检测噪声/丢失率模型（对比 bbox 统计特性）

录制内容（20Hz 采样）：
  det        : bbox (x,y,w,h)，丢失为 (-1,-1,-1,-1)
  att        : roll/pitch/yaw
  vel        : 自身 NED 速度（实际）
  cmd_vel    : 速度指令（/px4_1/fmu/in/trajectory_setpoint）
  self_pos   : 自身 GPS→NED（统计坐标系）
  target_pos : 目标 GPS→NED

用法（仿真运行中）:
  python -m guidance_rl.eval.record_gazebo_episode --out data/gazebo_ep1.npz \\
      --duration 60

校准示例:
  d = np.load("data/gazebo_ep1.npz")
  # 用 cmd_vel 与 vel 的一阶拟合估计 tau_v，更新 configs/default.yaml
"""
import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import (TrajectorySetpoint, VehicleOdometry,
                          VehicleLocalPosition, SensorGps)
from uav_common_msg.msg import RectMsg

from guidance_rl.geometry import quat_to_euler

_M_PER_DEG_LAT = 111320.0


class Recorder(Node):
    def __init__(self, duration: float):
        super().__init__("gazebo_episode_recorder")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)

        self.det = (-1, -1, -1, -1)
        self.att = (0.0, 0.0, 0.0)
        self.vel = (0.0, 0.0, 0.0)
        self.cmd_vel = (float("nan"),) * 3
        self.local_z = 0.0
        self.origin = None
        self.self_pos = (0.0, 0.0, 0.0)
        self.target_pos = (0.0, 0.0, 0.0)

        self.create_subscription(RectMsg, "/camera_detect_result",
                                 self._det_cb, 10)
        self.create_subscription(VehicleOdometry,
                                 "/px4_1/fmu/out/vehicle_odometry",
                                 self._odom_cb, qos)
        self.create_subscription(VehicleLocalPosition,
                                 "/px4_1/fmu/out/vehicle_local_position",
                                 self._lp_cb, qos)
        self.create_subscription(TrajectorySetpoint,
                                 "/px4_1/fmu/in/trajectory_setpoint",
                                 self._cmd_cb, 10)
        self.create_subscription(SensorGps,
                                 "/px4_1/fmu/out/vehicle_gps_position",
                                 self._gps1_cb, qos)
        self.create_subscription(SensorGps,
                                 "/px4_2/fmu/out/vehicle_gps_position",
                                 self._gps2_cb, qos)

        self.rows = []
        self.max_rows = int(duration / 0.05)
        self.timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(f"录制中（{duration:.0f}s @ 20Hz）...")

    def _det_cb(self, m):
        self.det = (m.x, m.y, m.width, m.height)

    def _odom_cb(self, m):
        self.att = quat_to_euler(m.q[0], m.q[1], m.q[2], m.q[3])

    def _lp_cb(self, m):
        self.vel = (m.vx, m.vy, m.vz)
        self.local_z = m.z

    def _cmd_cb(self, m):
        self.cmd_vel = tuple(m.velocity)

    def _gps_ned(self, m):
        lat0, lon0, alt0 = self.origin
        return ((m.latitude_deg - lat0) * _M_PER_DEG_LAT,
                (m.longitude_deg - lon0) * _M_PER_DEG_LAT
                * math.cos(math.radians(lat0)),
                -(m.altitude_msl_m - alt0))

    def _gps1_cb(self, m):
        if self.origin is None:
            self.origin = (m.latitude_deg, m.longitude_deg, m.altitude_msl_m)
        self.self_pos = self._gps_ned(m)

    def _gps2_cb(self, m):
        if self.origin is None:
            return
        self.target_pos = self._gps_ned(m)

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        self.rows.append((
            t, *self.det, *self.att, *self.vel, *self.cmd_vel,
            *self.self_pos, *self.target_pos, self.local_z))
        if len(self.rows) >= self.max_rows:
            self.timer.cancel()
            raise SystemExit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/gazebo_episode.npz")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    rclpy.init()
    node = Recorder(args.duration)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass

    arr = np.array(node.rows, dtype=np.float64)
    cols = ("t,det_x,det_y,det_w,det_h,roll,pitch,yaw,vx,vy,vz,"
            "cmd_vx,cmd_vy,cmd_vz,self_n,self_e,self_d,"
            "tgt_n,tgt_e,tgt_d,local_z")
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, data=arr, columns=cols)
    print(f"已保存 {len(arr)} 帧 → {args.out}")
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
