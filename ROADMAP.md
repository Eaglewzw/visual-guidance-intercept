# AeroIntercept 路线图

更新时间：2026-08-24

## 当前主线

高保真训练后端迁移为 Gazebo Harmonic 8、PX4 SITL 和 ROS 2 Humble。工程复用 PX4
现有 `x500_depth/OakD-Lite` 与 `x500` 资产，以及只读 `ros2_ws` 中的目标运动 C++
可执行程序。旧二维训练路径继续保留用于快速回归。

```text
Gazebo camera -> full-frame letterbox -> RGB Actor -> body velocity/yaw rate
                                                    -> PX4 offboard -> physics
PX4 odometry -> privileged Critic and labels only during training
```

## 实现状态

| 里程碑 | 状态 |
|---|---|
| 移除旧高保真后端包、配置、测试和专用文档 | 完成 |
| ROS Python 3.10 与 Conda Python 3.12 进程隔离 | 完成 |
| 1920×1080 完整 RGB 到 640×640 letterbox 协议 | 完成 |
| PX4 body FRD 动作到 NED 限幅和 offboard 桥 | 完成 |
| 15 维非对称 Critic 与辅助标签 | 完成 |
| PBR 写实公园、官方 Fuel 植被、双机 launcher | 真实验收完成 |
| OakD 光轴 `-π/2` 校准、reset 物理 look-at、目标真实入镜 | 完成 |
| CUDA PPO、checkpoint 保存/恢复、评估入口 | 512-step 与恢复验收完成 |
| 所有普通回归测试 | 53 项通过 |
| 多 Gazebo 实例并行 rollout | 接口已定义，隔离 launcher 待实现 |
| 材质、光照、相机和动力学域随机化 | 待实现 |
| 1M+ step、多 seed 收敛实验 | 待执行 |

## 下一阶段门槛

1. 为并行训练分配独立 Gazebo partition、`ROS_DOMAIN_ID`、Micro XRCE UDP 端口、
   PX4 instance/rootfs 和 Unix socket，再验证 2 个实例；不共享相机缓冲区。
2. 在不修改现有 C++ 的前提下增加可配置 target supervisor，补齐
   `hover_escape` 和逐 episode `mixed`。
3. 加入公园布局、材质、光照、曝光、噪声、控制延迟、质量和惯性随机化。
4. 完成多 seed 正式训练；实机阶段必须保留 watchdog、人工接管、地理围栏和 PX4
   failsafe，禁止把目标真值反馈给 Actor。

运行方式见 [Gazebo 训练指南](docs/gazebo_training.md)。
