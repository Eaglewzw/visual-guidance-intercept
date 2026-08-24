# Gazebo 高保真迁移设计

## 边界

`/home/verser/ros2_ws` 和 `/home/verser/PX4-Autopilot` 都作为只读依赖使用。工程不会
修改、重编译或覆盖其中的 C++、world 或模型。项目只新增公园 world、Python ROS
桥、训练环境和训练脚本。

ROS 2 Humble 的 Python 扩展属于 CPython 3.10，训练环境固定为 Python 3.12。为避免
把两套 ABI 混装到同一进程：

| 进程 | Python | 职责 |
|---|---|---|
| `ros_bridge.py` | Ubuntu `/usr/bin/python3` 3.10 | ROS 图像、PX4 odometry、offboard setpoint |
| PPO/评估 | `AeroIntercept` Conda Python 3.12 | RGB Actor、Critic、CUDA 优化、checkpoint |

两者使用权限为 0600 的 Unix socket 和带长度头的版本化消息。相机 payload 在每次
接收后复制，避免重复缓冲区别名。

## 旧接口映射

| 通用接口 | Gazebo 实现 |
|---|---|
| `FullFrameRenderer` | 不用于高保真路径；改为 Gazebo IMX214 `/camera/image` |
| 双帧 RGB | 原始 1920×1080 完整画面 letterbox 到 640×640 后按序号堆叠 |
| `decode_action` | 桥读取拦截机当前 yaw，在 Actor 外把 body FRD 转为 PX4 NED |
| 轻量动力学 | PX4 SITL 控制 x500，Gazebo physics 实际步进 |
| 目标运动 | 运行现有只读 `uav_target_sim` C++ 可执行程序 |
| 15 维 Critic | 两机 PX4 odometry 构造，仅由 PPO 训练容器读取 |
| 辅助标签 | odometry 真值生成，绝不拼接进 Actor 输入 |
| reset | PX4 position setpoint 物理返回 `[0,0,-6]` |

reset 和 launcher 初始悬停阶段会读取两机真值计算目标 bearing，并通过 PX4 yaw
setpoint 让机体物理转向；这段 look-at 只用于建立初始视觉条件。Actor 发出第一条
动作时桥立即关闭 look-at，之后 yaw 完全来自 Actor 的 `yaw_rate`，目标真值不会
进入 Actor。

## 相机协议

PX4 OakD-Lite 的 IMX214 为 1920×1080、水平 FOV 1.204 rad。统一变换：

1. 根据 ROS `encoding` 和 `step` 解码为 RGB uint8；
2. 完整帧按比例缩放到 640×360；
3. 上下各填充 140 个黑色像素，不做 bbox 裁剪；
4. 转为 CHW 并连续堆叠两帧，输出 `[N,2,3,640,640]`；
5. 模型内部转 float32、除以 255，再使用既有 ImageNet mean/std。

checkpoint 和导出 metadata 都记录 `full_frame_letterbox_v1`，部署端必须复用同一
变换。

`x500_depth` 的 OakD 成像轴相对 PX4 odometry 的 FRD yaw 存在固定 `-π/2` 安装
偏差。该符号由 Gazebo 动态 model pose 和真实相机画面共同校准，统一配置在
`gazebo.camera.mount_yaw_offset_rad`。FOV 奖励和 reset 对准都使用相机轴；Critic
和 body→NED 动作仍使用 PX4 机体系 yaw。

## 坐标与动作

Actor 输出归一化 `[forward,right,down,yaw_rate]`。前三维按整体向量模长限制，随后
使用 PX4 odometry 的当前 yaw 确定性转换为 NED：

```text
north = cos(yaw)*forward - sin(yaw)*right
east  = sin(yaw)*forward + cos(yaw)*right
down  = down
```

该转换只依赖自身 yaw，且位于 Actor 外部。Actor 的 forward 不接收状态参数；部署
导出只保存 Actor。

## 物理世界

`aerointercept_park.sdf` 使用 Gazebo ODE physics、传感器、IMU、气压计、磁力计和
NavSat system。场景使用项目内照片风格 PBR 草地/碎石材质、写实天空与阴影、
池塘、岩石、凉亭，以及 Gazebo Fuel 官方 Oak/Pine 网格。目标活动扇区保留开阔
草坪，避免植被长期遮挡小目标。双机仍由 PX4 自己从官方本地模型目录生成：
拦截机 `gz_x500_depth`，目标机 `gz_x500`；没有创建替代无人机或相机模型。

目标模型在 Gazebo 的 `(10,0,0)` 生成。PX4 当前 Gazebo bridge 的模型生成坐标与
local position 实测按 x→north、y→east 对齐，因此共同 NED 原点偏移为 `(10,0,0)`。
桥在构造两机物理距离时显式补偿该偏移，并另行应用前述相机安装 yaw 偏差。
