# AeroIntercept

AeroIntercept 是视觉无人机追踪与拦截训练工程，目前保留三条路径：

| 路径 | 输入 | 仿真后端 |
|---|---|---|
| 学习制导 | bbox 构造的 15 维特征 | 轻量二维环境 |
| 端到端快速回归 | 连续两张完整 RGB | 轻量二维环境 |
| 端到端高保真训练 | 连续两张 640×640 RGB | Gazebo Harmonic + PX4 SITL |

高保真主线使用现有 `/home/verser/ros2_ws` 中的 ROS 2 Humble 接口和目标运动 C++
可执行程序，但不会修改该工作区的任何 C++ 源码。无人机直接复用 PX4 的
`x500_depth + OakD-Lite IMX214` 和 `x500` 资产，不创建替代机体或相机模型。

## Gazebo 数据流

```text
Gazebo 1920×1080 IMX214 RGB
  -> ros_gz_bridge
  -> 完整帧 RGB letterbox（640×360，上下各填充 140）
  -> uint8 [N,2,3,640,640]
  -> 仅图像 Actor
  -> [forward,right,down,yaw_rate] ∈ [-1,1]
  -> Actor 外部使用 PX4 当前 yaw 做 body -> NED
  -> PX4 offboard velocity setpoint
  -> Gazebo 物理步进

PX4 odometry truth -> 15 维 Critic / auxiliary labels（仅训练）
```

训练、评估、相机查看和导出 checkpoint 都使用
`full_frame_letterbox_v1`。该变换不裁剪原始画面；Actor 不读取 bbox、LOS、GPS、
目标位置、目标速度、当前 yaw 或其他仿真真值。

ROS Humble 的 `rclpy` 是 CPython 3.10 扩展，而 AeroIntercept Conda 环境是
Python 3.12。因此 ROS/PX4 桥使用系统 `/usr/bin/python3`，CUDA PPO 保持运行在现有
`AeroIntercept` Conda 环境，二者通过本机 Unix socket 传输。

详细设计见 [Gazebo 迁移设计](docs/gazebo_migration.md)，完整命令见
[Gazebo 训练指南](docs/gazebo_training.md)。

## 最快查看仿真和训练

终端一启动带界面的写实公园、双机、相机桥和现有 C++ 目标控制器。首次运行会从
Gazebo Fuel 下载官方 Oak/Pine 两个植被资产，之后使用本地缓存：

```bash
bash /home/verser/Python/AeroIntercept/aerointercept/gazebo/scripts/launch_gazebo.sh \
  --mode circle --seed 31
```

终端二启动小规模 CUDA PPO：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AeroIntercept
cd /home/verser/Python/AeroIntercept
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 \
  --checkpoint-interval 256 --logdir runs/gazebo_e2e_gui --mode circle
```

参数名是 `--total-steps`。launcher 的悬停阶段会让 PX4 机体持续转向目标；要看
Actor 实际收到的 640×640 画面：

```bash
python -m aerointercept.gazebo.scripts.view_camera --display
```

按 `q` 或 Esc 关闭相机窗口；终端一按 Ctrl+C 会停止该脚本自己启动的进程，不会
使用 `killall`，也不会删除 PX4 或 ROS 工作区日志。

保存一张经过“目标起飞、物理转向、近距离入镜”检查的无界面预览：

```bash
python -m aerointercept.gazebo.scripts.smoke_camera \
  --launch --headless --visual-max-distance 8 \
  --output results/gazebo_camera_realistic_aligned.png
```

## 无界面冒烟与训练

一条命令自动管理单个 Gazebo 栈：

```bash
python -m aerointercept.gazebo.scripts.smoke_camera \
  --launch --headless --frames 8 --output results/gazebo_camera.png

python -m aerointercept.gazebo.scripts.smoke_env \
  --launch --headless --steps 32 --mode circle --seed 31

python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 --checkpoint-interval 256 \
  --logdir runs/gazebo_e2e_smoke --mode mixed
```

恢复训练、评估和 TensorBoard：

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 \
  --checkpoint runs/gazebo_e2e_smoke/checkpoints/last.pt --resume \
  --logdir runs/gazebo_e2e_smoke --mode mixed

python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --episodes 10 --mode circle \
  --checkpoint runs/gazebo_e2e_smoke/checkpoints/best.pt \
  --output results/gazebo_e2e_eval.json

tensorboard --logdir runs/gazebo_e2e_smoke --port 6006
```

## 测试

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AeroIntercept
cd /home/verser/Python/AeroIntercept
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

普通 pytest 覆盖协议和旧环境，但不能代替真实 Gazebo 验收。真实验收必须运行前述
`smoke_camera`、`smoke_env` 和 512-step CUDA PPO。

本机最近一次验收（2026-08-24，RTX 5070 Ti）：

| 项目 | 结果 |
|---|---|
| Gazebo / PX4 / ROS 相机 | 真实启动并通过，原始 `rgb8` 1920×1080 |
| Letterbox 输出 | RGB uint8 `[N,2,3,640,640]`，上下各 140 px |
| 写实相机验收 | 目标距离 5.43 m，6 帧动态 RGB，目标真实入镜；预览见 `results/gazebo_camera_realistic_aligned.png` |
| 光轴校准 | OakD/PX4 固定 yaw offset `-π/2`；reset 后误差 0.013 rad，8-step 保持可见 |
| 物理 rollout | reset、目标起飞、8-step 可见 rollout 通过，约 30.10 step/s |
| 512-step CUDA PPO | 32 次更新，Actor 参数变化 `6.73e-3` |
| 显存 | 整机峰值 3284 MB，PyTorch 峰值 1918 MB |
| 训练吞吐 | 相机 7.49 FPS，端到端训练 7.26 step/s |
| checkpoint 恢复 | global step 512→528，optimizer/RNG 恢复成功 |
| 普通回归 | 53 passed |

随机初始化策略在该 512-step 工程冒烟中的 hit rate 为 0；这证明的是仿真和优化闭环，
不代表策略收敛。

## 保留的轻量工作流

```bash
python -m aerointercept.training.collect_e2e_data --episodes 1000 --out data/e2e_bc
python -m aerointercept.training.train_e2e_bc --data data/e2e_bc --out checkpoints/e2e_bc.pt
python -m aerointercept.training.train_e2e_ppo \
  --bc-init checkpoints/e2e_bc.pt --out checkpoints/e2e_rl.pt --logdir runs/e2e_ppo
python -m aerointercept.evaluation.eval_e2e \
  --policy checkpoints/e2e_rl.pt --episodes 200 --out results/e2e_rl.csv
```

## 目录

```text
AeroIntercept/
├── assets/gazebo/worlds/             # 写实公园 world；不复制 PX4 机体资产
├── assets/gazebo/models/             # 项目内 PBR 草地/碎石地表
├── configs/gazebo_e2e.yaml           # Gazebo/相机/奖励/PPO 配置
├── configs/gazebo_camera_bridge.yaml # 原始 IMX214 ROS 桥话题
├── aerointercept/gazebo/
│   ├── ros_bridge.py                 # 系统 Python ROS/PX4 进程
│   ├── client.py, protocol.py        # Unix socket 边界
│   ├── camera.py                     # 全帧 RGB letterbox
│   ├── environment.py                # 物理 Gazebo 训练环境
│   └── scripts/                      # 启动、冒烟、训练、评估、看相机
├── aerointercept/end_to_end/         # 通用 Actor、分布、动作和运行时
├── aerointercept/training/           # 保留的轻量数据/BC/PPO
└── tests/
```

## 当前限制

- 现有 C++ 目标节点原生支持 `circle`、`sinusoidal`、`random_walk`；`mixed` 在启动时
  按 seed 选择其中一种。由于不能修改该 C++，当前没有 `hover_escape`。
- 自动 `--launch` 当前只管理一个 Gazebo/PX4 世界。多环境接口要求每个环境分别使用
  隔离的 Gazebo partition、ROS domain、XRCE 端口和 Unix socket，不能把一个世界
  伪装为多个并行物理环境。
- episode reset 使用 PX4 position setpoint 物理返航，不瞬移模型。严重坠毁并自动
  disarm 后可能需要重启该 Gazebo 栈。
- 写实公园布局目前固定，尚未接入逐 episode 材质、光照和相机参数随机化；官方
  Oak/Pine 首次运行需要联网，之后由 Gazebo Fuel 缓存复用。
- 512-step 只验证工程闭环和参数更新，不代表策略已经收敛。

后续计划见 [ROADMAP.md](ROADMAP.md)。
