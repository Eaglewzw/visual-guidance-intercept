# AeroIntercept

### 面向自主无人机拦截的端到端视觉强化学习与仿真验证平台

<div align="center">

[![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-F58113.svg)](https://gazebosim.org/)
[![PX4](https://img.shields.io/badge/PX4-SITL-0057B8.svg)](https://px4.io/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-22314E.svg)](https://docs.ros.org/en/humble/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_PPO-EE4C2C.svg)](https://pytorch.org/)

</div>

> AeroIntercept 是面向自主无人机拦截任务的实验与训练平台。
> 系统以连续机载 RGB 图像作为策略输入，直接生成飞行控制指令，
> 并在 Gazebo 与 PX4 构成的物理仿真闭环中完成动态目标跟踪与拦截策略训练。

---

## 项目简介

AeroIntercept 面向从视觉感知到飞行控制的端到端学习。Actor 的输入仅包含完整相机图像，
不包含检测框、GPS、目标位置或其他仿真真值；训练阶段采用独立 Critic，部署阶段仅保留视觉策略。

主要功能包括：

- **纯视觉控制**：连续两帧 RGB 直接生成三维速度与偏航角速度。
- **高保真仿真**：使用 Gazebo Harmonic、PX4 SITL 和机载相机传感器链路。
- **端到端训练**：支持 CUDA PPO、辅助任务、TensorBoard 和 checkpoint 恢复。
- **动态目标**：支持圆周、正弦、随机游走和混合运动模式。
- **训练部署一致性**：训练、评估和导出采用相同的图像与动作协议。
- **双后端验证**：Gazebo 用于物理仿真验证，轻量二维环境用于算法回归测试。

---

## 仿真演示

<div align="center">
  <img src="results/gazebo_camera_30m.png" alt="AeroIntercept Gazebo 机载相机画面" width="78%">
  <p><i>拦截机机载相机视角：目标无人机位于前方写实公园场景中</i></p>
</div>

默认仿真配置包含两架 PX4 x500 无人机，初始距离约为 10 米。目标机按照设定轨迹运动；
reset 阶段将拦截机相机朝向目标，后续速度与偏航指令由视觉策略生成。

---

## 系统能力

| 能力 | 简介 |
|---|---|
| **视觉策略** | 共享 CNN、空间注意力与双帧时序 Transformer |
| **飞行控制** | PX4 Offboard 三维速度和偏航角速度控制 |
| **训练方法** | 非对称 Actor-Critic、PPO 与辅助监督 |
| **目标轨迹** | `circle`、`sinusoidal`、`random_walk`、`mixed` |
| **训练工具** | 基础验证、策略评估、训练日志、checkpoint 保存与恢复 |
| **场景环境** | 写实公园、PBR 地表、天空、植被和环境光照 |

---

## 快速开始



### 1. 启动仿真

在终端一启动写实公园场景、双机系统、PX4 SITL 和相机桥：

```bash
cd /home/verser/Python/AeroIntercept
bash aerointercept/gazebo/scripts/launch_gazebo.sh --mode circle --seed 31
```

### 2. 查看机载相机

在终端二显示 Actor 实际接收的相机图像：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AeroIntercept
cd /home/verser/Python/AeroIntercept
python -m aerointercept.gazebo.scripts.view_camera --display
```

### 3. 运行端到端训练

保持仿真系统运行，并在另一个终端中激活上述 Conda 环境后执行：

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --device cuda:0 --mode circle --logdir runs/demo
```
