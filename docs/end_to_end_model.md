# 端到端模型设计

本文记录 AeroIntercept 当前 Gazebo 训练主线的模型和学习协议。运行命令见
[Gazebo 训练指南](gazebo_training.md)，仿真与进程边界见
[Gazebo 迁移设计](gazebo_migration.md)。

## 观测边界

Actor 唯一输入为连续两张完整 RGB 图像：

```text
[num_envs, 2, 3, 640, 640] uint8
```

Gazebo 相机原始画面经过等比例 letterbox，不裁剪目标。模型内部转换为 float32，
除以 255 后使用 ImageNet mean/std 归一化。Actor 不接收 bbox、LOS、GPS、目标真值、
飞行器状态或自身全局 yaw。

仿真真值只用于奖励、训练标签和独立 Critic。部署导出仅包含 Actor。

## Actor

```text
前一帧 RGB ─→ 共享轻量 CNN ─→ 空间注意力 ─→ 128 维帧 token ─┐
                                                               ├→ 时序 Transformer
当前帧 RGB ─→ 共享轻量 CNN ─→ 空间注意力 ─→ 128 维帧 token ─┘
                                                                        │
                         ┌──────────────────────┬───────────────────────┼──────────┐
                         ↓                      ↓                       ↓          ↓
                    四维动作预测          未来位置预测             碰撞风险    置信度
```

共享视觉编码器使用深度可分离残差块，通道为 `[24, 40, 64, 96]`，总下采样倍数为 8。
640×640 输入得到 96×80×80 特征图。一个 1×1 卷积生成空间注意力，并在 6400 个位置
上做 softmax 加权池化。每帧最终压缩为一个 128 维 token。

时序部分使用两层 Transformer Encoder：embedding 128、4 个 attention heads、FFN 256。
Transformer 只处理前后两帧 token，不直接处理 80×80 空间 token。

Actor 包含四个输出头：

- 动作：4 维 tanh 输出；
- `future_position`：3 维归一化机体系未来相对位置；
- `collision_risk`：二分类 logit；
- `confidence`：目标可见性 logit。

后三项使用仿真真值作为训练监督，但不会反馈为 Actor 输入。当前 Actor 参数量为
336,590。

## 动作协议

Actor 输出：

```text
[forward_velocity, right_velocity, down_velocity, yaw_rate] ∈ [-1, 1]
```

前三维在 Actor 外部缩放并进行整体向量模长限制。控制层读取拦截机当前 yaw，将机体系
速度确定性旋转为 PX4 NED 速度；yaw 不进入神经网络。当前速度上限为 8 m/s，偏航角
速度上限为 1 rad/s。

PPO 使用带 tanh 修正的 Gaussian 分布，动作 log probability 包含 squash Jacobian。

## 非对称 Critic

Critic 是独立的 `15 → 256 → 256 → 1` MLP，只在训练时使用。15 维特权状态由以下
内容组成：

| 内容 | 维度 |
|---|---:|
| 机体系相对位置 | 3 |
| 机体系相对速度 | 3 |
| 机体系目标速度 | 3 |
| 机体系拦截机速度 | 3 |
| 高度、双方距离、目标可见性 | 3 |

训练容器分别调用 `actor(frames)` 和 `critic(privileged)`；部署导出只保存 Actor。

## 奖励函数

单步奖励为：

```text
1.0 × 距离缩短
+50  × 命中
-0.01 × 时间
-0.1 × 视场中心误差
-0.2 × 目标不可见
-0.05 × 动作变化平方和
-20 × 触地
-30 × 非法状态
-20 × 越界
-10 × 超时
```

命中使用一个控制步内两机运动线段的最小物理距离判定，阈值为 0.8 m，不使用图像
标签。连续失视 30 step、触地、非法状态或越界会终止 episode；600 step 超时会截断
episode。

当前双机 reset 距离为 10±1 m。circle 目标的固定轨迹半径为 5 m，因此运动开始后
实际距离会变化，并不会锁定在 10 m。

## PPO 损失

```text
loss = policy_loss
     + 0.5 × value_loss
     - 0.001 × entropy
     + 0.1 × auxiliary_loss
```

辅助损失由未来位置 MSE、碰撞风险 BCE 和置信度 BCE 构成，内部权重分别为
`1.0 / 0.5 / 0.5`。训练使用 PPO clipping、GAE、梯度裁剪和 KL early stopping。

图像 rollout 以 CPU uint8 保存，minibatch 才传入 CUDA。Gazebo 默认冒烟配置为单
环境、16-step rollout、512 total steps 和 encoder chunk size 4。

## Checkpoint 与部署

Gazebo checkpoint 保存模型、优化器、global step、配置、随机状态和运行版本信息。
使用 `--resume` 可以恢复完整训练状态；只传 `--checkpoint` 则仅加载模型权重。

部署导出为 TorchScript Actor，并记录图像、颜色顺序、预处理和动作协议。导出模型
保留动作与辅助输出，不包含 Critic。
