# guidance_rl — 视觉拦截学习制导律（阶段一）

> 端到端拦截路线图的第一阶段：**感知冻结、决策学习**。
> 保留 YOLO+LightTrack 感知层（`uav_vision_dectect`），用 GRU 策略替换
> PNG 制导律（`uav_vision_png`），在轻量运动学 Gym 中以 BC+PPO 训练，
> 与 PNG 基线在同一指标体系下定量对比。

## 1. 总体架构

```
训练侧（本工程）                                  部署侧（ros2_ws/uav_rl_guidance）
┌──────────────────────────────────────┐         ┌──────────────────────────────────┐
│ 轻量 Gym（20Hz，~11k steps/s）        │         │ rl_guidance_node.py (rclpy)      │
│  拦截机: 一阶速度响应 + 前倾姿态        │ export  │  状态机: TAKE_OFF/SEARCHING/      │
│  目标机: 移植 uav_target_sim 4 模式    │ ──────→ │   INTERCEPT/TRACK_LOST/DONE      │
│  相机:   针孔投影+噪声/漏检/延迟        │policy.pt│  INTERCEPT: 策略推理 @20Hz        │
│ PNG老师(移植C++) → BC → PPO(非对称AC)  │         │  watchdog → 回退内置 PNG          │
└──────────────────────────────────────┘         └──────────────────────────────────┘
```

**为什么这样设计**
- 输入是 bbox 级抽象量（非像素），Gazebo 训练的策略换真实相机仍可用，sim2real 风险最小
- GRU 时序记忆替代 PNG 的 LOS 差分，可从 bbox 序列隐式学习目标机动模式与前置量
- 动作空间带 PNG 归纳偏置（零动作=纯追踪），BC 初始化即接近可用
- 非对称 Actor-Critic：Critic 用仿真真值（对应真实系统"统计专用"GPS），Actor 只看视觉特征

## 2. 接口定义

### 观测（15 维，`guidance_rl/features.py`，FEATURE_VERSION=1）

| # | 特征 | 说明 |
|---|---|---|
| 0-2 | los_v/(π/4), sin/cos(los_z) | NED 系 LOS 角（与 C++ `png_calculate` 同公式）|
| 3-4 | dlos_v/dt, dlos_z/dt | LOS 角速率（PNG 的核心输入）|
| 5-7 | ln w, ln h, dlnw/dt | bbox 对数尺寸 + 变化率（距离/接近速度代理）|
| 8-9 | valid, age | 检测有效标志 / 丢失时长（策略由此学惯性续飞）|
| 10-14 | vx,vy,vz,\|V\|,高度 | 自身状态 |

丢失帧：保持 last-known LOS/尺寸，速率清零，valid=0。

### 动作（4 维 tanh，`features.decode_action`）

| # | 解码 | 范围 |
|---|---|---|
| a0 | v_angle_v = clamp(los_v + a0·1.2, ±π/4) | 仰角前置偏移 |
| a1 | v_angle_z = los_z + a1·1.2 | 方位前置偏移 |
| a2 | speed = 2 + 3·(a2+1)/2 | 2~5 m/s |
| a3 | yaw_rate = a3·1.0 | ±1 rad/s |

速度合成与 `handle_intercept()`（vision_png_control.cpp:478-480）完全一致。
`dv_angle_max=1.2`：实测 0.8 会截断 PNG 对圆周目标的累积前置角。

### 奖励（`configs/default.yaml: env.reward`）

`r = 1.0·接近速率/speed_cmd + 50·命中 − 0.1·FOV偏离² − 0.2·丢失帧
   − 0.05·Δ动作² − 0.01·时间 − 20·丢失终止 − 10·超时`

终止：命中（步内线段最小距离 < 0.8m）/ 连续丢失 30 步 / 触地 / 1200 步超时。

## 3. 与现有 C++ 实现的对齐关系

| 移植内容 | 来源 | 移植到 |
|---|---|---|
| 像素→LOS、速度合成 | vision_png_control.cpp:266-393 | geometry.py（26 个 pytest 用例锁定）|
| PNG 全部细节（0.02 阈值/±π/4 限幅/yaw PD/速度爬升/k_ey 补偿/丢失分级）| 同上 + handle_intercept | png_teacher.py |
| 目标 3 种运动 + 边界回复力 | uav_target_sim.cpp:129-219 | envs/target_motion.py（+hover_escape）|
| 相机 1397.2px/1920×1080 | params.yaml / SDF | configs/default.yaml |
| 状态机/话题/CSV 格式 | vision_png_control.cpp | uav_rl_guidance/rl_guidance_node.py |

注意：C++ 圆周角速度以构造函数 `angular_speed_=0.05`（/0.1s = 0.5 rad/s）为准，
hpp 中的 0.0025 是被覆盖的死值。

## 4. 训练流程（依次执行）

```bash
conda activate guidance_rl     # 克隆自 YOLO-LT，已含 torch 2.12+cu132
cd /home/verser/Python/guidance_rl

# 0) 一致性测试（应 26 passed）
python -m pytest tests/ -q

# 1) BC 数据采集（PNG 老师，~2000 集，约 5 分钟）
python -m guidance_rl.train.collect_bc_data --episodes 2000 --out data/bc_dataset.npz

# 2) 行为克隆（约 10 分钟 GPU）
python -m guidance_rl.train.train_bc --data data/bc_dataset.npz \
    --out checkpoints/bc_policy.pt

# 3) PPO 微调（默认 5M 步；监控 rollout/hit_rate）
python -m guidance_rl.train.train_ppo --bc-init checkpoints/bc_policy.pt \
    --out checkpoints/rl_policy.pt --logdir runs/ppo
tensorboard --logdir runs/

# 4) 三方评估（PNG vs BC vs RL × 4 运动模式）
python -m guidance_rl.eval.eval_gym --policy png --episodes 200 --out results/png.csv
python -m guidance_rl.eval.eval_gym --policy checkpoints/bc_policy.pt --episodes 200 --out results/bc.csv
python -m guidance_rl.eval.eval_gym --policy checkpoints/rl_policy.pt --episodes 200 --out results/rl.csv --save-traj 3
python -m guidance_rl.eval.plot_results --csv results/png.csv results/bc.csv results/rl.csv \
    --labels PNG BC RL --traj results/rl_traj.npz --out results/plots

# 5) 导出部署（阶段一）
python -m guidance_rl.export --ckpt checkpoints/rl_policy.pt \
    --out /home/verser/ros2_ws/src/uav_rl_guidance/models/policy.pt

# ---------- 阶段二 ----------
# 6) BC 数据采集（500 集，PNG 老师闭环，JPEG 图像 + npz 索引）
python -m guidance_rl.train.collect_bc_data_v2 --episodes 500 --out data/bc_v2

# 7) BC 训练（动作 MSE + 辅助 bbox/conf 联合优化，MobileNetV3 预训练）
python -m guidance_rl.train.train_bc_v2 --data data/bc_v2 \
    --out checkpoints/bc_policy_v2.pt

# 8) PPO 微调（8 envs × 128 steps，BC 热启动）
python -m guidance_rl.train.train_ppo_v2 --bc-init checkpoints/bc_policy_v2.pt \
    --out checkpoints/rl_policy_v2.pt --logdir runs/ppo_v2

# 9) 三方对比（PNG vs V1 vs V2 × 4 运动模式）
python -m guidance_rl.eval.eval_gym --policy png --episodes 200 --out results/png.csv
python -m guidance_rl.eval.eval_gym --policy checkpoints/rl_policy.pt --episodes 200 --out results/rl_v1.csv
# V2 评估：与 V1/PNG 三方对比（同环境，同指标口径）
python -m guidance_rl.eval.eval_v2 --policy png --episodes 200 --out results/png_v2_env.csv
python -m guidance_rl.eval.eval_v2 --policy checkpoints/rl_policy_v2.pt --episodes 200 --out results/rl_v2.csv

# 10) 导出部署（阶段二）
python -m guidance_rl.export_v2 --ckpt checkpoints/rl_policy_v2.pt \
    --out /home/verser/ros2_ws/src/uav_rl_guidance/models/policy_v2.pt
```

**预期基线（PNG 老师在 Gym 内，30 集/模式）**：
circle ~67-83%（脱靶集中 0.8-1.3m，PN 稳态滞后 —— RL 的主要改进空间）、
sinusoidal ~97%、random_walk ~87%、hover_escape 100%。
RL 目标：circle/random_walk 命中率超过 PNG ≥10 个百分点，其余不退化。

## 5. Gazebo 部署与 A/B 对比

见 `guidance_rl/eval/batch_gazebo_eval.md`。要点：

**阶段一部署：**
```bash
ros2 launch uav_rl_guidance rl_guidance.launch.py              # RL 策略 (V1)
ros2 launch uav_rl_guidance rl_guidance.launch.py fallback_png:=true  # PNG 基线
```

**阶段二部署：**
```bash
ros2 launch uav_rl_guidance rl_guidance.launch.py model_version:=v2   # CNN+GRU (V2)
ros2 launch uav_rl_guidance rl_guidance.launch.py model_version:=v1   # V1 基线对比
ros2 launch uav_rl_guidance rl_guidance.launch.py fallback_png:=true  # PNG 基线
```

- 与原 vpng 实验流程完全一致：`run_swarm.sh` → `uav_target_sim` → `uav_vision_dectect` → 本节点
- V2 额外订阅 `/camera/image` 做 288×288 搜索区域裁剪（复用 LightTrack `get_search_bbox` 逻辑）
- 首次部署前用 `record_gazebo_episode.py` 录数据校准 `dynamics.tau_v`
- 策略异常时 watchdog 自动回退 PNG 并告警（连续 20 帧锁存）

## 6. 目录结构

```
guidance_rl/
├── configs/default.yaml          # 全部超参（相机/动力学/奖励/训练）
├── guidance_rl/
│   ├── geometry.py               # 像素↔LOS（与 C++ 逐行对齐）
│   ├── png_teacher.py            # PNG 老师 = BC 专家 = 部署回退（唯一实现）
│   ├── features.py               # 特征/动作编解码（训练与部署共用）
│   ├── envs/                     # target_motion / interceptor_dynamics /
│   │                             #   camera_model / intercept_env(+VecEnv)
│   ├── models/policy.py          # GRU Actor + 特权前馈 Critic
│   ├── train/                    # collect_bc_data / train_bc / ppo / train_ppo
│   ├── eval/                     # eval_gym / plot_results /
│   │                             #   record_gazebo_episode / batch_gazebo_eval.md
│   └── export.py                 # → TorchScript + meta json
└── tests/test_geometry.py        # 26 个与 C++ 的一致性用例

ros2_ws/src/uav_rl_guidance/      # 部署包（ament_python）
├── uav_rl_guidance/
│   ├── rl_guidance_node.py       # 状态机移植 + 50Hz 心跳 + 20Hz 策略
│   └── policy_runtime.py         # TorchScript 加载 + watchdog
├── config/params.yaml  launch/  models/
```

## 7. 已知简化与后续工作

- Gym 拦截机为质点+一阶速度响应，无桨叶/气动细节 → 用 record_gazebo_episode 校准 tau_v 弥合
- `hover_escape` 模式 Gym 有而 uav_target_sim 暂无，Gazebo 验证需补 C++ 实现或跳过
- 统计层 GPS→NED 用平面近似（C++ 用 GeographicLib），<1km 场景误差可忽略
- 阶段二入口：把 `envs/camera_model.py` 替换为渲染图像 + 共享 CNN 编码器（见拦截路线图）
