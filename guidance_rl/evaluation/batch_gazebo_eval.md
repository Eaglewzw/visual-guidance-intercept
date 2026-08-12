# Gazebo 批量评估手册（RL 策略 vs PNG 基线）

## 前置条件

1. 已完成训练并导出模型：
   ```bash
   conda activate guidance_rl
   cd /home/verser/Python/guidance_rl
   python -m guidance_rl.export --ckpt checkpoints/rl_policy.pt \
       --out /home/verser/ros2_ws/src/uav_rl_guidance/models/policy.pt
   ```
2. 系统 Python（ROS2 用）已安装依赖：
   ```bash
   /usr/bin/python3 -m pip install -e /home/verser/Python/guidance_rl --user
   ```
3. `colcon build --packages-select uav_rl_guidance --symlink-install`

## 单集实验流程（与 vpng 实验完全一致，只换第 3 步）

```bash
# 终端 1：仿真环境（同原流程）
cd ~/PX4-Autopilot && ./run_swarm.sh

# 终端 2：目标机（逐模式切换 circle/sinusoidal/random_walk）
ros2 run uav_target_sim uav_target_sim --ros-args -p motion_mode:=circle

# 终端 3：视觉检测（不变）
ros2 run uav_vision_dectect uav_vision_dectect

# 终端 4A：RL 策略
ros2 launch uav_rl_guidance rl_guidance.launch.py \
    csv_path:=/home/verser/ros2_ws/results/rl_circle_01.csv

# 终端 4B：PNG 基线（二选一；同一节点跑基线保证其余条件全同）
ros2 launch uav_rl_guidance rl_guidance.launch.py fallback_png:=true \
    csv_path:=/home/verser/ros2_ws/results/png_circle_01.csv
# （或运行原 uav_vision_png 节点交叉验证 Python 移植的 PNG 与 C++ 一致）
```

每集结束（命中→DONE 或手动 Ctrl+C），CSV 自动写入摘要行
（最近接距离/命中时刻/丢失帧统计，与 vpng_intercept_stats.csv 同格式，
可直接用现有 `vpng_plot.m` 绘图）。

## 建议实验矩阵

| 运动模式 | RL 集数 | PNG 集数 | 备注 |
|---|---|---|---|
| circle | 10 | 10 | 重点：近脱靶改善（Gym 中 PNG 脱靶集中在 0.8~1.3m）|
| sinusoidal | 10 | 10 | |
| random_walk | 10 | 10 | 每集天然随机 |
| hover_escape* | 5 | 5 | *需给 uav_target_sim 增加该模式，或沿用 random_walk |

汇总指标（逐 CSV 摘要行收集）：命中率、最近接距离均值、命中时刻、
丢失帧率、watchdog 触发次数（RL 专属，应为 0）。

## sim-to-sim 校准（首次部署前建议做一次）

```bash
# 仿真运行中录制 60s 闭环数据
python3 -m guidance_rl.evaluation.record_gazebo_episode --out data/gazebo_ep1.npz --duration 60
```

校准点：
1. **tau_v**：取 `cmd_v*` 与 `v*` 列做一阶系统拟合，
   若与 0.40s 偏差大，更新 `configs/default.yaml: dynamics.tau_v` 后重新训练；
2. **检测丢失率/噪声**：统计 `det_w == -1` 比例与 bbox 抖动方差，
   对照 `camera.miss_*` / `pixel_noise_frac`；
3. **倾角幅值**：加速段 `pitch` 峰值对照 Gym 的 tilt 模型（tilt_max/tau_att）。

## 故障排查

- 节点报 `No module named guidance_rl` → 前置条件 2 未对系统 Python 执行
- 节点报特征版本不匹配 → 用当前 guidance_rl 代码重新 export
- 日志频繁出现 `[watchdog]` → 策略输出异常，检查模型与 meta 是否配套；
  watchdog 会自动回退 PNG，实验数据仍有效但应标注
- 想跳过起飞在台架上验证链路：`--ros-args -p bench_test:=true`
