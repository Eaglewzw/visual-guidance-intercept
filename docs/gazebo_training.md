# Gazebo / PX4 训练指南

## 前置检查

固定环境：Gazebo Harmonic、ROS 2 Humble、PX4 SITL、Micro XRCE Agent，以及现有
`AeroIntercept` Conda 环境。不要创建新 Conda 环境，也不要修改 `ros2_ws` C++。

```bash
gz sim --versions
test -x /home/verser/PX4-Autopilot/build/px4_sitl_default/bin/px4
test -x /home/verser/ros2_ws/build/uav_target_sim/uav_target_sim
```

launcher 若发现已有 PX4、Gazebo 或 Micro XRCE 进程会直接拒绝启动，不会静默
`killall`。

## GUI 观察

终端一：

```bash
bash /home/verser/Python/AeroIntercept/aerointercept/gazebo/scripts/launch_gazebo.sh \
  --mode circle --seed 31
```

终端二：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AeroIntercept
cd /home/verser/Python/AeroIntercept
python -m aerointercept.gazebo.scripts.view_camera --display
```

Gazebo GUI 显示外部视角；`view_camera` 显示 Actor 收到的精确 640×640 RGB 输入。

## 冒烟

```bash
python -m aerointercept.gazebo.scripts.smoke_camera \
  --launch --headless --frames 8 --mode circle \
  --output results/gazebo_camera.png

python -m aerointercept.gazebo.scripts.smoke_env \
  --launch --headless --steps 32 --mode sinusoidal --seed 31
```

camera smoke 检查形状、dtype、范围、常量帧、连续序号和独立缓冲区；env smoke 检查
物理 reset、PX4 action、15 维 Critic、奖励和终止信息。

## 512-step CUDA PPO

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 \
  --config configs/gazebo_e2e.yaml --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 --encoder-chunk-size 4 \
  --checkpoint-interval 256 --logdir runs/gazebo_e2e_smoke --mode mixed
```

输出必须显示 `parameter_delta > 0` 才算实际完成 CUDA 参数更新。checkpoint 位于：

```text
runs/gazebo_e2e_smoke/checkpoints/best.pt
runs/gazebo_e2e_smoke/checkpoints/last.pt
runs/gazebo_e2e_smoke/checkpoints/step_XXXXXXXXX.pt
```

## 恢复、正式训练和评估

```bash
python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 512 --rollout-steps 16 \
  --checkpoint runs/gazebo_e2e_smoke/checkpoints/last.pt --resume \
  --checkpoint-interval 256 --logdir runs/gazebo_e2e_smoke --mode mixed

python -m aerointercept.gazebo.scripts.train_e2e_ppo \
  --launch --headless --device cuda:0 --num-envs 1 --seed 31 \
  --total-steps 1000000 --rollout-steps 32 \
  --checkpoint-interval 10000 --logdir runs/gazebo_e2e_formal --mode mixed

python -m aerointercept.gazebo.scripts.evaluate \
  --launch --headless --device cuda:0 --episodes 10 --mode circle \
  --checkpoint runs/gazebo_e2e_smoke/checkpoints/best.pt \
  --output results/gazebo_e2e_eval.json

tensorboard --logdir runs/gazebo_e2e_smoke --port 6006
```

正式百万步训练前应先完成 512-step 实测并确认 16GB 显存安全。Gazebo/PX4 按真实
时间异步运行，吞吐会明显低于内存内向量化二维环境。

## 并行限制

`GazeboVectorEnv` 接受多个 `--socket`，但每个 socket 必须对应独立的 Gazebo
partition、ROS domain、Micro XRCE 端口和 PX4 进程。当前自动 launcher 只支持一套；
在隔离 launcher 完成并经过相机 buffer 测试前，不允许用重复 socket 冒充并行环境。

## 已测性能

2026-08-24 在 RTX 5070 Ti 16GB、seed 31、单环境、rollout 16、总步数 512、
`encoder_chunk_size=4` 下：

| 指标 | 实测 |
|---|---:|
| PPO updates | 32 |
| Actor parameter max delta | 0.00672755 |
| 整机 GPU 峰值 | 3284 MB |
| PyTorch allocated 峰值 | 1918 MB |
| 相机 FPS | 7.49 |
| 端到端训练 FPS | 7.26 step/s |
| PPO 更新累计耗时 | 17.46 s |
| 总耗时 | 70.53 s |

恢复测试从 global step 512 正确继续到 528，并重新保存 optimizer、RNG、last 和周期
checkpoint。随机初始化仅训练 512 步的 hit rate 为 0，不应作为收敛结果引用。

## 回归测试

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate AeroIntercept
cd /home/verser/Python/AeroIntercept
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

普通回归测试覆盖模型协议、旧环境和无需启动 Gazebo 的任务逻辑。真实物理链路仍需
使用本文前面的 camera smoke、environment smoke 和 CUDA PPO 冒烟命令验收。

## 保留的轻量工作流

轻量二维环境继续用于快速数据采集、行为克隆和算法回归，不代替 Gazebo 高保真验收：

```bash
python -m aerointercept.training.collect_e2e_data \
  --episodes 1000 --out data/e2e_bc

python -m aerointercept.training.train_e2e_bc \
  --data data/e2e_bc --out checkpoints/e2e_bc.pt

python -m aerointercept.training.train_e2e_ppo \
  --bc-init checkpoints/e2e_bc.pt \
  --out checkpoints/e2e_rl.pt --logdir runs/e2e_ppo

python -m aerointercept.evaluation.eval_e2e \
  --policy checkpoints/e2e_rl.pt --episodes 200 \
  --out results/e2e_rl.csv
```
