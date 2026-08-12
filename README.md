# guidance_rl

视觉无人机拦截学习工程，包含两个有效阶段：

| 阶段 | 状态 | 输入 | 输出 |
|---|---|---|---|
| 学习制导 | 已保留 | bbox 构造的 15 维特征 | 相对 LOS 制导动作 |
| 全端到端 | 已实现，待正式训练 | 连续两张完整 RGB 图像 | 机体系速度与偏航角速率 |

两个阶段共享目标运动、拦截机动力学和命中判定。全端到端 Actor 只接收完整图像，不读取 bbox、LOS、GPS 或仿真真值。

## 核心结构

```text
共享仿真内核
├── 学习制导：bbox → 15维特征 → GRU → LOS制导动作
└── 全端到端：完整RGB双帧 → CNN注意力 → Transformer
                              ├── 速度动作
                              ├── 未来位置
                              └── 风险与置信度
```

全端到端阶段输出 `[forward, right, down, yaw_rate]`。部署时只使用飞控当前 yaw 将机体系速度旋转为 NED，不依赖目标检测结果。

## 安装与测试

推荐 Python 3.10+：

```bash
conda activate guidance-rl
cd /home/verser/Python/guidance_rl
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

## 全端到端使用流程

### 1. 收集 PNG 专家数据

```bash
python -m guidance_rl.training.collect_e2e_data \
  --episodes 1000 --out data/e2e_bc
```

数据按 episode 保存，已有目录默认不会被覆盖；需要重建时添加 `--overwrite`。

### 2. 行为克隆

```bash
python -m guidance_rl.training.train_e2e_bc \
  --data data/e2e_bc --out checkpoints/e2e_bc.pt
```

### 3. PPO 微调

```bash
python -m guidance_rl.training.train_e2e_ppo \
  --bc-init checkpoints/e2e_bc.pt \
  --out checkpoints/e2e_rl.pt --logdir runs/e2e_ppo
```

### 4. 评估

```bash
# PNG 基线
python -m guidance_rl.evaluation.eval_e2e \
  --policy png --episodes 200 --out results/e2e_png.csv

# 全端到端策略与注意力图
python -m guidance_rl.evaluation.eval_e2e \
  --policy checkpoints/e2e_rl.pt --episodes 200 \
  --out results/e2e_rl.csv \
  --attention-dir results/attention --attention-count 20
```

评估指标包括命中率、最近接距离、拦截时间、风险、置信度和建议回退比例。添加 `--apply-fallback` 可评估低置信度时切换 PNG 的组合策略。

### 5. 导出

```bash
python -m guidance_rl.export_e2e \
  --ckpt checkpoints/e2e_rl.pt --out export/e2e_policy.pt
```

生成：

- `e2e_policy.pt`：TorchScript Actor；
- `e2e_policy_meta.json`：输入格式、动作协议、安全阈值和校验值。

运行时示例：

```python
from guidance_rl.end_to_end.runtime import EndToEndRuntime

runtime = EndToEndRuntime("export/e2e_policy.pt")
result = runtime.step(camera_rgb, current_yaw)

if result.fallback_required:
    activate_safety_controller(result.reason)
else:
    publish_ned_velocity(result.command.ned_velocity,
                         result.command.yaw_rate)
```

## 学习制导使用流程

```bash
python -m guidance_rl.training.collect_bc_data \
  --episodes 2000 --out data/bc_dataset.npz
python -m guidance_rl.training.train_bc \
  --data data/bc_dataset.npz --out checkpoints/bc_policy.pt
python -m guidance_rl.training.train_ppo \
  --bc-init checkpoints/bc_policy.pt --out checkpoints/rl_policy.pt
python -m guidance_rl.evaluation.eval_gym \
  --policy checkpoints/rl_policy.pt --episodes 200 --out results/guidance.csv
python -m guidance_rl.export \
  --ckpt checkpoints/rl_policy.pt --out export/policy.pt
```

## 目录结构

```text
guidance_rl/
├── configs/default.yaml          # 全部配置
├── guidance_rl/
│   ├── environments/             # 共享物理内核与学习制导环境
│   ├── models/policy.py          # 学习制导模型
│   ├── end_to_end/               # 全端到端环境、模型、动作和运行时
│   ├── training/                 # 数据采集、BC、PPO
│   ├── evaluation/               # 评估与注意力可视化
│   ├── export.py                 # 学习制导模型导出
│   └── export_e2e.py             # 全端到端模型导出
└── tests/                        # 两个阶段的测试
```

默认参数位于 `configs/default.yaml`。全端到端阶段使用 `end_to_end.*` 配置项。

## 当前限制

- 全端到端阶段已完成代码和冒烟验证，但尚未完成正式大规模训练。
- 当前视觉环境为快速 2D 域随机化渲染，实机前仍需 Gazebo/高保真仿真验证。
- 环境从目标已进入视场的拦截状态开始，不负责起飞和全局搜索。
- ROS2/PX4 全端到端节点尚未接入本仓库，真实部署必须保留 watchdog 和人工接管。

更详细的阶段说明和验收标准见 [ROADMAP.md](ROADMAP.md)。
