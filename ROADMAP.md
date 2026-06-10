# 端到端视觉拦截无人机——三阶段路线图

> **项目背景**：当前系统 `<home/verser/ros2_ws>` 为模块化拦截方案——`uav_vision_dectect`
> (YOLOv5s+LightTrack, TensorRT→RectMsg) 产出像素 bbox，`uav_vision_png`
> (vision_png_control.cpp) 用经典比例导引 (PNG) 将 bbox 转为 NED 速度指令，
> 经 PX4 Offboard 控制拦截机。整个链路在 Gazebo+PX4 SITL 验证，已积累
> `vpng_intercept_stats*.csv` 基线数据。

> **目标**：从模块化流水线逐步演进到统一的端到端学习模型，每一阶段独立可验证、
> 可发论文，核心指标（命中率/最近接距离/拦截时间）口径连续可比。


## 总览

```
阶段一（✅ 已完成）                阶段二（✅ 工程建成 · 训练待执行）      阶段三（远期）
感知冻结 · 决策学习                视觉特征+决策联合学习             全端到端
┌────────────────────┐           ┌────────────────────┐           ┌────────────────────┐
│ YOLO + LightTrack  │           │ 轻量CNN共享编码器    │           │                    │
│   ↓ bbox (4 个数)  │           │   ↓ 特征图+bbox     │           │  图像 → 单模型     │
│ GRU 策略 → 速度指令 │           │ GRU 策略 → 速度指令  │           │    → 速度指令      │
│                    │           │                    │           │                    │
│ 训练: 轻量Gym(11k/s)│          │ 训练: IsaacLab/     │           │ 训练: GPU并行仿真   │
│ BC+PPO, 非对称Critic│          │  AerialGym(GPU并行) │           │ 大规模域随机化      │
└────────────────────┘           └────────────────────┘           └────────────────────┘
  sim2real: bbox抽象天然解耦        sim2real: 需要域随机化             sim2real: 最大挑战
  ~10万参数, CPU部署                ~500万参数, TensorRT部署          ~千万参数
```

三层递进关系：
- **阶段一是地基**：证明了"bbox→控制"可学习且优于手调PNG，建立了训练/评估/部署/基线四套基础设施
- **阶段二是主战场**：真正把视觉特征纳入学习，模型能利用零散视觉线索（目标朝向、背景运动视差、螺旋桨模糊方向）实现"直觉前置"，同时保留感知层做安全壳
- **阶段三是终局**：单一模型从像素到控制，但需要阶段二积累的真机经验作为前提


## 阶段一：学习制导律（✅ 已建成）

### 核心思路

**保留感知层，只替换制导律。** YOLO+LightTrack 继续产出 bbox，用 GRU 策略替换
`vision_png_control.cpp` 里的 PNG 算法。输入仍是 bbox 序列 + 自身状态，
输出仍是 NED 速度指令（TrajectorySetpoint，PX4 Offboard 速度模式）。

### 为什么从这里开始

| 优势 | 说明 |
|---|---|
| **sim2real 免费** | 输入是 bbox（4 个数），策略"看不见"像素。Gazebo 训的策略换真实相机只要检测器工作就能直接用 |
| **训练极快** | 轻量 Gym（运动学质点模型+针孔投影+检测噪声），~11k steps/s，百万步几分钟 |
| **基线现成** | PNG 老师移植自你的 C++（逐行对齐，26 个 pytest 锁），BC 热启动让 RL 从可用起步 |
| **论文价值** | "学习制导律 vs 经典 PNG 对机动目标"是干净的研究命题，消融实验简单（BC/PPO/无特权Critic）|
| **工程安全** | PNG 全程在回路（BC 老师→watchdog 兜底），策略异常自动回退 |

### 技术方案（已实现）

**观测（15 维特征，`guidance_rl/features.py`）**：

| 维度 | 含义 | 为什么需要 |
|---|---|---|
| los_v, sin/cos(los_z) | 经机体→NED 旋转的视线角 | 目标在空间中的方向（PNG 的核心输入）|
| dlos_v/dt, dlos_z/dt | 视线角速率 | **单帧不含运动信息**，差分/GRU 从这里提取接近速度与目标机动 |
| ln w, ln h, δln w | bbox 对数尺寸及变化率 | 距离的代理信号（w ∝ 1/R），变化率 = 接近速率代理 |
| valid, age | 检测有效性 + 丢失时长 | 策略从中学习"惯性续飞"（PNG 手写的那套丢失分级）|
| vx,vy,vz,\|V\|,高度 | 自身运动状态 | 为"自身机动对 LOS 的影响"提供自车运动补偿 |

> 丢失帧：保持 last-known LOS/尺寸，速率清零，valid=0、age 递增。
> 相当于把 C++ 里 `handle_intercept:453-469` 那段丢失分级逻辑从 if-else
> 变成了策略自己习得的潜行为。

**动作（4 维 tanh，带 PNG 归纳偏置）**：

```
零动作 ≡ 纯追踪（速度沿 LOS 方向，中速）：策略初始行为天然稳定
a0 → 仰角前置偏移 (LOS + a0×1.2, clamp ±45°)
a1 → 方位前置偏移 (LOS + a1×1.2)
a2 → 速度 2~5 m/s
a3 → 偏航角速率 ±1 rad/s
→ 合成 NED 速度指令（公式与 handle_intercept:478-480 一致）
```

**奖励函数**：

```
r = + 1.0 · (距离缩短速率 / 期望速度)     ← 密集主信号
    + 50.0 · 命中 (<0.8m)              ← 稀疏终局
    − 0.1  · FOV偏离²                 ← 保持目标在画面内
    − 0.2  · 丢失帧                   ← 鼓励少丢
    − 0.05 · 动作突变²                ← 保护真机
    − 0.01 · 时间                     ← 尽快拦截
    − 20/10 · 丢失终止/超时            ← 失败惩罚
```

**训练方案：BC 热启动 + Recurrent PPO + 非对称 Actor-Critic**

```
                    ┌──── BC (行为克隆) ────┐
PNG 老师在 Gym       │  策略网络克隆 PNG 行为 │  策略获得"可用"初始：命中率 70-90%
闭环采样 ~2000 集    │  MSE 损失，监督学习    │  从"PNG 做不好的地方"开始学
20 万条时序样本 ────→│  epochs=20, lr=3e-4   │
                    └───────────────────────┘
                                    ↓ BC 权重热启动
    ┌───────────────── PPO (强化学习) ──────────────────┐
    │  64 并行 Gym, GRU Actor, 特权前馈 Critic          │
    │  rollout 128 步 → GAE(γ=0.995,λ=0.95)            │
    │  clip 0.2, 熵 0.001, KL 提前停止(0.03)            │
    │  5M steps, lr=1e-4 (微调避免灾难性遗忘)            │
    └────────────────────────────────────────────────────┘
```

为什么这三个组合：
- **BC 热启动**：纯 RL 从随机策略出发，极难碰见"命中"这个稀疏大奖励。先克隆 PNG 到可运行水平，RL 只需做增量改善
- **Recurrent PPO**：单帧 bbox 不含距离（单目 bearing-only）——这和你 PNG 里做 LOS 差分的原因相同。GRU 把"若干帧历史"编码为隐状态，隐式提取接近速度和目标机动模式。PPO（on-policy）因为环境便宜（11k steps/s）比 SAC（off-policy）更稳定
- **非对称 Critic**：Actor 只看 15 维视觉特征（部署可得），Critic 额外看真值位置/速度（仿真独有）。效果：优势估计质量大幅提升 → Actor 梯度方向更准 → 收敛快。部署时 Critic 丢弃，纯视觉原则不破

### 部署架构

```
ros2_ws/src/uav_rl_guidance/（ament_python, rclpy）
与 uav_vision_png 话题/QoS/状态机/CSV 格式完全一致，一行 launch 切换 A/B

┌────────────── rl_guidance_node.py ────────────────────┐
│  状态机: TAKE_OFF → SEARCHING → INTERCEPT → DONE      │
│           TRACK_LOST（制动慢旋搜索）                    │
│                                                        │
│  INTERCEPT 态:                                         │
│    20Hz FeatureBuilder(15维) → TorchScript GRU        │
│    → decode_action → 50Hz 心跳发布 TrajectorySetpoint │
│                                                        │
│  watchdog（自动回退内置 PNG）:                          │
│    NaN/Inf 输出 → 隐藏状态清零 → 1s 锁存 PNG           │
│    速度越界 (>7.5m/s) → 同上                           │
│    推理异常 → 同上                                     │
│                                                        │
│  fallback_png:=true → 全程 PNG 基线（A/B 实验用）       │
│  bench_test:=true  → 跳过起飞直接 SEARCHING（台架测）    │
└────────────────────────────────────────────────────────┘
```

### 当前进度

| 组件 | 状态 |
|---|---|
| PNG 老师 (移植 C++ 全部细节) | ✅ 26 个一致性测试通过 |
| 轻量 Gym (目标 4 模式 + 一阶动力学 + 针孔相机 + 检测噪声/丢失/延迟) | ✅ ~11k steps/s |
| GRU Actor + 特权 Critic | ✅ |
| BC 数据采集 + 训练 | ✅ 冒烟通过 |
| Recurrent PPO (BC 热启动) | ✅ 冒烟通过 |
| 评估 (PNG/BC/RL × 运动模式) + 导出 TorchScript | ✅ |
| ROS2 部署节点 (状态机 + watchdog) | ✅ 编译、台架测试通过 |
| **训练执行** (2000 集 BC → PPO 5M 步) | ⏳ 用户执行中 |
| **Gazebo A/B 对比** | ⏳ 训练完成后执行 |

**Gym 内 PNG 基线**（当前运动模型参数，30 集/模式）：
circle ~67-83%（脱靶 0.8-1.3m，PN 稳态滞后——RL 的改进空间）、sinusoidal ~97%、
random_walk ~87%、hover_escape 100%。

### 阶段一的论文切入点

1. **学习制导律 (Learned Guidance)**：证明一个小 GRU 能在 bearing-only 条件下超越经典 PNG——输入端只用 4 维 bbox 和自身姿态，不需要距离/目标速度，但能隐式从时序中学出等效量
2. **非对称 Actor-Critic 训练**：方法贡献——Critic 在训练时用仿真真值（恰好对应系统中"统计专用、导引禁用"的 GPS 数据），Actor 只用部署可得的视觉特征。理论上这等价于用真值做贝尔曼目标的高质量自助法，不违反纯视觉原则
3. **从比例导引到学习策略的平滑迁移**：BC 热启动 → PPO 微调，全程不随机探索即收敛，可作为"传统控制器如何驱动学习"的案例研究


## 阶段二：半端到端——视觉特征+决策联合学习

### 核心思路

**把 LightTrack 的搜索区域图像作为策略的一部分输入，共享 CNN 编码器。**
YOLO 仍做全局目标捕获（SEARCHING→INTERCEPT 切换），但 INTERCEPT 阶段
不再用几何投影抽象——而是直接喂裁剪后的搜索区域图像给网络。

### 已实现组件（2026-06-10）

| 组件 | 文件 | 状态 |
|---|---|---|
| 2D 精灵渲染器（替代 Gazebo 渲染）| `envs/uav_renderer.py` | ✅ 冒烟通过 |
| MobileNetV3-Small + GRU + 辅助 bbox/conf 头 | `models/policy_v2.py` | ✅ 梯度链路验证 |
| 图像观测 Gym 环境 | `envs/intercept_env_v2.py` | ✅ PNG 闭环 8/10 hit |
| BC 数据采集（JPEG + npz 索引）| `train/collect_bc_data_v2.py` | ✅ 5 集测试通过 |
| BC 训练（动作 MSE + 辅助 bbox/conf）| `train/train_bc_v2.py` | ✅ 2 epoch loss 下降 |
| PPO 微调（8 envs × 128 steps）| `train/train_ppo_v2.py` | ✅ 待完整训练 |
| TorchScript 导出 + 部署 runtime | `export_v2.py` + `policy_runtime_v2.py` | ✅ watchdog 验证 |
| ROS2 节点 V1/V2 切换 | `rl_guidance_node.py` + `model_version` 参数 | ✅ 编译通过 |

### 为什么需要阶段二

阶段一的 bbox 抽象丢掉了信息：目标朝向、螺旋桨模糊方向（暗示速度矢量）、
阴影-地面关系（暗示高度）、运动视差（背景与目标的相对运动）。这些人类飞手
和优秀竞技飞手都在用，但 4 个数无论如何承载不了。

具体来说，阶段一在以下场景里接近理论上限：
- 目标直线匀速 → bbox 轨迹已经很充分
- 目标突然急转弯 → bbox 变化有延迟（先姿态变化后位移变化），图像里能提前看见

### 技术方案（设计稿）

**模型架构**：

```
搜索区域图像 (288×288×3)
        │
  ┌─────▼──────────────────┐
  │  CNN 编码器              │  可用 LightTrack 初始化权重预热
  │  (MobileNetV3/EfficientNet│  或阶段二专用预训练
  │   轻量骨干)              │
  │   ↓ feature map 7×7×C   │
  ├───────────┬─────────────┤
  │  GAP+MLP  │  Conv head  │
  │  128维    │  bbox回归    │ ← 辅助监督头（稳定表征，防 RL 漂移）
  │  视觉特征  │  (MSE)      │
  └─────┬─────┴─────────────┘
        │ concat
  ┌─────▼──────────────────────┐
  │ 自身状态(速度/姿态/高度)      │
  │ GRU(128) + MLP 头           │
  ├─────────────────────────────┤
  │ → 速度指令 (同阶段一动作空间) │
  └─────────────────────────────┘
```

**关键设计**：
- **辅助 bbox 回归头**是核心技巧：纯 RL 要在几十万步后才学会从 CNN 特征里提取目标位置；加一个 MSE 监督（标签=裁剪后的 bbox 真值），表征在几千步内就稳定，RL 从头即高效
- **CNN 初始化来源**：LightTrack 的 init 分支（127×127 模板）和 update 分支的骨干可作为热启动，它们是专门针对小目标跟踪搜索出来的轻量架构
- **仍保留 YOLO 做 SEARCHING**：全局搜索是大视场全图扫描，没必要用小骨干重新学——YOLO 已经够好。阶段二的网络只负责"目标已锁定，如何最优追踪和拦截"

**训练环境升级**：

| 维度 | 阶段一 | 阶段二 |
|---|---|---|
| 环境 | 轻量 Gym (质点+针孔投影) | Isaac Lab / Aerial Gym (GPU 并行) |
| 渲染 | 不需要 | 需要（Gazebo 资源可直接复用） |
| 速度 | ~11k steps/s | ~1-5k steps/s (GPU 并行) |
| 域随机化 | bbox 级噪声 | 光照/天气/纹理/目标外观/相机参数 |

**sim2real 挑战与对策**：
- 搜索区域图像（288×288）是**局部裁剪**，比全帧 1920×1080 更容易做域随机化——随机化的对象是小区域内的纹理/光照，不涉及场景级别的多样性
- 可以用少量真实无人机视频做 fine-tune（冻结 CNN 骨干，只调 GRU 头），类似"单目标跟踪 fine-tune"的范式
- 如果你的 OAK-D Lite 有双目深度，还能把深度图加入通道——距离不再需要从 bbox 尺寸间接推理

**部署**：
- CNN 骨干导出 TensorRT（和 YOLO/LightTrack 同一推理框架），GRU 头在 CPU 上走 LibTorch
- 搜索区域裁剪沿用 LightTrack 已有的 `get_search_bbox()` 逻辑 (`main.py:76-103`)，不需要额外工程

### 阶段二的论文切入点

1. **"看出来的前置量"**：证明视觉外观线索（姿态/桨叶模糊/阴影）能提供 bbox 轨迹之外的独特控制价值——消融 "只用 bbox vs bbox+图像" 即可
2. **辅助监督稳定 RL 表征**：方法贡献——证明在视觉+RL 的联合训练中，一个廉价的 bbox 回归头让表征学习从几十万步降到几千步
3. **从仿真渲染到真实跟踪的域自适应**：少量实机视频做轻量 fine-tune，展示 sim2real transfer


## 阶段三：全端到端——像素进、控制出

### 核心思路

**单一模型，输入图像 → 输出速度指令。** 不分感知/制导/控制模块，一个网络
完成从像素到拦截的完整映射。这是路线图的终局，也是学术价值最高的形态。

### 为什么放到最后

1. **数据需求爆炸**：阶段一千万步训练只用 bbox 抽象；阶段三每一帧样本都需要渲染完整图像 → 训练时间从小时级变天级，调试周期从分钟级变小时级
2. **sim2real 最困难**：Gazebo 的图像和真实 OAK-D Lite 图像差距大（材质/光照/大气/运动模糊/卷帘快门），渲染逼真度直接决定迁移成败
3. **需要阶段一和阶段二的真机经验**：只有经过前两阶段积累的实飞数据、动力学校准、视觉 pipeline 稳定性经验，才能评估全端到端方案是否在现实中有竞争力
4. **纯端到端的可解释性与安全认证**：无人机拦截是安全关键系统，一个不可解释的黑箱不受监管机构认可。需要做好可解释性包装（如注意力图显示"模型正在看目标机身"）和形式化验证

### 技术方案（远期规划）

**模型**：Siamese 风格的两帧输入 + Transformer 时序融合 + 多任务头（速度指令 + 未来 N 步目标位置预测 + 碰撞风险评估）

**训练**：
- 大算力平台（H100 级），Isaac Lab + 大规模场景域随机化
- 分阶段课程学习：静止目标 → 低速直飞 → 机动规避 → 对抗性逃逸
- 真机数据微调：少量实飞数据做 DAgger（数据集聚合），纠正仿真偏差

**安全兜底**：保留阶段一的 watchdog（速度越界→切 PNG），且端到端模型的"当前控制指令"和"置信度"一并输出，低置信度时自动降级

### 阶段三的论文切入点

这是完整系统级贡献，不只是 ML 论文：你展示了从经典模块化方案到端到端方案的平滑演进路径，保留了完整的对比基线链（阶段一 PNG→RL、阶段二半端到端、阶段三全端到端），每个阶段的增量价值都可定量量化。


## 附录

### A. 实验基线链（三个阶段的共同参照系）

```
                                                     阶段三
                                                       │
                                              ┌────────┴────────┐
                                              │  全端到端模型     │
                                              │  图像→速度指令    │
                                              └────────┬────────┘
                                                       │
                                              ┌────────┴────────┐
                                              │  半端到端模型     │
                               阶段二           │  CNN+GRU         │
                                 │             │  YOLO只做搜索     │
                       ┌────────┴────────┐     └────────┬────────┘
                       │  GRU 策略        │              │
         阶段一         │  bbox → 速度指令  │     ┌────────┴────────┐
           │           └────────┬────────┘     │  PNG 经典制导     │
  ┌────────┴────────┐          │              │  (C++ 手写基线)   │
  │ YOLO + LightTrack│          │              └──────────────────┘
  │ (感知层，不改动)  │          │
  └──────────────────┘          │
                                │
              所有方案共享同一套评估体系：
              命中率 / 最近接距离 / 拦截时间 / 丢失率
              × {circle, sinusoidal, random_walk, hover_escape}
              × {Gym (统计足够样本量), Gazebo (验证最关键几组)}
```

### B. 与现有工程的对应关系

| 工程路径 | 角色 | 阶段一 | 阶段二 | 阶段三 |
|---|---|---|---|---|
| `ros2_ws/src/uav_vision_dectect` | 感知层 | 不改动，继续产出 RectMsg | 只留 YOLO 做搜索；LightTrack 被 CNN 替换 | 被端到端模型取代 |
| `ros2_ws/src/uav_vision_png` | 制导层 | 被 `uav_rl_guidance` 替换 | 被阶段二部署节点替换 | 被端到端部署节点替换 |
| `ros2_ws/src/uav_target_sim` | 目标仿真 | 改，移植到 Gym (target_motion.py) | 改，移植到 Isaac Lab 环境 | 改，移植到 Isaac Lab 环境 |
| `ros2_ws/src/uav_ibvs_control` | IBVS 早期方案 | 无关 | 无关 | 无关 |
| `Python/YOLO_LT/inference_py` | 训练/推理 | 不直接使用，仅参考其检测流水线 | 可能复用 LightTrack 骨干训练代码 | 无关 |
| `Python/guidance_rl` | **← 新建训练工程** | **主战场** | 扩展：加 CNN 编码器 | 扩展：全端到端训练 |

### C. 风险与对策

| 风险 | 阶段 | 对策 |
|---|---|---|
| GYM→Gazebo 动力学偏差导致策略部署后表现差 | 一 | record_gazebo_episode 校准 τ_v；watchdog PNG 兜底 |
| 小目标检测在真实相机中比 Gazebo 差得多 | 一/二 | 噪声参数按实飞数据校准；噪音/漏检训练时故意给得更差 |
| Isaac Lab 环境搭建工作量大 | 二 | Aerial Gym (开源无人机 RL 环境) 已有 PX4 动力学+渲染，可复用 |
| 端到端模型不可解释 | 三 | 输出注意力图 + 碰撞置信度；保留模块化 fallback |
| 真机测试许可/成本 | 全 | 方案按 Gazebo→真机→更多真机的渐进节奏设计 |

### D. 快速命令速查

```bash
# 阶段一训练（当前）
conda activate guidance_rl && cd /home/verser/Python/guidance_rl
python -m guidance_rl.train.collect_bc_data --episodes 2000 --out data/bc_dataset.npz
python -m guidance_rl.train.train_bc --data data/bc_dataset.npz --out checkpoints/bc_policy.pt
python -m guidance_rl.train.train_ppo --bc-init checkpoints/bc_policy.pt --out checkpoints/rl_policy.pt --logdir runs/ppo
python -m guidance_rl.eval.eval_gym --policy checkpoints/rl_policy.pt --episodes 200 --out results/rl.csv
python -m guidance_rl.export --ckpt checkpoints/rl_policy.pt --out /home/verser/ros2_ws/src/uav_rl_guidance/models/policy.pt

# 阶段一部署（Gazebo）
ros2 launch uav_rl_guidance rl_guidance.launch.py              # RL 策略
ros2 launch uav_rl_guidance rl_guidance.launch.py fallback_png:=true  # PNG 基线

# ========== 阶段二 ==========
# 阶段二训练
python -m guidance_rl.train.collect_bc_data_v2 --episodes 500 --out data/bc_v2
python -m guidance_rl.train.train_bc_v2 --data data/bc_v2 --out checkpoints/bc_policy_v2.pt
python -m guidance_rl.train.train_ppo_v2 --bc-init checkpoints/bc_policy_v2.pt --out checkpoints/rl_policy_v2.pt --logdir runs/ppo_v2
python -m guidance_rl.export_v2 --ckpt checkpoints/rl_policy_v2.pt --out /home/verser/ros2_ws/src/uav_rl_guidance/models/policy_v2.pt

# 阶段二部署（Gazebo）
ros2 launch uav_rl_guidance rl_guidance.launch.py model_version:=v2
# 同一节点的 V1 基线
ros2 launch uav_rl_guidance rl_guidance.launch.py model_version:=v1
```
