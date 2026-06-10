"""guidance_rl: 视觉拦截学习制导律（阶段一）

保留感知层（YOLO+LightTrack → bbox），用 GRU 策略替换 PNG 制导律。
训练在轻量运动学 Gym 中完成（BC + PPO），部署到 ros2_ws/uav_rl_guidance。
"""

__version__ = "0.1.0"
