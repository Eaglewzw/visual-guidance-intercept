"""CUDA PPO training against physical PX4 vehicles in Gazebo Harmonic."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.gazebo.checkpoint import load_model_weights, save
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.environment import GazeboVectorEnv
from aerointercept.gazebo.process import maybe_launch
from aerointercept.training.train_e2e_ppo import ImageRolloutBuffer, ppo_update


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--socket", action="append", default=None,
                        help="one independently launched bridge socket per environment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-steps", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logdir", default="runs/gazebo_e2e")
    parser.add_argument("--mode", choices=("circle", "sinusoidal", "random_walk", "mixed"), default="mixed")
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    parser.add_argument("--encoder-chunk-size", type=int, default=4)
    parser.add_argument("--launch", action="store_true",
                        help="start and own one project Gazebo stack for this run")
    parser.add_argument("--headless", action="store_true",
                        help="used with --launch; training itself never fabricates images")
    return parser.parse_args()


def _gpu_usage_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


def _restore_rng(checkpoint: dict) -> None:
    state = checkpoint.get("rng_state", {})
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if "numpy" in state:
        np.random.set_state(state["numpy"])


def main():
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Gazebo PPO requires CUDA; use --device cuda:0")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.total_steps < args.num_envs * args.rollout_steps:
        raise ValueError("--total-steps must contain at least one complete rollout")
    rollout_size = args.num_envs * args.rollout_steps
    if args.total_steps % rollout_size:
        raise ValueError("--total-steps must be divisible by num_envs * rollout_steps")

    cfg = load_gazebo_config(args.config)
    cfg.end_to_end.model.encoder_chunk_size = args.encoder_chunk_size
    cfg.end_to_end.ppo.num_minibatches = max(4, int(cfg.end_to_end.ppo.num_minibatches))
    sockets = args.socket or [str(cfg.gazebo.bridge.socket)]
    if len(sockets) != args.num_envs:
        raise ValueError(
            f"--num-envs={args.num_envs} requires exactly that many --socket arguments; "
            "each socket must be an isolated Gazebo/PX4 world"
        )
    if args.launch and args.num_envs != 1:
        raise ValueError("automatic --launch is intentionally limited to one stack")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    stack = maybe_launch(args, sockets[0])
    environments = None
    writer = None
    try:
        environments = GazeboVectorEnv(cfg, sockets)
        frames, training_info, _ = environments.reset()
        if frames.shape[1:] != (2, 3, 640, 640) or frames.dtype != np.uint8:
            raise RuntimeError(f"Gazebo observation contract failed: {frames.shape} {frames.dtype}")
        if training_info["critic_obs"].shape != (args.num_envs, 15):
            raise RuntimeError("Gazebo privileged Critic contract is not [N,15]")

        model = EndToEndActorCritic(cfg.end_to_end.model).to(device).eval()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(cfg.end_to_end.ppo.learning_rate),
            weight_decay=1.0e-4,
        )
        global_step = 0
        best_hit_rate = -1.0
        if args.checkpoint:
            source = Path(args.checkpoint)
            checkpoint = torch.load(source, map_location=device, weights_only=False)
            if checkpoint.get("backend") not in (None, "gazebo_harmonic_px4_sitl"):
                raise ValueError("checkpoint was produced by a different simulator backend")
            load_model_weights(model, checkpoint, dict(cfg.end_to_end.model))
            if args.resume:
                if "optimizer" not in checkpoint:
                    raise ValueError("--resume checkpoint has no optimizer state")
                optimizer.load_state_dict(checkpoint["optimizer"])
                global_step = int(checkpoint.get("global_step", 0))
                best_hit_rate = float(checkpoint.get("best_hit_rate", -1.0))
                _restore_rng(checkpoint)
            print(
                f"[AeroIntercept] loaded {source} resume={args.resume} global_step={global_step}",
                flush=True,
            )

        logdir = Path(args.logdir)
        checkpoint_dir = logdir / "checkpoints"
        last_path = checkpoint_dir / "last.pt"
        best_path = checkpoint_dir / "best.pt"
        writer = SummaryWriter(str(logdir), purge_step=global_step if args.resume else None)
        recent_rewards = deque(maxlen=200)
        recent_outcomes = deque(maxlen=200)
        recent_minimum = deque(maxlen=200)
        recent_lengths = deque(maxlen=200)
        episode_rewards = np.zeros(args.num_envs, dtype=np.float64)
        initial_parameter = model.actor.action_head[-1].weight.detach().clone()
        parameter_delta = 0.0
        peak_gpu_mb = _gpu_usage_mb()
        ppo_seconds = 0.0
        started = time.perf_counter()
        camera_start = environments.camera_frames
        updates = args.total_steps // rollout_size
        latest_metrics = {}
        scalar_metrics = {}

        for update in range(1, updates + 1):
            buffer = ImageRolloutBuffer(
                args.rollout_steps, args.num_envs, frames.shape[1:], 15, 4
            )
            model.eval()
            for _ in range(args.rollout_steps):
                frame_tensor = torch.from_numpy(frames).to(device, non_blocking=True)
                privileged = torch.from_numpy(training_info["critic_obs"]).to(
                    device, non_blocking=True
                )
                with torch.no_grad():
                    outputs = model.act(
                        frame_tensor, privileged, False,
                        cfg.end_to_end.ppo.log_std_min,
                        cfg.end_to_end.ppo.log_std_max,
                    )
                actions, log_probability, values = outputs[:3]
                (next_frames, rewards, terminated, truncated,
                 next_training, infos) = environments.step(actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated)
                buffer.add(
                    frames, training_info, actions, log_probability,
                    rewards, dones.astype(np.float32), values,
                )
                episode_rewards += rewards
                for index, info in enumerate(infos):
                    final = info.get("final")
                    if final is not None:
                        recent_rewards.append(float(final["episode_reward"]))
                        recent_outcomes.append(final["outcome"])
                        recent_minimum.append(float(final["minimum_distance"]))
                        recent_lengths.append(int(final["episode_length"]))
                        episode_rewards[index] = 0.0
                frames = next_frames
                training_info = next_training
                global_step += args.num_envs
                peak_gpu_mb = max(peak_gpu_mb, _gpu_usage_mb())

            with torch.no_grad():
                last_value = model.critic(
                    torch.from_numpy(training_info["critic_obs"]).to(device)
                )
            buffer.compute_gae(
                last_value, cfg.end_to_end.ppo.gamma,
                cfg.end_to_end.ppo.gae_lambda,
            )
            update_started = time.perf_counter()
            latest_metrics = ppo_update(
                model, optimizer, buffer, cfg.end_to_end.ppo,
                cfg.end_to_end.auxiliary, device,
            )
            torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            ppo_seconds += update_seconds
            parameter_delta = float(
                (model.actor.action_head[-1].weight - initial_parameter)
                .detach().abs().max()
            )
            peak_gpu_mb = max(peak_gpu_mb, _gpu_usage_mb())
            elapsed = time.perf_counter() - started
            hit_rate = (
                sum(outcome == "hit" for outcome in recent_outcomes) / len(recent_outcomes)
                if recent_outcomes else 0.0
            )
            scalar_metrics = {
                **latest_metrics,
                "hit_rate": hit_rate,
                "episode_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                "minimum_distance": float(np.mean(recent_minimum)) if recent_minimum else float("nan"),
                "interception_time": (
                    float(np.mean(recent_lengths)) / 20.0 if recent_lengths else float("nan")
                ),
                "fov_lost_rate": (
                    sum(value == "fov_lost" for value in recent_outcomes) / len(recent_outcomes)
                    if recent_outcomes else 0.0
                ),
                "ground_collision_rate": (
                    sum(value == "ground" for value in recent_outcomes) / len(recent_outcomes)
                    if recent_outcomes else 0.0
                ),
                "episode_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                "training_fps": update * rollout_size / elapsed,
                "camera_fps": (environments.camera_frames - camera_start) / elapsed,
                "simulation_fps": update * rollout_size / elapsed,
                "gpu_memory_mb": peak_gpu_mb,
                "torch_peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                "ppo_update_seconds": update_seconds,
                "parameter_delta": parameter_delta,
            }
            for name, value in scalar_metrics.items():
                if np.isfinite(value):
                    writer.add_scalar(name, value, global_step)
            writer.flush()
            print(
                f"[AeroIntercept] update={update}/{updates} step={global_step} "
                f"policy={latest_metrics['policy_loss']:.4f} "
                f"value={latest_metrics['value_loss']:.4f} "
                f"kl={latest_metrics['approx_kl']:.5f} "
                f"delta={parameter_delta:.3e} gpu={peak_gpu_mb:.0f}MB",
                flush=True,
            )
            checkpoint_kwargs = dict(
                model=model, optimizer=optimizer, cfg=cfg,
                global_step=global_step, seed=args.seed,
                best_hit_rate=max(best_hit_rate, hit_rate), metrics=scalar_metrics,
            )
            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                save(best_path, **checkpoint_kwargs)
            run_steps = update * rollout_size
            if run_steps % args.checkpoint_interval == 0 or update == updates:
                save(last_path, **checkpoint_kwargs)
                save(checkpoint_dir / f"step_{global_step:09d}.pt", **checkpoint_kwargs)

        if parameter_delta <= 0.0:
            raise AssertionError("CUDA PPO completed without changing Actor parameters")
        elapsed = time.perf_counter() - started
        report = {
            "backend": "gazebo_harmonic_px4_sitl",
            "global_step": global_step,
            "run_steps": args.total_steps,
            "updates": updates,
            "device": str(device),
            "parameter_delta": parameter_delta,
            "last_checkpoint": str(last_path),
            "best_checkpoint": str(best_path),
            "periodic_checkpoint": str(checkpoint_dir / f"step_{global_step:09d}.pt"),
            "gpu_peak_total_mb": peak_gpu_mb,
            "gpu_peak_torch_mb": torch.cuda.max_memory_allocated() / 2**20,
            "camera_fps": (environments.camera_frames - camera_start) / elapsed,
            "simulation_fps": args.total_steps / elapsed,
            "training_fps": args.total_steps / elapsed,
            "ppo_update_seconds": ppo_seconds,
            "elapsed_seconds": elapsed,
            "metrics": scalar_metrics,
        }
        print("AEROINTERCEPT_GAZEBO_PPO=" + json.dumps(report), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if environments is not None:
            environments.close()
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    main()
