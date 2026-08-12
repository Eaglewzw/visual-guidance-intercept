"""PPO fine-tuning for the detector-free full-frame image actor."""
import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..config import load_config
from ..end_to_end.environment import VecEndToEndInterceptEnv
from ..end_to_end.policy import EndToEndActorCritic


class ImageRolloutBuffer:
    """CPU uint8 image storage plus dense PPO/auxiliary tensors."""

    def __init__(self, steps, environments, frame_shape, critic_dim, action_dim):
        shape = (steps, environments, *frame_shape)
        self.frames = np.empty(shape, dtype=np.uint8)
        self.privileged = torch.empty(steps, environments, critic_dim)
        self.actions = torch.empty(steps, environments, action_dim)
        self.log_probability = torch.empty(steps, environments)
        self.rewards = torch.empty(steps, environments)
        self.dones = torch.empty(steps, environments)
        self.values = torch.empty(steps, environments)
        self.future_position = torch.empty(steps, environments, 3)
        self.collision_risk = torch.empty(steps, environments)
        self.confidence = torch.empty(steps, environments)
        self.steps = steps
        self.environments = environments
        self.index = 0

    def add(self, frames, training_info, actions, log_probability,
            rewards, dones, values):
        index = self.index
        self.frames[index] = frames
        self.privileged[index].copy_(torch.from_numpy(
            training_info["critic_obs"]))
        self.actions[index].copy_(actions.detach().cpu())
        self.log_probability[index].copy_(log_probability.detach().cpu())
        self.rewards[index].copy_(torch.from_numpy(rewards))
        self.dones[index].copy_(torch.from_numpy(dones))
        self.values[index].copy_(values.detach().cpu())
        self.future_position[index].copy_(torch.from_numpy(
            training_info["future_position"]))
        self.collision_risk[index].copy_(torch.from_numpy(
            training_info["collision_risk"]))
        self.confidence[index].copy_(torch.from_numpy(
            training_info["confidence"]))
        self.index += 1

    def compute_gae(self, last_value, gamma: float, gae_lambda: float):
        advantages = torch.zeros_like(self.rewards)
        accumulator = torch.zeros(self.environments)
        last_value = last_value.detach().cpu()
        for index in reversed(range(self.steps)):
            next_value = (
                last_value if index == self.steps - 1
                else self.values[index + 1]
            )
            nonterminal = 1.0 - self.dones[index]
            delta = (
                self.rewards[index] + gamma * next_value * nonterminal
                - self.values[index]
            )
            accumulator = (
                delta + gamma * gae_lambda * nonterminal * accumulator)
            advantages[index] = accumulator
        self.advantages = advantages
        self.returns = advantages + self.values


def auxiliary_loss(future_prediction, risk_logit, confidence_logit,
                   future_target, risk_target, confidence_target, cfg):
    visible = confidence_target
    future_error = (future_prediction - future_target).square().mean(dim=-1)
    future = (future_error * visible).sum() / visible.sum().clamp_min(1.0)
    risk = F.binary_cross_entropy_with_logits(risk_logit, risk_target)
    confidence = F.binary_cross_entropy_with_logits(
        confidence_logit, confidence_target)
    return (
        cfg.future_coef * future
        + cfg.risk_coef * risk
        + cfg.confidence_coef * confidence
    ), future, risk, confidence


def ppo_update(model, optimizer, buffer, cfg, auxiliary_cfg, device):
    total = buffer.steps * buffer.environments
    advantages = buffer.advantages.reshape(-1)
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8)
    returns = buffer.returns.reshape(-1)
    old_log_probability = buffer.log_probability.reshape(-1)
    actions = buffer.actions.reshape(total, -1)
    privileged = buffer.privileged.reshape(total, -1)
    future_target = buffer.future_position.reshape(total, -1)
    risk_target = buffer.collision_risk.reshape(-1)
    confidence_target = buffer.confidence.reshape(-1)
    frames = buffer.frames.reshape(total, *buffer.frames.shape[2:])

    minibatch_size = max(1, total // int(cfg.num_minibatches))
    indices = np.arange(total)
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "auxiliary_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "updates": 0,
    }
    early_stop = False

    for _ in range(int(cfg.epochs)):
        if early_stop:
            break
        np.random.shuffle(indices)
        for start in range(0, total, minibatch_size):
            selected = indices[start:start + minibatch_size]
            frames_batch = torch.from_numpy(frames[selected]).to(
                device, non_blocking=True)
            privileged_batch = privileged[selected].to(device)
            action_batch = actions[selected].to(device)
            old_log_batch = old_log_probability[selected].to(device)
            advantage_batch = advantages[selected].to(device)
            return_batch = returns[selected].to(device)

            outputs = model.evaluate_actions(
                frames_batch, privileged_batch, action_batch,
                cfg.log_std_min, cfg.log_std_max)
            (log_probability, entropy, value, future_prediction,
             risk_logit, confidence_logit, _) = outputs
            log_ratio = log_probability - old_log_batch
            ratio = log_ratio.exp()
            unclipped = ratio * advantage_batch
            clipped = torch.clamp(
                ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            policy_loss = -torch.minimum(
                unclipped, clipped * advantage_batch).mean()
            value_loss = 0.5 * (value - return_batch).square().mean()
            entropy_mean = entropy.mean()
            aux_loss, _, _, _ = auxiliary_loss(
                future_prediction, risk_logit, confidence_logit,
                future_target[selected].to(device),
                risk_target[selected].to(device),
                confidence_target[selected].to(device),
                auxiliary_cfg,
            )
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy_mean
                + cfg.auxiliary_coef * aux_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > cfg.clip_eps).float().mean()
            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy_mean.detach())
            metrics["auxiliary_loss"] += float(aux_loss.detach())
            metrics["approx_kl"] += float(approximate_kl)
            metrics["clip_fraction"] += float(clip_fraction)
            metrics["updates"] += 1

            if (cfg.target_kl is not None
                    and float(approximate_kl) > cfg.target_kl):
                early_stop = True
                break

    count = max(1, metrics.pop("updates"))
    return {key: value / count for key, value in metrics.items()}


def save_checkpoint(path, model, cfg, global_step, hit_rate):
    torch.save({
        "phase": 3,
        "model": model.state_dict(),
        "model_config": dict(cfg.end_to_end.model),
        "render_config": dict(cfg.end_to_end.render),
        "action_config": dict(cfg.end_to_end.action),
        "label_config": dict(cfg.end_to_end.labels),
        "safety_config": dict(cfg.end_to_end.safety),
        "global_step": global_step,
        "hit_rate": hit_rate,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--bc-init", default=None)
    parser.add_argument("--out", default="checkpoints/e2e_rl.pt")
    parser.add_argument("--logdir", default="runs/e2e_ppo")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="mixed")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ppo_cfg = cfg.end_to_end.ppo
    num_envs = args.num_envs or ppo_cfg.num_envs
    rollout_steps = args.rollout_steps or ppo_cfg.rollout_steps
    total_steps = args.total_steps or ppo_cfg.total_steps
    if total_steps < num_envs * rollout_steps:
        raise ValueError("total_steps must cover at least one rollout")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    environments = VecEndToEndInterceptEnv(
        num_envs, cfg, mode=args.mode, seed=args.seed)
    model = EndToEndActorCritic(cfg.end_to_end.model).to(device)
    if args.bc_init:
        checkpoint = torch.load(args.bc_init, map_location=device)
        if checkpoint.get("phase") != 3:
            raise ValueError("--bc-init is not an end-to-end checkpoint")
        for checkpoint_key, current in (
            ("model_config", dict(cfg.end_to_end.model)),
            ("render_config", dict(cfg.end_to_end.render)),
            ("action_config", dict(cfg.end_to_end.action)),
            ("label_config", dict(cfg.end_to_end.labels)),
        ):
            stored = checkpoint.get(checkpoint_key)
            if stored is not None and stored != current:
                raise ValueError(
                    f"BC checkpoint {checkpoint_key} differs from --config")
        result = model.load_state_dict(checkpoint["model"], strict=True)
        print(
            f"loaded BC checkpoint {args.bc_init}; "
            f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ppo_cfg.learning_rate, weight_decay=1e-4)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.logdir)

    frames, training_info, _ = environments.reset()
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    recent_rewards = deque(maxlen=200)
    recent_outcomes = deque(maxlen=200)
    best_hit_rate = -1.0
    global_step = 0
    updates = total_steps // (num_envs * rollout_steps)
    started = time.time()

    for update in range(1, updates + 1):
        buffer = ImageRolloutBuffer(
            rollout_steps, num_envs, frames.shape[1:],
            cfg.end_to_end.model.critic_dim, 4)

        model.eval()
        for _ in range(rollout_steps):
            frame_tensor = torch.from_numpy(frames).to(
                device, non_blocking=True)
            privileged_tensor = torch.from_numpy(
                training_info["critic_obs"]).to(device)
            outputs = model.act(
                frame_tensor, privileged_tensor, False,
                ppo_cfg.log_std_min, ppo_cfg.log_std_max)
            actions, log_probability, values = outputs[:3]
            next_frames, rewards, dones, next_training_info, infos = (
                environments.step(actions.cpu().numpy()))
            buffer.add(
                frames, training_info, actions, log_probability,
                rewards, dones, values)

            episode_rewards += rewards
            for index, info in enumerate(infos):
                if "final" in info:
                    recent_rewards.append(float(episode_rewards[index]))
                    recent_outcomes.append(info["final"]["outcome"])
                    episode_rewards[index] = 0.0
            frames = next_frames
            training_info = next_training_info
            global_step += num_envs

        with torch.no_grad():
            last_value = model.critic(torch.from_numpy(
                training_info["critic_obs"]).to(device))
        buffer.compute_gae(
            last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        # Keep BatchNorm/dropout in inference mode during PPO replay.  Gradients
        # still flow in eval mode, while the policy density remains identical to
        # the one used to collect the rollout.
        model.eval()
        metrics = ppo_update(
            model, optimizer, buffer, ppo_cfg,
            cfg.end_to_end.auxiliary, device)

        hit_rate = (
            sum(outcome == "hit" for outcome in recent_outcomes)
            / len(recent_outcomes)
            if recent_outcomes else 0.0
        )
        mean_reward = (
            float(np.mean(recent_rewards)) if recent_rewards else 0.0)
        writer.add_scalar("rollout/hit_rate", hit_rate, global_step)
        writer.add_scalar("rollout/episode_reward", mean_reward, global_step)
        writer.add_scalar(
            "train/log_std", float(model.actor.log_std.detach().mean()), global_step)
        for name, value in metrics.items():
            writer.add_scalar(f"train/{name}", value, global_step)

        if update % 5 == 0 or update == updates:
            speed = global_step / max(time.time() - started, 1e-6)
            print(
                f"update {update:4d}/{updates} step={global_step:,} "
                f"hit={hit_rate:.1%} reward={mean_reward:.2f} "
                f"kl={metrics['approx_kl']:.4f} {speed:,.0f} steps/s")

        if len(recent_outcomes) >= 50 and hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            save_checkpoint(output, model, cfg, global_step, hit_rate)
        if update % 25 == 0:
            save_checkpoint(
                output.with_name(output.stem + "_last.pt"),
                model, cfg, global_step, hit_rate)

    if best_hit_rate < 0:
        save_checkpoint(output, model, cfg, global_step, None)
    writer.close()
    environments.close()
    print(f"end-to-end PPO complete; saved {output}")


if __name__ == "__main__":
    main()
