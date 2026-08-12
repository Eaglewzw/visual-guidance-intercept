"""Full-frame contracts: pixel-only observation, direct action, model and data."""
import json
import math

import numpy as np
import torch

from aerointercept.config import DotDict, load_config
from aerointercept.end_to_end.actions import (
    body_to_ned,
    decode_action,
    encode_velocity_command,
)
from aerointercept.end_to_end.data import EpisodeSequenceDataset
from aerointercept.end_to_end.distributions import (
    squashed_normal_log_probability,
    squashed_normal_sample,
)
from aerointercept.end_to_end.environment import EndToEndInterceptEnv
from aerointercept.end_to_end.policy import EndToEndActorCritic
from aerointercept.end_to_end.renderer import FullFrameRenderer
from aerointercept.end_to_end.runtime import EndToEndRuntime
from aerointercept.environments import InterceptEnv
from aerointercept.training.train_e2e_bc import compute_losses
from aerointercept.training.train_e2e_ppo import ImageRolloutBuffer, ppo_update


def test_direct_action_roundtrip_and_coordinate_rotation():
    yaw = 0.73
    velocity_ned = np.array([3.0, -1.5, 0.7])
    action = encode_velocity_command(
        velocity_ned, 0.4, yaw, velocity_max=8.0, yaw_rate_max=1.0)
    command = decode_action(
        action, yaw, velocity_max=8.0, yaw_rate_max=1.0)
    assert np.allclose(command.ned_velocity, velocity_ned, atol=1e-6)
    assert abs(command.yaw_rate - 0.4) < 1e-6
    assert np.allclose(body_to_ned([2.0, 0.0, 0.0], math.pi / 2),
                       [0.0, 2.0, 0.0], atol=1e-9)


def test_direct_action_enforces_vector_speed_limit():
    command = decode_action(
        np.ones(4), 0.0, velocity_max=5.0, yaw_rate_max=1.0)
    assert np.linalg.norm(command.body_velocity) <= 5.0 + 1e-9


def test_full_frame_renderer_places_target_without_crop():
    cfg = load_config()
    rng = np.random.default_rng(2)
    renderer = FullFrameRenderer(
        cfg.camera, cfg.end_to_end.render, rng)
    result = renderer.render(np.array([12.0, 0.0, 0.0]), 0.0, 0.0, 0.0)
    assert result.image.shape == (108, 192, 3)
    assert result.image.dtype == np.uint8
    assert result.visible
    assert np.allclose(result.center_normalized, [0.0, 0.0], atol=0.03)
    assert result.image.std() > 1.0


def test_end_to_end_environment_exposes_pixels_only_and_times_out():
    cfg = load_config()
    cfg["env"]["episode_max_steps"] = 1
    cfg["camera"]["latency_frames_max"] = 0
    cfg["camera"]["miss_base"] = 0.0
    cfg["camera"]["miss_small_scale"] = 0.0
    env = EndToEndInterceptEnv(cfg, mode="random_walk", seed=7)
    observation, info = env.reset(seed=7)
    assert set(observation) == {"frames"}
    assert observation["frames"].shape == (2, 3, 108, 192)
    assert info["critic_obs"].shape == (15,)
    assert info["teacher_action"].shape == (4,)
    _, _, terminated, truncated, final = env.step(info["teacher_action"])
    assert not terminated and truncated
    assert final["outcome"] == "timeout"
    env.close()


def test_learned_guidance_environment_keeps_public_contract():
    env = InterceptEnv(seed=11)
    observation, info = env.reset(seed=11)
    assert observation.shape == (15,)
    assert info["teacher_action"].shape == (4,)
    next_observation, reward, terminated, truncated, _ = env.step(
        info["teacher_action"])
    assert next_observation.shape == (15,)
    assert np.isfinite(reward)
    assert not (terminated and truncated)


def test_workflows_share_seeded_physical_initial_conditions():
    cfg = load_config()
    learned_guidance = InterceptEnv(cfg, mode="circle", seed=19)
    end_to_end = EndToEndInterceptEnv(cfg, mode="circle", seed=19)
    learned_guidance.reset(seed=19)
    end_to_end.reset(seed=19)
    assert np.allclose(learned_guidance.dynamics.pos, end_to_end.dynamics.pos)
    assert np.allclose(learned_guidance.target_pos, end_to_end.target_pos)
    assert learned_guidance.dynamics.yaw == end_to_end.dynamics.yaw


def test_squashed_distribution_probability_matches_sample():
    torch.manual_seed(5)
    latent_mean = torch.randn(8, 4) * 0.2
    log_std = torch.full((4,), -0.8)
    action, sampled_log_probability = squashed_normal_sample(
        latent_mean, log_std)
    recomputed = squashed_normal_log_probability(
        action, latent_mean, log_std)
    assert torch.allclose(sampled_log_probability, recomputed, atol=2e-5)
    assert torch.all(action.abs() < 1.0)


def test_image_policy_shapes_gradients_and_torchscript():
    cfg = load_config()
    model = EndToEndActorCritic(cfg.end_to_end.model)
    frames = torch.randint(
        0, 256, (2, 2, 3, 108, 192), dtype=torch.uint8)
    action, future, risk, confidence, attention = model.actor(frames)
    assert action.shape == (2, 4)
    assert future.shape == (2, 3)
    assert risk.shape == confidence.shape == (2,)
    assert attention.shape[0] == 2
    assert torch.all(action.abs() <= 1.0)

    batch = {
        "frames": frames.unsqueeze(1),
        "actions": torch.zeros(2, 1, 4),
        "future_position": torch.zeros(2, 1, 3),
        "collision_risk": torch.zeros(2, 1),
        "confidence": torch.ones(2, 1),
        "mask": torch.ones(2, 1),
    }
    losses = compute_losses(
        model.actor, batch, "cpu", cfg.end_to_end.auxiliary)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert any(parameter.grad is not None for parameter in model.actor.parameters())

    model.actor.eval()
    scripted = torch.jit.script(model.actor)
    scripted_outputs = scripted(frames)
    assert [tuple(value.shape) for value in scripted_outputs[:4]] == [
        (2, 4), (2, 3), (2,), (2,)]


def _write_episode(path, value, length=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.full((length, 3, 4, 6), value, dtype=np.uint8),
        actions=np.zeros((length, 4), dtype=np.float32),
        future_position=np.zeros((length, 3), dtype=np.float32),
        collision_risk=np.zeros(length, dtype=np.float32),
        confidence=np.ones(length, dtype=np.float32),
    )


def test_episode_dataset_history_never_crosses_reset(tmp_path):
    first = tmp_path / "episodes" / "episode_000000.npz"
    second = tmp_path / "episodes" / "episode_000001.npz"
    _write_episode(first, 10, length=2)
    _write_episode(second, 200, length=2)
    dataset = EpisodeSequenceDataset(
        [first, second], sequence_length=2, history_frames=2)
    first_second_episode_window = dataset[1]
    first_history = first_second_episode_window["frames"][0]
    assert np.all(first_history[0] == 200)
    assert np.all(first_history[1] == 200)


def test_exported_runtime_maintains_frame_history(tmp_path):
    cfg = load_config()
    model = EndToEndActorCritic(cfg.end_to_end.model).actor.eval()
    model_path = tmp_path / "policy.pt"
    torch.jit.script(model).save(str(model_path))
    metadata = {
        "phase": 3,
        "input": {
            "image_width": 192,
            "image_height": 108,
            "history_frames": 2,
        },
        "action_protocol": {
            "velocity_max": 8.0,
            "yaw_rate_max": 1.0,
        },
        "safety": {"confidence_threshold": 0.0},
    }
    metadata_path = tmp_path / "policy_meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    runtime = EndToEndRuntime(model_path, metadata_path)
    result = runtime.step(
        np.zeros((216, 384, 3), dtype=np.uint8), yaw=0.2)
    assert result.command is not None
    assert not result.fallback_required
    assert result.attention.ndim == 2


def test_image_ppo_update_smoke():
    cfg = load_config()
    model = EndToEndActorCritic(cfg.end_to_end.model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    frame_shape = (2, 3, 108, 192)
    buffer = ImageRolloutBuffer(2, 1, frame_shape, 15, 4)
    training_info = {
        "critic_obs": np.zeros((1, 15), dtype=np.float32),
        "future_position": np.zeros((1, 3), dtype=np.float32),
        "collision_risk": np.zeros(1, dtype=np.float32),
        "confidence": np.ones(1, dtype=np.float32),
    }
    for _ in range(2):
        frames = np.random.default_rng(4).integers(
            0, 256, (1, *frame_shape), dtype=np.uint8)
        with torch.no_grad():
            action, log_probability, value = model.act(
                torch.from_numpy(frames), torch.zeros(1, 15))[:3]
        buffer.add(
            frames, training_info, action, log_probability,
            np.ones(1, dtype=np.float32), np.zeros(1, dtype=np.float32), value)
    buffer.compute_gae(torch.zeros(1), gamma=0.99, gae_lambda=0.95)
    ppo_cfg = DotDict({
        "num_minibatches": 1,
        "epochs": 1,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.001,
        "auxiliary_coef": 0.1,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "log_std_min": -2.5,
        "log_std_max": -0.1,
    })
    metrics = ppo_update(
        model, optimizer, buffer, ppo_cfg,
        cfg.end_to_end.auxiliary, torch.device("cpu"))
    assert all(np.isfinite(value) for value in metrics.values())
