"""Fast Gazebo protocol tests; real simulator acceptance lives in smoke scripts."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from aerointercept.end_to_end.actions import decode_action
from aerointercept.end_to_end.policy import EndToEndActor
from aerointercept.gazebo.camera import decode_ros_image, letterbox_rgb
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.environment import GazeboInterceptEnv
from aerointercept.gazebo.protocol import image_from_snapshot, receive_packet, send_packet
from aerointercept.gazebo.task_logic import (
    CRITIC_DIM,
    build_training_info,
    camera_target_yaw_geometry,
    compute_reward,
    segment_minimum_distance,
    target_visibility,
    termination_flags,
)


class FakeBridgeClient:
    def __init__(self, _path, timeout=1.0):
        self.timeout = timeout
        self.sequence = 0
        self.position = np.array([0.0, 0.0, -6.0])
        self.velocity = np.zeros(3)
        self.yaw = 0.0

    def connect(self, _wait):
        return {"backend": "gazebo_px4", "ready": True}

    def ping(self):
        return {"backend": "gazebo_px4", "ready": True}

    def reset(self, position_ned, yaw=0.0, *, look_at_target=True):
        self.position = np.asarray(position_ned, dtype=float)
        self.velocity.fill(0.0)
        self.look_at_target = bool(look_at_target)
        # Default x500_depth camera is -pi/2 from PX4 body-forward.
        self.yaw = np.pi / 2.0 if look_at_target else float(yaw)
        return {"mode": "position"}

    def action(self, values):
        action = np.asarray(values)
        decoded = decode_action(action, self.yaw, velocity_max=8.0, yaw_rate_max=1.0)
        self.velocity = decoded.ned_velocity
        self.position += self.velocity * 0.05
        return {"mode": "velocity", "ned_velocity": decoded.ned_velocity.tolist()}

    def snapshot(self, after_sequence=-1, timeout=None):
        assert self.sequence <= after_sequence + 1 or after_sequence == -1
        self.sequence = max(self.sequence + 1, after_sequence + 1)
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[..., 0] = self.sequence % 255
        image[200:260, 300:360, 1] = 220
        return {
            "sequence": self.sequence,
            "image": image.tobytes(),
            "image_shape": (640, 640, 3),
            "image_dtype": "uint8",
            "camera_metadata": {"name": "full_frame_letterbox_v1"},
            "interceptor_position": self.position.tolist(),
            "interceptor_velocity": self.velocity.tolist(),
            "interceptor_yaw": self.yaw,
            "target_position": [10.0, 0.0, -6.0],
            "target_velocity": [0.0, 1.0, 0.0],
        }

    def close(self):
        pass


def test_gazebo_overlay_preserves_legacy_and_sets_actor_contract():
    cfg = load_gazebo_config()
    assert cfg.camera.image_width == 1920
    assert cfg.end_to_end.render.source_image_width == 1920
    assert cfg.end_to_end.render.image_width == cfg.end_to_end.render.image_height == 640
    assert cfg.end_to_end.render.channel_order == "RGB"
    assert cfg.end_to_end.render.transform == "full_frame_letterbox_v1"
    assert cfg.end_to_end.model.encoder_chunk_size == 4
    assert cfg.gazebo.target.initial_model_separation_m == 10.0


def test_camera_letterbox_preserves_full_16_by_9_frame():
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)
    source[:, :10] = [255, 0, 0]
    source[:, -10:] = [0, 255, 0]
    output, metadata = letterbox_rgb(source)
    assert output.shape == (640, 640, 3)
    assert output.dtype == np.uint8
    assert metadata["resized_width"] == 640
    assert metadata["resized_height"] == 360
    assert metadata["pad_top"] == metadata["pad_bottom"] == 140
    assert output[320, 0, 0] > 0
    assert output[320, -1, 1] > 0
    assert np.all(output[:140] == 0) and np.all(output[500:] == 0)


def test_ros_bgr_stride_decode_outputs_rgb():
    row = bytes([1, 2, 3, 4, 5, 6, 99, 99])
    image = decode_ros_image(row, width=2, height=1, step=8, encoding="bgr8")
    np.testing.assert_array_equal(image, [[[3, 2, 1], [6, 5, 4]]])


def test_socket_protocol_round_trip_supports_camera_payload():
    class MemoryConnection:
        def __init__(self):
            self.buffer = bytearray()

        def sendall(self, data):
            self.buffer.extend(data)

        def recv(self, size):
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    connection = MemoryConnection()
    value = {"image": bytes(640 * 640 * 3), "sequence": 3}
    send_packet(connection, value)
    received = receive_packet(connection)
    assert received["sequence"] == 3
    assert len(received["image"]) == 640 * 640 * 3


def test_image_snapshot_returns_independent_chw_buffers():
    raw = np.arange(640 * 640 * 3, dtype=np.uint8).tobytes()
    snapshot = {"image": raw, "image_shape": (640, 640, 3), "image_dtype": "uint8"}
    first = image_from_snapshot(snapshot)
    second = image_from_snapshot(snapshot)
    assert first.shape == (3, 640, 640) and first.dtype == np.uint8
    assert not np.shares_memory(first, second)


def test_environment_reset_uses_two_new_frames_and_actor_gets_only_rgb():
    cfg = load_gazebo_config()
    environment = GazeboInterceptEnv(cfg, "/fake", client_factory=FakeBridgeClient)
    frames, training, info = environment.reset()
    assert frames.shape == (2, 3, 640, 640)
    assert not np.array_equal(frames[0], frames[1])
    assert training["critic_obs"].shape == (CRITIC_DIM,)
    assert info["backend"] == "gazebo_px4"
    assert info["target_distance_m"] == pytest.approx(10.0)
    assert info["requested_target_distance_m"] == pytest.approx(10.0)
    next_frames, reward, terminated, truncated, training, _ = environment.step(
        np.zeros(4, dtype=np.float32)
    )
    assert next_frames.shape == frames.shape and np.isfinite(reward)
    assert not terminated and not truncated
    assert training["future_position"].shape == (3,)


def test_actor_has_no_privileged_argument_or_critic_module():
    actor = EndToEndActor(load_gazebo_config().end_to_end.model).eval()
    assert tuple(inspect.signature(actor.forward).parameters) == ("frames",)
    assert not any(name.startswith("critic") for name, _ in actor.named_modules())
    with pytest.raises(TypeError):
        actor(torch.zeros(1, 2, 3, 32, 32, dtype=torch.uint8), torch.zeros(1, 15))


def test_critic_and_auxiliary_protocol_uses_ned_truth_only_for_training():
    cfg = load_gazebo_config()
    state = {
        "interceptor_position": [0.0, 0.0, -6.0],
        "interceptor_velocity": [1.0, 0.0, 0.0],
        "target_position": [10.0, 2.0, -5.0],
        "target_velocity": [0.0, 1.0, 0.0],
        "interceptor_yaw": 0.2,
    }
    training = build_training_info(state, cfg.end_to_end.labels, True)
    assert training["critic_obs"].shape == (15,)
    assert set(training) == {"critic_obs", "future_position", "collision_risk", "confidence"}
    assert all(np.isfinite(value).all() for value in training.values())


def test_action_is_norm_limited_and_body_to_ned_is_external():
    command = decode_action([1, 1, 1, 1], 0.5, velocity_max=8.0, yaw_rate_max=1.0)
    assert np.linalg.norm(command.body_velocity) == pytest.approx(8.0)
    assert np.linalg.norm(command.ned_velocity) == pytest.approx(8.0)
    assert command.yaw_rate == pytest.approx(1.0)


def test_reset_camera_look_at_geometry_uses_target_bearing_not_fixed_yaw():
    bearing, desired_yaw, error = camera_target_yaw_geometry(
        [0.0, 0.0, -6.0], [10.0, 10.0, -5.0], 0.0, 0.0
    )
    assert bearing == pytest.approx(np.pi / 4.0)
    assert desired_yaw == pytest.approx(np.pi / 4.0)
    assert error == pytest.approx(np.pi / 4.0)
    _, _, aligned_error = camera_target_yaw_geometry(
        [0.0, 0.0, -6.0], [10.0, 10.0, -5.0], desired_yaw, 0.0
    )
    assert aligned_error == pytest.approx(0.0)
    _, mounted_desired_yaw, mounted_error = camera_target_yaw_geometry(
        [0.0, 0.0, -6.0], [10.0, 0.0, -5.0], np.pi / 2.0, -np.pi / 2.0
    )
    assert mounted_desired_yaw == pytest.approx(np.pi / 2.0)
    assert mounted_error == pytest.approx(0.0)


def test_visibility_segment_reward_and_all_termination_paths():
    cfg = load_gazebo_config()
    visible, center = target_visibility([10, 0, 0], 1.204, 0.753139)
    assert visible and center == pytest.approx(0.0)
    minimum = segment_minimum_distance([0, 0, 0], [2, 0, 0], [1, .2, 0], [1, .2, 0])
    assert minimum == pytest.approx(0.2)
    flags = termination_flags(
        step_minimum_distance=minimum, lost_count=0,
        interceptor_position=[0, 0, -6], invalid=False,
        episode_step=1, cfg=cfg.gazebo.task,
    )
    assert flags["hit"] and flags["terminated"]
    reward, terms = compute_reward(
        previous_distance=2.0, distance=1.0, visible=True, center_error=0.0,
        action=np.zeros(4), previous_action=np.zeros(4), flags=flags,
        cfg=cfg.gazebo.rewards,
    )
    assert set(terms) == {
        "close", "hit", "time", "fov_center", "lost", "smooth",
        "ground", "invalid", "out_of_bounds", "timeout",
    }
    assert reward > 50.0


@pytest.mark.parametrize(
    "position,lost,invalid,step,expected",
    [
        ([0, 0, -6], 30, False, 1, "fov_lost"),
        ([0, 0, -.1], 0, False, 1, "ground"),
        ([0, 0, -6], 0, True, 1, "invalid"),
        ([36, 0, -6], 0, False, 1, "out_of_bounds"),
        ([0, 0, -6], 0, False, 600, "timed_out"),
    ],
)
def test_independent_termination_conditions(position, lost, invalid, step, expected):
    cfg = load_gazebo_config()
    flags = termination_flags(
        step_minimum_distance=10.0, lost_count=lost,
        interceptor_position=position, invalid=invalid,
        episode_step=step, cfg=cfg.gazebo.task,
    )
    assert flags[expected]
