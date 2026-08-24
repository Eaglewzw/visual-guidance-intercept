"""Validate that two full Gazebo camera frames survive the 640 letterbox path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from aerointercept.gazebo.client import GazeboBridgeClient
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.process import maybe_launch
from aerointercept.gazebo.protocol import image_from_snapshot
from aerointercept.gazebo.task_logic import camera_target_yaw_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--socket", default=None)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--interval-seconds", type=float, default=0.25)
    parser.add_argument(
        "--visual-max-distance", type=float, default=12.0,
        help="wait for the moving target to be close enough for visual inspection; <=0 disables",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mode", default="circle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_gazebo_config(args.config)
    socket_path = args.socket or cfg.gazebo.bridge.socket
    stack = maybe_launch(args, socket_path)
    client = GazeboBridgeClient(socket_path, float(cfg.gazebo.bridge.timeout_s))
    try:
        status = client.connect(float(cfg.gazebo.bridge.startup_timeout_s))
        reset_command = client.reset(
            cfg.gazebo.task.reset_position_ned,
            float(cfg.gazebo.task.reset_yaw),
            look_at_target=True,
        )
        alignment_deadline = time.monotonic() + float(cfg.gazebo.task.reset_timeout_s)
        alignment_snapshot = None
        sequence = -1
        yaw_error = float("inf")
        while time.monotonic() < alignment_deadline:
            alignment_snapshot = client.snapshot(sequence, timeout=3.0)
            sequence = int(alignment_snapshot["sequence"])
            _, _, yaw_error = camera_target_yaw_geometry(
                alignment_snapshot["interceptor_position"],
                alignment_snapshot["target_position"],
                float(alignment_snapshot["interceptor_yaw"]),
                float(cfg.gazebo.camera.mount_yaw_offset_rad),
            )
            interceptor_position = np.asarray(
                alignment_snapshot["interceptor_position"], dtype=np.float64
            )
            interceptor_velocity = np.asarray(
                alignment_snapshot["interceptor_velocity"], dtype=np.float64
            )
            target_position = np.asarray(
                alignment_snapshot["target_position"], dtype=np.float64
            )
            target_status = alignment_snapshot.get("target_vehicle_status")
            target_ready = (
                -float(target_position[2]) >= float(cfg.gazebo.task.target_minimum_altitude)
                and (
                    target_status is None
                    or (target_status.get("armed") and target_status.get("offboard"))
                )
            )
            interceptor_ready = (
                np.linalg.norm(
                    interceptor_position
                    - np.asarray(cfg.gazebo.task.reset_position_ned, dtype=np.float64)
                ) <= float(cfg.gazebo.task.reset_tolerance_m)
                and np.linalg.norm(interceptor_velocity)
                <= float(cfg.gazebo.task.reset_speed_tolerance)
            )
            visual_distance_ready = (
                args.visual_max_distance <= 0.0
                or np.linalg.norm(target_position - interceptor_position)
                <= args.visual_max_distance
            )
            if (
                target_ready
                and interceptor_ready
                and visual_distance_ready
                and abs(yaw_error) <= float(cfg.gazebo.task.reset_yaw_tolerance_rad)
            ):
                break
        else:
            raise TimeoutError(
                f"camera did not physically yaw toward target: last_error={yaw_error:.4f} rad"
            )
        time.sleep(args.warmup_seconds)
        snapshots = []
        images = []
        for _ in range(args.frames):
            snapshot = client.snapshot(sequence, timeout=5.0)
            sequence = int(snapshot["sequence"])
            image = image_from_snapshot(snapshot)
            snapshots.append(snapshot)
            images.append(image)
            if args.interval_seconds > 0.0:
                time.sleep(args.interval_seconds)
        array = np.stack(images)
        if array.shape != (args.frames, 3, 640, 640) or array.dtype != np.uint8:
            raise AssertionError(f"invalid camera tensor: {array.shape} {array.dtype}")
        if not np.isfinite(array).all() or int(array.max()) <= int(array.min()):
            raise AssertionError("camera pixels are invalid or constant")
        differences = np.mean(np.abs(array[1:].astype(np.int16) - array[:-1].astype(np.int16)), axis=(1, 2, 3))
        if not np.any(differences > 0.01):
            raise AssertionError("new Gazebo camera sequences contain repeated static buffers")
        if np.shares_memory(array[0], array[1]):
            raise AssertionError("successive frames alias the same image buffer")
        final = snapshots[-1]
        relative = np.asarray(final["target_position"], dtype=np.float64) - np.asarray(
            final["interceptor_position"], dtype=np.float64
        )
        yaw = float(final["interceptor_yaw"])
        camera_yaw = yaw + float(cfg.gazebo.camera.mount_yaw_offset_rad)
        cosine, sine = np.cos(camera_yaw), np.sin(camera_yaw)
        relative_body = np.array([
            cosine * relative[0] + sine * relative[1],
            -sine * relative[0] + cosine * relative[1],
            relative[2],
        ])
        horizontal_angle = float(np.arctan2(relative_body[1], relative_body[0]))
        vertical_angle = float(np.arctan2(
            relative_body[2], max(np.hypot(*relative_body[:2]), 1.0e-9)
        ))
        horizontal_fov = float(cfg.gazebo.camera.horizontal_fov)
        vertical_fov = float(cfg.gazebo.camera.vertical_fov)
        target_in_fov = bool(
            relative_body[0] > 0.0
            and abs(horizontal_angle) <= horizontal_fov / 2.0
            and abs(vertical_angle) <= vertical_fov / 2.0
        )
        if not target_in_fov:
            raise AssertionError(
                "physical target is outside the mounted camera FOV after look-at reset: "
                f"horizontal={horizontal_angle:.4f} vertical={vertical_angle:.4f}"
            )
        transform = final["camera_metadata"]
        projected_pixel = [
            (0.5 + horizontal_angle / horizontal_fov) * float(transform["resized_width"]),
            float(transform["pad_top"])
            + (0.5 + vertical_angle / vertical_fov) * float(transform["resized_height"]),
        ]
        if args.output:
            import cv2
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.transpose(array[-1], (1, 2, 0))[..., ::-1])
        report = {
            "backend": status["backend"],
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "minimum": int(array.min()),
            "maximum": int(array.max()),
            "std": float(array.std()),
            "mean_frame_difference": float(differences.mean()),
            "sequences": [int(item["sequence"]) for item in snapshots],
            "camera": snapshots[-1]["camera_metadata"],
            "reset_command": reset_command,
            "target_bearing_rad": float(snapshots[-1]["target_bearing"]),
            "interceptor_yaw_rad": float(snapshots[-1]["interceptor_yaw"]),
            "camera_target_yaw_error_rad": float(
                snapshots[-1]["camera_target_yaw_error"]
            ),
            "camera_look_at_active": bool(snapshots[-1]["camera_look_at_active"]),
            "target_altitude_m": -float(final["target_position"][2]),
            "target_distance_m": float(np.linalg.norm(relative)),
            "interceptor_position_ned": final["interceptor_position"],
            "target_position_ned": final["target_position"],
            "target_horizontal_angle_rad": horizontal_angle,
            "target_vertical_angle_rad": vertical_angle,
            "target_in_camera_fov": target_in_fov,
            "target_projected_pixel_640_from_odometry": projected_pixel,
            "output": args.output,
        }
        print("AEROINTERCEPT_GAZEBO_CAMERA=" + json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        client.close()
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    main()
