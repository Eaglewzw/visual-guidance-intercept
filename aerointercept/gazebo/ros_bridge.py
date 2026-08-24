#!/usr/bin/env python3
"""ROS 2 / PX4 bridge for the Conda Gazebo training environment.

Run this file with Ubuntu's ``/usr/bin/python3`` after sourcing ROS Humble and
the read-only ``ros2_ws/install`` overlay.  It must not run inside the Python
3.12 Conda process because Humble's rclpy extension targets CPython 3.10.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from px4_msgs.msg import VehicleOdometry, VehicleStatus
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image

from aerointercept.end_to_end.actions import decode_action
from aerointercept.gazebo.camera import decode_ros_image, letterbox_rgb
from aerointercept.gazebo.protocol import (
    PROTOCOL_VERSION,
    receive_packet,
    send_packet,
)
from aerointercept.gazebo.task_logic import camera_target_yaw_geometry


def _finite_vector(values, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def _yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = quaternion
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class GazeboRosBridge(Node):
    def __init__(self, args):
        super().__init__("aerointercept_gazebo_bridge")
        self.args = args
        self._condition = threading.Condition()
        self._image_bytes = None
        self._image_metadata = None
        self._sequence = 0
        self._interceptor = None
        self._target = None
        self._vehicle_status = None
        self._target_status = None
        self._mode = "position"
        self._position = np.asarray(args.reset_position_ned, dtype=np.float64)
        self._yaw = float(args.reset_yaw)
        # Point the mounted camera at the target during initial hover as well
        # as every explicit reset.  The first Actor action disables this.
        self._look_at_target = True
        self._velocity = np.zeros(3, dtype=np.float64)
        self._yaw_rate = 0.0
        self._setpoint_count = 0
        self._last_reported_flight_state = None
        self._last_reported_target_state = None
        self._stop = threading.Event()
        self._socket_path = Path(args.socket)

        self.create_subscription(Image, args.image_topic, self._camera_callback, qos_profile_sensor_data)
        self.create_subscription(
            VehicleOdometry,
            "/px4_1/fmu/out/vehicle_odometry",
            self._interceptor_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleOdometry,
            "/px4_2/fmu/out/vehicle_odometry",
            self._target_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleStatus,
            "/px4_1/fmu/out/vehicle_status",
            self._status_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            VehicleStatus,
            "/px4_2/fmu/out/vehicle_status",
            self._target_status_callback,
            qos_profile_sensor_data,
        )
        self._offboard_publisher = self.create_publisher(
            OffboardControlMode, "/px4_1/fmu/in/offboard_control_mode", 10
        )
        self._setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, "/px4_1/fmu/in/trajectory_setpoint", 10
        )
        self._command_publisher = self.create_publisher(
            VehicleCommand, "/px4_1/fmu/in/vehicle_command", 10
        )
        self._target_command_publisher = self.create_publisher(
            VehicleCommand, "/px4_2/fmu/in/vehicle_command", 10
        )
        self.create_timer(1.0 / float(args.control_rate), self._control_timer)
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()
        self.get_logger().info(
            f"bridge socket={self._socket_path} image={args.image_topic}; "
            "camera transform=full_frame_letterbox_v1"
        )

    def _timestamp(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def _camera_callback(self, message: Image) -> None:
        try:
            rgb = decode_ros_image(
                bytes(message.data), message.width, message.height,
                message.step, message.encoding,
            )
            output, transform = letterbox_rgb(rgb)
        except Exception as exc:
            self.get_logger().error(f"camera conversion rejected a frame: {exc}")
            return
        with self._condition:
            self._image_bytes = output.tobytes()
            self._image_metadata = {
                **transform,
                "source": "Gazebo Harmonic sensor via ros_gz_bridge",
                "ros_topic": self.args.image_topic,
                "source_encoding": message.encoding,
                "output_width": 640,
                "output_height": 640,
                "output_dtype": "uint8",
            }
            self._sequence += 1
            self._condition.notify_all()

    @staticmethod
    def _odometry(message: VehicleOdometry, origin_ned: np.ndarray) -> dict:
        position = np.asarray(message.position, dtype=np.float64) + origin_ned
        velocity = np.asarray(message.velocity, dtype=np.float64)
        quaternion = np.asarray(message.q, dtype=np.float64)
        return {
            "position": position,
            "velocity": velocity,
            "quaternion": quaternion,
            "yaw": _yaw_from_wxyz(quaternion),
            "timestamp": int(message.timestamp),
        }

    def _interceptor_callback(self, message: VehicleOdometry) -> None:
        with self._condition:
            self._interceptor = self._odometry(message, np.zeros(3))
            self._condition.notify_all()

    def _target_callback(self, message: VehicleOdometry) -> None:
        with self._condition:
            self._target = self._odometry(
                message, np.asarray(self.args.target_origin_ned, dtype=np.float64)
            )
            self._condition.notify_all()

    def _status_callback(self, message: VehicleStatus) -> None:
        with self._condition:
            self._vehicle_status = {
                "armed": message.arming_state == VehicleStatus.ARMING_STATE_ARMED,
                "offboard": message.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                "preflight_ok": bool(message.pre_flight_checks_pass),
                "failsafe": bool(message.failsafe),
            }
            self._condition.notify_all()

    def _target_status_callback(self, message: VehicleStatus) -> None:
        with self._condition:
            self._target_status = {
                "armed": message.arming_state == VehicleStatus.ARMING_STATE_ARMED,
                "offboard": message.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD,
                "preflight_ok": bool(message.pre_flight_checks_pass),
                "failsafe": bool(message.failsafe),
            }
            self._condition.notify_all()

    def _publish_vehicle_command(self, command: int, param1=0.0, param2=0.0) -> None:
        message = VehicleCommand()
        message.timestamp = self._timestamp()
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.command = int(command)
        message.target_system = 2
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self._command_publisher.publish(message)

    def _publish_target_vehicle_command(self, command: int, param1=0.0, param2=0.0) -> None:
        message = VehicleCommand()
        message.timestamp = self._timestamp()
        message.param1 = float(param1)
        message.param2 = float(param2)
        message.command = int(command)
        message.target_system = 3
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self._target_command_publisher.publish(message)

    def _control_timer(self) -> None:
        with self._condition:
            mode = self._mode
            position = self._position.copy()
            if (
                mode == "position"
                and self._look_at_target
                and self._interceptor is not None
                and self._target is not None
            ):
                _, self._yaw, _ = camera_target_yaw_geometry(
                    self._interceptor["position"],
                    self._target["position"],
                    self._interceptor["yaw"],
                    self.args.camera_mount_yaw_offset,
                )
            yaw = self._yaw
            velocity = self._velocity.copy()
            yaw_rate = self._yaw_rate
        offboard = OffboardControlMode()
        offboard.timestamp = self._timestamp()
        offboard.position = mode == "position"
        offboard.velocity = mode == "velocity"
        offboard.acceleration = False
        offboard.attitude = False
        offboard.body_rate = False
        self._offboard_publisher.publish(offboard)

        setpoint = TrajectorySetpoint()
        setpoint.timestamp = self._timestamp()
        nan = float("nan")
        if mode == "position":
            setpoint.position = position.astype(np.float32).tolist()
            setpoint.velocity = [nan, nan, nan]
            setpoint.yaw = float(yaw)
            setpoint.yawspeed = nan
        else:
            setpoint.position = [nan, nan, nan]
            setpoint.velocity = velocity.astype(np.float32).tolist()
            setpoint.yaw = nan
            setpoint.yawspeed = float(yaw_rate)
        self._setpoint_publisher.publish(setpoint)

        self._setpoint_count += 1
        if self._setpoint_count >= int(self.args.warmup_setpoints):
            with self._condition:
                status = dict(self._vehicle_status) if self._vehicle_status is not None else None
            armed = bool(status and status["armed"])
            offboard = bool(status and status["offboard"])
            retry_ticks = max(1, int(round(float(self.args.control_rate))))
            if (not armed or not offboard) and (
                self._setpoint_count - int(self.args.warmup_setpoints)
            ) % retry_ticks == 0:
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            flight_state = (armed, offboard, bool(status and status["preflight_ok"]))
            if flight_state != self._last_reported_flight_state:
                self.get_logger().info(
                    f"PX4 state armed={armed} offboard={offboard} "
                    f"preflight_ok={flight_state[2]}"
                )
                self._last_reported_flight_state = flight_state

            with self._condition:
                target_status = dict(self._target_status) if self._target_status is not None else None
            target_armed = bool(target_status and target_status["armed"])
            target_offboard = bool(target_status and target_status["offboard"])
            if (not target_armed or not target_offboard) and (
                self._setpoint_count - int(self.args.warmup_setpoints)
            ) % retry_ticks == 0:
                # Motion setpoints and offboard heartbeat still come from the
                # existing read-only C++ target node. This only retries commands
                # that may have arrived before PX4/DDS discovery was complete.
                self._publish_target_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0
                )
                self._publish_target_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0
                )
            target_flight_state = (
                target_armed,
                target_offboard,
                bool(target_status and target_status["preflight_ok"]),
            )
            if target_flight_state != self._last_reported_target_state:
                self.get_logger().info(
                    f"target PX4 state armed={target_armed} offboard={target_offboard} "
                    f"preflight_ok={target_flight_state[2]}"
                )
                self._last_reported_target_state = target_flight_state

    def _ready(self) -> bool:
        return self._image_bytes is not None and self._interceptor is not None and self._target is not None

    def _snapshot(self, after_sequence: int, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (not self._ready() or self._sequence <= after_sequence) and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = []
                    if self._image_bytes is None:
                        missing.append("camera")
                    if self._interceptor is None:
                        missing.append("interceptor_odometry")
                    if self._target is None:
                        missing.append("target_odometry")
                    raise TimeoutError(
                        f"no newer synchronized snapshot; missing={missing} sequence={self._sequence}"
                    )
                self._condition.wait(min(remaining, 0.25))
            interceptor = self._interceptor
            target = self._target
            target_bearing, desired_yaw, yaw_error = camera_target_yaw_geometry(
                interceptor["position"], target["position"], interceptor["yaw"],
                self.args.camera_mount_yaw_offset,
            )
            return {
                "sequence": self._sequence,
                "image": self._image_bytes,
                "image_shape": (640, 640, 3),
                "image_dtype": "uint8",
                "camera_metadata": dict(self._image_metadata),
                "interceptor_position": interceptor["position"].tolist(),
                "interceptor_velocity": interceptor["velocity"].tolist(),
                "interceptor_yaw": float(interceptor["yaw"]),
                "target_bearing": target_bearing,
                "camera_desired_body_yaw": desired_yaw,
                "camera_target_yaw_error": yaw_error,
                "camera_mount_yaw_offset": float(self.args.camera_mount_yaw_offset),
                "camera_look_at_active": bool(self._look_at_target),
                "target_position": target["position"].tolist(),
                "target_velocity": target["velocity"].tolist(),
                "interceptor_timestamp": interceptor["timestamp"],
                "target_timestamp": target["timestamp"],
                "vehicle_status": dict(self._vehicle_status) if self._vehicle_status else None,
                "target_vehicle_status": dict(self._target_status) if self._target_status else None,
            }

    def _handle(self, message: dict) -> dict:
        if not isinstance(message, dict) or message.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("incompatible Gazebo bridge request")
        command = message.get("command")
        if command == "ping":
            with self._condition:
                return {
                    "ready": self._ready(),
                    "sequence": self._sequence,
                    "backend": "gazebo_px4",
                    "simulator": "Gazebo Harmonic",
                    "physics": "gz-sim physics",
                    "vehicle_status": dict(self._vehicle_status) if self._vehicle_status else None,
                    "target_vehicle_status": dict(self._target_status) if self._target_status else None,
                }
        if command == "snapshot":
            return {"snapshot": self._snapshot(
                int(message.get("after_sequence", -1)),
                float(message.get("timeout", 3.0)),
            )}
        if command == "reset":
            position = _finite_vector(message.get("position"), 3, "reset position")
            yaw = float(message.get("yaw", 0.0))
            look_at_target = bool(message.get("look_at_target", True))
            if not math.isfinite(yaw):
                raise ValueError("reset yaw must be finite")
            with self._condition:
                if look_at_target and self._interceptor is not None and self._target is not None:
                    _, yaw, _ = camera_target_yaw_geometry(
                        self._interceptor["position"], self._target["position"],
                        self._interceptor["yaw"], self.args.camera_mount_yaw_offset,
                    )
                self._mode = "position"
                self._position = position
                self._yaw = yaw
                self._look_at_target = look_at_target
            return {"command": {
                "mode": "position", "position_ned": position.tolist(), "yaw": yaw,
                "look_at_target": look_at_target,
            }}
        if command == "action":
            action = _finite_vector(message.get("action"), 4, "action")
            with self._condition:
                if self._interceptor is None:
                    raise RuntimeError("interceptor odometry is not ready")
                yaw = float(self._interceptor["yaw"])
            decoded = decode_action(
                action,
                yaw,
                velocity_max=float(self.args.velocity_max),
                yaw_rate_max=float(self.args.yaw_rate_max),
            )
            with self._condition:
                self._mode = "velocity"
                self._velocity = decoded.ned_velocity
                self._yaw_rate = decoded.yaw_rate
                self._look_at_target = False
            return {"command": {
                "mode": "velocity",
                "body_velocity": decoded.body_velocity.tolist(),
                "ned_velocity": decoded.ned_velocity.tolist(),
                "yaw_rate": decoded.yaw_rate,
                "yaw_used": yaw,
                "vector_norm_limited": True,
            }}
        raise ValueError(f"unknown command: {command!r}")

    def _serve_client(self, connection: socket.socket) -> None:
        with connection:
            while not self._stop.is_set():
                try:
                    message = receive_packet(connection)
                except EOFError:
                    return
                try:
                    payload = self._handle(message)
                    response = {"ok": True, "protocol": PROTOCOL_VERSION, **payload}
                except Exception as exc:
                    response = {
                        "ok": False,
                        "protocol": PROTOCOL_VERSION,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self.get_logger().error(response["error"])
                send_packet(connection, response)

    def _serve(self) -> None:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._socket_path.unlink(missing_ok=True)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self._socket_path))
                os.chmod(self._socket_path, 0o600)
                server.listen(4)
                server.settimeout(0.5)
                while not self._stop.is_set():
                    try:
                        connection, _ = server.accept()
                    except socket.timeout:
                        continue
                    threading.Thread(
                        target=self._serve_client, args=(connection,), daemon=True
                    ).start()
        finally:
            self._socket_path.unlink(missing_ok=True)

    def destroy_node(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._server_thread.join(timeout=2.0)
        return super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default="/tmp/aerointercept_gazebo/bridge.sock")
    parser.add_argument("--image-topic", default="/camera/image")
    parser.add_argument("--target-origin-ned", nargs=3, type=float, default=(10.0, 0.0, 0.0))
    parser.add_argument("--reset-position-ned", nargs=3, type=float, default=(0.0, 0.0, -6.0))
    parser.add_argument("--reset-yaw", type=float, default=0.0)
    parser.add_argument(
        "--camera-mount-yaw-offset", type=float,
        default=-math.pi / 2.0,
        help="fixed yaw from PX4 body-forward to the Gazebo camera optical axis",
    )
    parser.add_argument("--velocity-max", type=float, default=8.0)
    parser.add_argument("--yaw-rate-max", type=float, default=1.0)
    parser.add_argument("--control-rate", type=float, default=20.0)
    parser.add_argument("--warmup-setpoints", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = GazeboRosBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
