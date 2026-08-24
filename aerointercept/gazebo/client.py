"""Conda-side client for the system-Python Gazebo ROS bridge."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .protocol import receive_packet, request, require_response, send_packet


class GazeboBridgeClient:
    def __init__(self, socket_path: str | Path, timeout: float = 10.0):
        self.socket_path = str(socket_path)
        self.timeout = float(timeout)
        self._socket: socket.socket | None = None

    def connect(self, wait_seconds: float = 30.0) -> dict:
        deadline = time.monotonic() + float(wait_seconds)
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout)
            try:
                connection.connect(self.socket_path)
                self._socket = connection
                status = self.ping()
                while not status.get("ready", False) and time.monotonic() < deadline:
                    time.sleep(0.2)
                    status = self.ping()
                if status.get("ready", False):
                    return status
                self.close()
                raise TimeoutError(
                    "Gazebo bridge socket exists but camera and PX4 odometry did not become ready"
                )
            except OSError as exc:
                last_error = exc
                connection.close()
                time.sleep(0.1)
        raise ConnectionError(
            f"Gazebo bridge did not appear at {self.socket_path}: {last_error}"
        )

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def _call(self, command: str, **values) -> dict:
        if self._socket is None:
            raise ConnectionError("Gazebo bridge client is not connected")
        send_packet(self._socket, request(command, **values))
        return require_response(receive_packet(self._socket))

    def ping(self) -> dict:
        return self._call("ping")

    def snapshot(self, after_sequence: int = -1, timeout: float | None = None) -> dict:
        return self._call(
            "snapshot",
            after_sequence=int(after_sequence),
            timeout=float(self.timeout if timeout is None else timeout),
        )["snapshot"]

    def action(self, values: Iterable[float]) -> dict:
        action = np.asarray(values, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite four-vector")
        return self._call("action", action=action.tolist())["command"]

    def reset(
        self,
        position_ned: Iterable[float],
        yaw: float = 0.0,
        *,
        look_at_target: bool = True,
    ) -> dict:
        position = np.asarray(position_ned, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("reset position must be a finite NED three-vector")
        return self._call(
            "reset", position=position.tolist(), yaw=float(yaw),
            look_at_target=bool(look_at_target),
        )["command"]

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()
