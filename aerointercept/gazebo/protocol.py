"""Versioned local transport shared by Python 3.10 ROS and Python 3.12 CUDA."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

import numpy as np


PROTOCOL_VERSION = 1
HEADER = struct.Struct("!Q")
MAX_PACKET_BYTES = 64 * 1024 * 1024


class BridgeProtocolError(RuntimeError):
    """Raised for malformed, incompatible, or failed bridge requests."""


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Gazebo bridge closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(connection: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > MAX_PACKET_BYTES:
        raise BridgeProtocolError(f"packet too large: {len(payload)} bytes")
    connection.sendall(HEADER.pack(len(payload)) + payload)


def receive_packet(connection: socket.socket) -> Any:
    size = HEADER.unpack(_recv_exact(connection, HEADER.size))[0]
    if size > MAX_PACKET_BYTES:
        raise BridgeProtocolError(f"refusing {size}-byte packet")
    return pickle.loads(_recv_exact(connection, size))


def require_response(response: dict) -> dict:
    if not isinstance(response, dict):
        raise BridgeProtocolError("bridge response is not a mapping")
    if not response.get("ok", False):
        raise BridgeProtocolError(str(response.get("error", "unknown bridge error")))
    if response.get("protocol") != PROTOCOL_VERSION:
        raise BridgeProtocolError("Gazebo bridge protocol version mismatch")
    return response


def image_from_snapshot(snapshot: dict) -> np.ndarray:
    """Decode the bridge's standardized 640x640 RGB letterbox image."""
    shape = tuple(snapshot.get("image_shape", ()))
    if shape != (640, 640, 3):
        raise BridgeProtocolError(f"camera output is not 640x640 RGB: {shape}")
    if snapshot.get("image_dtype") != "uint8":
        raise BridgeProtocolError("camera storage dtype must be uint8")
    data = snapshot.get("image")
    if not isinstance(data, bytes):
        raise BridgeProtocolError("camera payload must be immutable bytes")
    expected = int(np.prod(shape))
    if len(data) != expected:
        raise BridgeProtocolError(
            f"camera payload length {len(data)} does not match {expected}"
        )
    # copy() ensures successive snapshots and vector environments never alias.
    image = np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()
    return np.transpose(image, (2, 0, 1))


def request(command: str, **values: Any) -> dict:
    return {"protocol": PROTOCOL_VERSION, "command": command, **values}
