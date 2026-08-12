"""Minimal deployment runtime with preprocessing and confidence watchdog."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .actions import BodyVelocityCommand, decode_action


@dataclass(frozen=True)
class RuntimeResult:
    command: BodyVelocityCommand | None
    action: np.ndarray
    future_position_body_m: np.ndarray
    collision_risk: float
    confidence: float
    attention: np.ndarray
    fallback_required: bool
    reason: str


class EndToEndRuntime:
    """Stateful frame-history adapter around the exported TorchScript actor."""

    def __init__(self, model_path, metadata_path=None, device="cpu"):
        model_path = Path(model_path)
        metadata_path = (
            Path(metadata_path) if metadata_path
            else model_path.with_name(model_path.stem + "_meta.json")
        )
        with metadata_path.open("r", encoding="utf-8") as stream:
            self.metadata = json.load(stream)
        if self.metadata.get("phase") != 3:
            raise ValueError("metadata does not describe an end-to-end policy")
        expected_digest = self.metadata.get("sha256")
        if expected_digest:
            actual_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError("policy checksum does not match metadata")

        input_meta = self.metadata["input"]
        self.width = int(input_meta["image_width"])
        self.height = int(input_meta["image_height"])
        self.history_frames = int(input_meta["history_frames"])
        protocol = self.metadata["action_protocol"]
        self.velocity_max = float(protocol["velocity_max"])
        self.yaw_rate_max = float(protocol["yaw_rate_max"])
        self.confidence_threshold = float(
            self.metadata["safety"]["confidence_threshold"])
        self.max_command_speed = float(
            self.metadata["safety"].get(
                "max_command_speed", self.velocity_max))
        self.future_position_norm = float(
            self.metadata.get("auxiliary_contract", {}).get(
                "future_position_norm_m", 1.0))
        if self.history_frames < 1:
            raise ValueError("history_frames must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        if self.max_command_speed <= 0.0:
            raise ValueError("max_command_speed must be positive")
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
        self._history = deque(maxlen=self.history_frames)

    def reset(self):
        self._history.clear()

    def step(self, image, yaw: float, *, color_order="RGB") -> RuntimeResult:
        frame = self._prepare_frame(image, color_order)
        if not self._history:
            for _ in range(self.history_frames - 1):
                self._history.append(frame.copy())
        self._history.append(frame)
        frames = np.stack(tuple(self._history), axis=0)
        tensor = torch.from_numpy(frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, future, risk_logit, confidence_logit, attention = self.model(tensor)
        action_np = action[0].cpu().numpy()
        future_np = future[0].cpu().numpy() * self.future_position_norm
        risk = float(torch.sigmoid(risk_logit[0]))
        confidence = float(torch.sigmoid(confidence_logit[0]))
        attention_np = attention[0].cpu().numpy()

        arrays_finite = (
            np.isfinite(action_np).all()
            and np.isfinite(future_np).all()
            and np.isfinite(attention_np).all()
            and np.isfinite(risk)
            and np.isfinite(confidence)
        )
        if not arrays_finite:
            return RuntimeResult(
                None, action_np, future_np, risk, confidence, attention_np,
                True, "non_finite_model_output")
        if confidence < self.confidence_threshold:
            return RuntimeResult(
                None, action_np, future_np, risk, confidence, attention_np,
                True, "low_visual_confidence")

        command = decode_action(
            action_np, yaw,
            velocity_max=self.velocity_max,
            yaw_rate_max=self.yaw_rate_max,
        )
        if np.linalg.norm(command.ned_velocity) > self.max_command_speed + 1e-6:
            return RuntimeResult(
                None, action_np, future_np, risk, confidence, attention_np,
                True, "command_speed_limit_exceeded")
        return RuntimeResult(
            command, action_np, future_np, risk, confidence, attention_np,
            False, "")

    def _prepare_frame(self, image, color_order):
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("image must have HWC shape with three channels")
        if color_order.upper() == "BGR":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color_order.upper() != "RGB":
            raise ValueError("color_order must be 'RGB' or 'BGR'")
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        return np.ascontiguousarray(frame.transpose(2, 0, 1))
