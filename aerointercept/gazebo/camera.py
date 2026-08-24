"""One camera transform shared by Gazebo training, evaluation, and tests."""

from __future__ import annotations

import numpy as np


OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 640
COLOR_ORDER = "RGB"
TRANSFORM = "full_frame_letterbox_v1"


def decode_ros_image(
    data: bytes, width: int, height: int, step: int, encoding: str
) -> np.ndarray:
    """Decode common ROS image encodings to contiguous RGB without cropping."""
    encoding = encoding.lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported camera encoding: {encoding!r}")
    channels = channels_by_encoding[encoding]
    row_bytes = int(width) * channels
    if int(step) < row_bytes or len(data) < int(step) * int(height):
        raise ValueError("invalid ROS image row stride or payload length")
    rows = np.frombuffer(data, dtype=np.uint8, count=int(step) * int(height)).reshape(
        int(height), int(step)
    )
    image = rows[:, :row_bytes].reshape(int(height), int(width), channels)
    if encoding in ("bgr8", "bgra8"):
        image = image[..., [2, 1, 0, 3] if channels == 4 else [2, 1, 0]]
    if channels == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def letterbox_rgb(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Resize a complete RGB frame into 640 square pixels with symmetric pads.

    No source pixel is cropped.  OpenCV is imported lazily because the Conda
    model/unit-test path should not need ROS vision dependencies.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected an HxWx3 uint8 RGB image")
    source_height, source_width = image.shape[:2]
    if source_width < 1 or source_height < 1:
        raise ValueError("camera dimensions must be positive")
    scale = min(OUTPUT_WIDTH / source_width, OUTPUT_HEIGHT / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the Gazebo letterbox transform") from exc
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )
    left = (OUTPUT_WIDTH - resized_width) // 2
    top = (OUTPUT_HEIGHT - resized_height) // 2
    output = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    output[top:top + resized_height, left:left + resized_width] = resized
    metadata = {
        "name": TRANSFORM,
        "source_width": int(source_width),
        "source_height": int(source_height),
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": left,
        "pad_right": OUTPUT_WIDTH - resized_width - left,
        "pad_top": top,
        "pad_bottom": OUTPUT_HEIGHT - resized_height - top,
        "color_order": COLOR_ORDER,
    }
    return output, metadata
