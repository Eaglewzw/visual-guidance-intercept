"""Export an end-to-end checkpoint as a self-describing TorchScript policy."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .config import DotDict, load_config
from .end_to_end.policy import EndToEndActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="export/e2e_policy.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if checkpoint.get("phase") != 3:
        raise ValueError("checkpoint is not an end-to-end policy")
    model_config = DotDict(checkpoint.get(
        "model_config", dict(cfg.end_to_end.model)))
    model = EndToEndActorCritic(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    actor = model.actor.eval()
    scripted = torch.jit.script(actor)

    render_config = checkpoint.get(
        "render_config", dict(cfg.end_to_end.render))
    action_config = checkpoint.get(
        "action_config", dict(cfg.end_to_end.action))
    label_config = checkpoint.get(
        "label_config", dict(cfg.end_to_end.labels))
    safety_config = checkpoint.get(
        "safety_config", dict(cfg.end_to_end.safety))
    height = int(render_config["image_height"])
    width = int(render_config["image_width"])
    history = int(model_config.history_frames)
    example = torch.zeros(1, history, 3, height, width, dtype=torch.uint8)
    outputs = scripted(example)
    expected_shapes = ((1, 4), (1, 3), (1,), (1,))
    for output, shape in zip(outputs[:4], expected_shapes):
        if tuple(output.shape) != shape:
            raise RuntimeError(
                f"export smoke test failed: {tuple(output.shape)} != {shape}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "phase": 3,
        "policy_version": "end_to_end_full_frame_v1",
        "sha256": digest,
        "input": {
            "name": "frames",
            "layout": "N,F,C,H,W",
            "dtype": "uint8",
            "color_order": "RGB",
            "history_frames": history,
            "image_width": width,
            "image_height": height,
            "normalization": "embedded_in_model",
        },
        "outputs": [
            {"name": "action", "shape": [1, 4], "activation": "tanh"},
            {"name": "future_position_normalized", "shape": [1, 3], "activation": "tanh"},
            {"name": "collision_risk_logit", "shape": [1]},
            {"name": "confidence_logit", "shape": [1]},
            {"name": "attention", "shape": list(outputs[4].shape)},
        ],
        "action_protocol": {
            "version": "body_velocity_yaw_rate_v1",
            "components": ["forward", "right", "down", "yaw_rate"],
            "velocity_max": float(action_config["velocity_max"]),
            "yaw_rate_max": float(action_config["yaw_rate_max"]),
            "frame_conversion": "heading body velocity rotated to NED with current yaw",
        },
        "auxiliary_contract": {
            "future_frame": "heading body [forward,right,down]",
            "future_horizon_s": float(
                label_config["future_horizon_s"]),
            "future_position_norm_m": float(
                label_config["position_norm"]),
            "risk_horizon_s": float(label_config["risk_horizon_s"]),
            "risk_radius_m": float(label_config["risk_radius"]),
        },
        "safety": safety_config,
        "source_checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_hit_rate": checkpoint.get("hit_rate"),
        "checkpoint_global_step": checkpoint.get("global_step"),
    }
    metadata_path = output_path.with_name(output_path.stem + "_meta.json")
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    print(f"TorchScript: {output_path}")
    print(f"metadata:    {metadata_path}")


if __name__ == "__main__":
    main()
