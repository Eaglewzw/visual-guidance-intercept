"""Train the full-frame image actor with PNG behavior cloning and aux labels."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import load_config
from ..end_to_end.data import (
    EpisodeSequenceDataset,
    episode_files,
    load_manifest,
    split_episode_files,
)
from ..end_to_end.policy import EndToEndActorCritic


@dataclass
class Losses:
    total: torch.Tensor
    action: torch.Tensor
    future: torch.Tensor
    risk: torch.Tensor
    confidence: torch.Tensor


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_losses(actor, batch, device, auxiliary_cfg) -> Losses:
    frames = batch["frames"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    future_target = batch["future_position"].to(device, non_blocking=True)
    risk_target = batch["collision_risk"].to(device, non_blocking=True)
    confidence_target = batch["confidence"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)

    batch_size, sequence_length = frames.shape[:2]
    frames = frames.reshape(batch_size * sequence_length, *frames.shape[2:])
    action_pred, future_pred, risk_logit, confidence_logit, _ = actor(frames)
    action_pred = action_pred.view(batch_size, sequence_length, -1)
    future_pred = future_pred.view(batch_size, sequence_length, -1)
    risk_logit = risk_logit.view(batch_size, sequence_length)
    confidence_logit = confidence_logit.view(batch_size, sequence_length)

    action_loss = masked_mean(
        (action_pred - actions).square().mean(dim=-1), mask)
    visible_mask = mask * confidence_target
    future_loss = masked_mean(
        (future_pred - future_target).square().mean(dim=-1), visible_mask)
    risk_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            risk_logit, risk_target, reduction="none"), mask)
    confidence_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            confidence_logit, confidence_target, reduction="none"), mask)
    total = (
        action_loss
        + auxiliary_cfg.future_coef * future_loss
        + auxiliary_cfg.risk_coef * risk_loss
        + auxiliary_cfg.confidence_coef * confidence_loss
    )
    return Losses(total, action_loss, future_loss, risk_loss, confidence_loss)


def run_epoch(actor, loader, device, auxiliary_cfg, optimizer=None):
    training = optimizer is not None
    actor.train(training)
    totals = {name: 0.0 for name in Losses.__annotations__}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            losses = compute_losses(actor, batch, device, auxiliary_cfg)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer.step()
            for name in totals:
                totals[name] += float(getattr(losses, name).detach())
            batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="data/e2e_bc")
    parser.add_argument("--out", default="checkpoints/e2e_bc.pt")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bc_cfg = cfg.end_to_end.bc
    torch.manual_seed(args.seed)
    manifest = load_manifest(args.data)
    if int(manifest["history_frames"]) != int(cfg.end_to_end.model.history_frames):
        raise ValueError("dataset/model history_frames mismatch")
    expected_image_size = (
        int(cfg.end_to_end.render.image_width),
        int(cfg.end_to_end.render.image_height),
    )
    dataset_image_size = (
        int(manifest["image_width"]), int(manifest["image_height"]))
    if dataset_image_size != expected_image_size:
        raise ValueError(
            f"dataset image size {dataset_image_size} does not match config "
            f"{expected_image_size}")

    sequence_length = args.sequence_length or bc_cfg.sequence_length
    batch_size = args.batch_size or bc_cfg.batch_size
    epochs = args.epochs or bc_cfg.epochs
    learning_rate = args.learning_rate or bc_cfg.learning_rate
    train_files, validation_files = split_episode_files(
        episode_files(args.data), bc_cfg.val_fraction, args.seed)
    train_dataset = EpisodeSequenceDataset(
        train_files, sequence_length, cfg.end_to_end.model.history_frames)
    validation_dataset = EpisodeSequenceDataset(
        validation_files, sequence_length, cfg.end_to_end.model.history_frames)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=int(bc_cfg.num_workers),
        pin_memory=args.device.startswith("cuda"),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs)

    model = EndToEndActorCritic(cfg.end_to_end.model).to(args.device)
    optimizer = torch.optim.AdamW(
        model.actor.parameters(), lr=learning_rate, weight_decay=1e-4)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")

    print(
        f"end-to-end BC: {len(train_files)} train episodes / "
        f"{len(validation_files)} validation episodes; "
        f"{len(train_dataset)} train windows")
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model.actor, train_loader, args.device,
            cfg.end_to_end.auxiliary, optimizer)
        validation_metrics = run_epoch(
            model.actor, validation_loader, args.device,
            cfg.end_to_end.auxiliary)

        saved = ""
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            torch.save({
                "phase": 3,
                "model": model.state_dict(),
                "model_config": dict(cfg.end_to_end.model),
                "render_config": dict(cfg.end_to_end.render),
                "action_config": dict(cfg.end_to_end.action),
                "label_config": dict(cfg.end_to_end.labels),
                "safety_config": dict(cfg.end_to_end.safety),
                "validation_loss": best_validation,
                "dataset_manifest": manifest,
            }, output)
            saved = " <- saved"
        print(
            f"epoch {epoch:3d}/{epochs} "
            f"train={train_metrics['total']:.4f} "
            f"val={validation_metrics['total']:.4f} "
            f"action={validation_metrics['action']:.4f} "
            f"future={validation_metrics['future']:.4f} "
            f"risk={validation_metrics['risk']:.4f} "
            f"conf={validation_metrics['confidence']:.4f}{saved}")

    print(f"best validation loss {best_validation:.5f}; saved {output}")


if __name__ == "__main__":
    main()
