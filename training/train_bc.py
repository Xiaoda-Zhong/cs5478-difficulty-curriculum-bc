"""Reusable minibatch trainer for the two-head behavior-cloning policy."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.bc_policy import BCPolicy
from training.bc_dataset import BCTransitionDataset


def select_device(requested: str) -> torch.device:
    """Resolve ``auto`` to Apple MPS when available, otherwise CPU."""
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if normalized not in {"cpu", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', or 'mps'")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(normalized)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_behavior_cloning(
    *,
    train_observations: np.ndarray,
    train_actions: np.ndarray,
    validation_observations: np.ndarray,
    validation_actions: np.ndarray,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    hidden_dims: Sequence[int],
    batch_size: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    lambda_gripper: float,
    normalization_epsilon: float,
    seed: int,
    device: torch.device,
    checkpoint_dir: str | Path,
    checkpoint_metadata: dict[str, Any],
    normalize_motion_targets: bool = False,
    early_stopping_patience: int | None = None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Train on arbitrary transition arrays and save best/final checkpoints."""
    seed_everything(seed)
    checkpoint_directory = Path(checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    train_dataset = BCTransitionDataset(train_observations, train_actions)
    validation_dataset = BCTransitionDataset(
        validation_observations, validation_actions
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    observation_dim = int(train_observations.shape[1])
    if validation_observations.shape[1] != observation_dim:
        raise ValueError("training and validation observation dimensions differ")
    model = BCPolicy(observation_dim=observation_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    motion_loss_function = nn.MSELoss()
    gripper_loss_function = nn.BCEWithLogitsLoss()
    mean_tensor = torch.as_tensor(observation_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(observation_std, dtype=torch.float32, device=device)
    max_motion_norm = float(np.linalg.norm(train_actions[:, :3], axis=1).max())
    motion_scale = max_motion_norm if normalize_motion_targets else 1.0

    history: dict[str, list[float]] = {
        key: []
        for key in (
            "train_total_loss",
            "train_motion_mse",
            "train_gripper_accuracy",
            "validation_total_loss",
            "validation_motion_mse",
            "validation_gripper_accuracy",
        )
    }
    best_metrics: dict[str, Any] = {}
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    final_checkpoint: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            mean_tensor,
            std_tensor,
            normalization_epsilon,
            lambda_gripper,
            motion_loss_function,
            gripper_loss_function,
            device,
            optimizer,
            motion_scale,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            mean_tensor,
            std_tensor,
            normalization_epsilon,
            lambda_gripper,
            motion_loss_function,
            gripper_loss_function,
            device,
            optimizer=None,
            motion_scale=motion_scale,
        )
        for key, value in train_metrics.items():
            history[f"train_{key}"].append(value)
        for key, value in validation_metrics.items():
            history[f"validation_{key}"].append(value)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss={train_metrics['total_loss']:.6f} "
            f"motion={train_metrics['motion_mse']:.8f} "
            f"grip_acc={train_metrics['gripper_accuracy']:.3f} | "
            f"val loss={validation_metrics['total_loss']:.6f} "
            f"motion={validation_metrics['motion_mse']:.8f} "
            f"grip_acc={validation_metrics['gripper_accuracy']:.3f}"
        )

        checkpoint = _checkpoint(
            model=model,
            epoch=epoch,
            validation_metrics=validation_metrics,
            hidden_dims=hidden_dims,
            observation_mean=observation_mean,
            observation_std=observation_std,
            normalization_epsilon=normalization_epsilon,
            max_motion_norm=max_motion_norm,
            motion_scale=motion_scale,
            motion_targets_normalized=normalize_motion_targets,
            seed=seed,
            metadata=checkpoint_metadata,
        )
        final_checkpoint = checkpoint
        if validation_metrics["total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["total_loss"]
            epochs_without_improvement = 0
            best_metrics = {
                "epoch": epoch,
                **validation_metrics,
                "max_motion_norm": max_motion_norm,
                "parameter_count": sum(p.numel() for p in model.parameters()),
            }
            torch.save(checkpoint, checkpoint_directory / "best.pt")
        else:
            epochs_without_improvement += 1
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; no validation improvement "
                f"for {early_stopping_patience} epochs"
            )
            break

    if final_checkpoint is None:
        raise RuntimeError("Training produced no checkpoint")
    torch.save(final_checkpoint, checkpoint_directory / "final.pt")
    return history, best_metrics


def _run_epoch(
    model: BCPolicy,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    epsilon: float,
    lambda_gripper: float,
    motion_loss_function: nn.Module,
    gripper_loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    motion_scale: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_examples = 0
    total_loss_sum = 0.0
    motion_loss_sum = 0.0
    correct_gripper = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for observations, actions in loader:
            observations = observations.to(device)
            actions = actions.to(device)
            normalized = (observations - mean) / (std + epsilon)
            predicted_motion, gripper_logits = model(normalized)
            motion_targets = actions[:, :3] / motion_scale
            motion_loss = motion_loss_function(predicted_motion, motion_targets)
            gripper_targets = (actions[:, 3] > 0.0).float().unsqueeze(1)
            gripper_loss = gripper_loss_function(gripper_logits, gripper_targets)
            total_loss = motion_loss + lambda_gripper * gripper_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            batch_size = observations.shape[0]
            total_examples += batch_size
            total_loss_sum += float(total_loss.item()) * batch_size
            motion_loss_sum += float(motion_loss.item()) * batch_size
            predictions = torch.sigmoid(gripper_logits) >= 0.5
            correct_gripper += int((predictions == gripper_targets.bool()).sum().item())
    return {
        "total_loss": total_loss_sum / total_examples,
        "motion_mse": motion_loss_sum / total_examples,
        "gripper_accuracy": correct_gripper / total_examples,
    }


def _checkpoint(
    *,
    model: BCPolicy,
    epoch: int,
    validation_metrics: dict[str, float],
    hidden_dims: Sequence[int],
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    normalization_epsilon: float,
    max_motion_norm: float,
    motion_scale: float,
    motion_targets_normalized: bool,
    seed: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "architecture": {"observation_dim": model.observation_dim, "hidden_dims": list(hidden_dims)},
        "observation_mean": observation_mean.tolist(),
        "observation_std": observation_std.tolist(),
        "normalization_epsilon": normalization_epsilon,
        "max_motion_norm": max_motion_norm,
        "motion_scale": motion_scale,
        "motion_targets_normalized": motion_targets_normalized,
        "training_seed": seed,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        **metadata,
    }


def save_history(path: str | Path, history: dict[str, list[float]]) -> None:
    Path(path).write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train_fixed_epoch_behavior_cloning(
    *,
    train_observations: np.ndarray,
    train_actions: np.ndarray,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    hidden_dims: Sequence[int],
    batch_size: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    lambda_gripper: float,
    normalization_epsilon: float,
    seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, float]:
    """Train scaled-motion BC for a fixed budget without held-out selection."""
    seed_everything(seed)
    dataset = BCTransitionDataset(train_observations, train_actions)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    observation_dim = int(train_observations.shape[1])
    model = BCPolicy(observation_dim=observation_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    motion_loss_function = nn.MSELoss()
    gripper_loss_function = nn.BCEWithLogitsLoss()
    mean_tensor = torch.as_tensor(observation_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(observation_std, dtype=torch.float32, device=device)
    motion_scale = float(np.linalg.norm(train_actions[:, :3], axis=1).max())
    metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        metrics = _run_epoch(
            model,
            loader,
            mean_tensor,
            std_tensor,
            normalization_epsilon,
            lambda_gripper,
            motion_loss_function,
            gripper_loss_function,
            device,
            optimizer,
            motion_scale,
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:03d}/{epochs} "
                f"loss={metrics['total_loss']:.6f} "
                f"motion={metrics['motion_mse']:.6f} "
                f"grip_acc={metrics['gripper_accuracy']:.4f}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": {"observation_dim": observation_dim, "hidden_dims": list(hidden_dims)},
        "observation_mean": observation_mean.tolist(),
        "observation_std": observation_std.tolist(),
        "normalization_epsilon": normalization_epsilon,
        "max_motion_norm": motion_scale,
        "motion_scale": motion_scale,
        "motion_targets_normalized": True,
        "training_seed": seed,
        "epoch": epochs,
        "training_metrics": metrics,
        **checkpoint_metadata,
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return metrics
