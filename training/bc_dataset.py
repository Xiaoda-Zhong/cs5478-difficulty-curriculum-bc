"""Trajectory-level splitting and transition loading for behavior cloning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.trajectory import load_trajectory, validate_trajectory


class BCTransitionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory state-action pairs flattened after trajectory selection."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray) -> None:
        if observations.ndim != 2 or observations.shape[1] <= 0:
            raise ValueError("observations must have shape (N, observation_dim)")
        if actions.shape != (observations.shape[0], 4):
            raise ValueError("actions must have shape (N, 4)")
        if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
            raise ValueError("dataset arrays must contain only finite values")
        self.observations = torch.from_numpy(observations.astype(np.float32, copy=False))
        self.actions = torch.from_numpy(actions.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[index], self.actions[index]


def generate_trajectory_split(
    trajectory_ids: Sequence[str], train_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Generate a deterministic disjoint split of sorted trajectory IDs."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    ordered_ids = sorted(str(value) for value in trajectory_ids)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("trajectory IDs must be unique")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ordered_ids, dtype=str)[rng.permutation(len(ordered_ids))]
    train_count = int(round(len(ordered_ids) * train_fraction))
    return sorted(shuffled[:train_count].tolist()), sorted(shuffled[train_count:].tolist())


def load_or_create_split(
    split_path: str | Path,
    trajectory_ids: Sequence[str],
    *,
    dataset_fingerprint: str,
    split_seed: int,
    train_fraction: float,
) -> dict[str, Any]:
    """Load the frozen split, or create it once if it does not exist."""
    path = Path(split_path)
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        if split["dataset_fingerprint"] != dataset_fingerprint:
            raise ValueError("BC split dataset fingerprint mismatch")
        if int(split["split_seed"]) != split_seed:
            raise ValueError("Existing BC split uses a different split seed")
    else:
        train_ids, validation_ids = generate_trajectory_split(
            trajectory_ids, train_fraction, split_seed
        )
        split = {
            "dataset_fingerprint": dataset_fingerprint,
            "split_seed": split_seed,
            "train_fraction": train_fraction,
            "train_trajectory_ids": train_ids,
            "validation_trajectory_ids": validation_ids,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    validate_split(split, trajectory_ids)
    return split


def validate_split(split: dict[str, Any], trajectory_ids: Sequence[str]) -> None:
    train_ids = set(split["train_trajectory_ids"])
    validation_ids = set(split["validation_trajectory_ids"])
    expected = set(trajectory_ids)
    if train_ids & validation_ids:
        raise ValueError("Train and validation trajectory IDs overlap")
    if train_ids | validation_ids != expected:
        raise ValueError("Train and validation split does not cover the dataset")


def load_transitions(
    demonstration_dir: str | Path, trajectory_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Load and flatten transitions from only the requested trajectories."""
    directory = Path(demonstration_dir)
    observation_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for trajectory_id in trajectory_ids:
        trajectory = load_trajectory(directory / f"{trajectory_id}.npz")
        validate_trajectory(trajectory, require_success=True)
        stored_id = str(np.asarray(trajectory["trajectory_id"]).item())
        if stored_id != trajectory_id:
            raise ValueError(f"Trajectory ID mismatch in {trajectory_id}.npz")
        observation_parts.append(np.asarray(trajectory["observations"], dtype=np.float32))
        action_parts.append(np.asarray(trajectory["actions"], dtype=np.float32))
    return np.concatenate(observation_parts), np.concatenate(action_parts)


def compute_observation_statistics(
    training_observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension population statistics from training data only."""
    observations = np.asarray(training_observations, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[1] <= 0:
        raise ValueError("training observations must have shape (N, observation_dim)")
    if not np.all(np.isfinite(observations)):
        raise ValueError("training observations must be finite")
    return observations.mean(axis=0).astype(np.float32), observations.std(axis=0).astype(np.float32)
