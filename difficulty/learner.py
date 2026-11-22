"""Out-of-fold learner difficulty and fold assignment helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np


def learner_errors(
    predicted_motion: np.ndarray,
    gripper_logits: np.ndarray,
    expert_actions: np.ndarray,
    motion_scale: float,
) -> dict[str, float]:
    """Compute per-trajectory normalized motion SSE and gripper BCE."""
    predicted = np.asarray(predicted_motion, dtype=np.float64)
    logits = np.asarray(gripper_logits, dtype=np.float64).reshape(-1)
    actions = np.asarray(expert_actions, dtype=np.float64)
    if predicted.shape != actions[:, :3].shape or logits.shape != (len(actions),):
        raise ValueError("learner prediction arrays have incompatible shapes")
    if motion_scale <= 0.0:
        raise ValueError("motion_scale must be positive")
    normalized_error = (predicted - actions[:, :3]) / motion_scale
    motion_error = float(np.mean(np.sum(normalized_error**2, axis=1)))
    targets = (actions[:, 3] > 0.0).astype(np.float64)
    bce = np.maximum(logits, 0.0) - logits * targets + np.log1p(
        np.exp(-np.abs(logits))
    )
    result = {
        "learner_motion_error": motion_error,
        "learner_gripper_error": float(np.mean(bce)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("learner errors must be finite")
    return result


def generate_oof_folds(
    trajectory_ids: Sequence[str], number_of_folds: int, seed: int
) -> list[dict[str, Any]]:
    """Assign every trajectory to exactly one deterministic held-out fold."""
    ordered = sorted(str(value) for value in trajectory_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("trajectory IDs must be unique")
    if number_of_folds <= 1 or len(ordered) % number_of_folds != 0:
        raise ValueError("trajectory count must divide evenly across folds")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(ordered, dtype=str)[rng.permutation(len(ordered))]
    held_out_groups = np.split(shuffled, number_of_folds)
    all_ids = set(ordered)
    folds = []
    for fold_index, held_out in enumerate(held_out_groups):
        held_out_ids = sorted(held_out.tolist())
        training_ids = sorted(all_ids.difference(held_out_ids))
        folds.append(
            {
                "fold_index": fold_index,
                "training_trajectory_ids": training_ids,
                "held_out_trajectory_ids": held_out_ids,
            }
        )
    validate_oof_folds(folds, ordered)
    return folds


def validate_oof_folds(
    folds: Sequence[dict[str, Any]], expected_ids: Sequence[str]
) -> None:
    expected = set(expected_ids)
    scored: list[str] = []
    for fold in folds:
        training = set(fold["training_trajectory_ids"])
        held_out = set(fold["held_out_trajectory_ids"])
        if training & held_out:
            raise ValueError("OOF training and held-out IDs overlap")
        if training | held_out != expected:
            raise ValueError("OOF fold does not cover the expected training IDs")
        scored.extend(fold["held_out_trajectory_ids"])
    if len(scored) != len(expected) or set(scored) != expected:
        raise ValueError("Each OOF trajectory must be held out exactly once")


def split_fingerprint(trajectory_ids: Sequence[str]) -> str:
    payload = json.dumps(sorted(trajectory_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
