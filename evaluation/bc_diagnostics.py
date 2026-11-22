"""Offline validation diagnostics for behavior-cloning checkpoints."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data.trajectory import load_trajectory, validate_trajectory
from models.bc_policy import BCPolicyAgent


PHASE_ORDER = (
    "RAISE",
    "APPROACH",
    "DESCEND",
    "GRASP",
    "LIFT",
    "TRANSPORT",
    "PLACE",
    "RELEASE",
    "RETREAT",
)


def load_validation_with_phases(
    demonstration_dir: str | Path, trajectory_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    for trajectory_id in trajectory_ids:
        trajectory = load_trajectory(Path(demonstration_dir) / f"{trajectory_id}.npz")
        validate_trajectory(trajectory, require_success=True)
        trajectory_phases = np.asarray(trajectory["phases"]).astype(str)
        if trajectory_phases.shape != (int(trajectory["trajectory_length"]),):
            raise ValueError(f"Invalid phase labels in {trajectory_id}")
        observations.append(np.asarray(trajectory["observations"], dtype=np.float32))
        actions.append(np.asarray(trajectory["actions"], dtype=np.float32))
        phases.append(trajectory_phases)
    return np.concatenate(observations), np.concatenate(actions), np.concatenate(phases)


def compute_bc_diagnostics(
    policy: BCPolicyAgent,
    observations: np.ndarray,
    actions: np.ndarray,
    phases: np.ndarray,
    motion_scale: float,
) -> dict[str, Any]:
    predicted_motion, close_probabilities = policy.predict_batch(observations)
    expert_motion = np.asarray(actions[:, :3], dtype=np.float64)
    predicted_close = close_probabilities >= 0.5
    target_close = actions[:, 3] > 0.0
    overall = _motion_metrics(predicted_motion, expert_motion)
    overall["relative_motion_error"] = overall["mean_l2_error"] / motion_scale

    per_phase = {}
    for phase in PHASE_ORDER:
        mask = phases == phase
        metrics = _motion_metrics(predicted_motion[mask], expert_motion[mask])
        metrics["transitions"] = int(mask.sum())
        per_phase[phase] = metrics

    true_positive = int(np.sum(predicted_close & target_close))
    false_positive = int(np.sum(predicted_close & ~target_close))
    false_negative = int(np.sum(~predicted_close & target_close))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    counts = Counter("close" if value else "open" for value in target_close)
    gripper = {
        "open_targets": counts["open"],
        "close_targets": counts["close"],
        "open_fraction": counts["open"] / len(target_close),
        "close_fraction": counts["close"] / len(target_close),
        "accuracy": float(np.mean(predicted_close == target_close)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return {"motion": overall, "per_phase": per_phase, "gripper": gripper}


def _motion_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if len(target) == 0:
        return {
            "motion_mse": None,
            "rmse_per_component": None,
            "mean_l2_error": None,
            "median_l2_error": None,
            "mean_expert_action_magnitude": None,
            "mean_predicted_action_magnitude": None,
            "mean_cosine_similarity": None,
        }
    error = predicted - target
    l2_error = np.linalg.norm(error, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    valid_direction = (target_norm > 1e-9) & (predicted_norm > 1e-9)
    cosine = np.sum(predicted[valid_direction] * target[valid_direction], axis=1) / (
        predicted_norm[valid_direction] * target_norm[valid_direction]
    )
    return {
        "motion_mse": float(np.mean(error**2)),
        "rmse_per_component": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "mean_l2_error": float(np.mean(l2_error)),
        "median_l2_error": float(np.median(l2_error)),
        "mean_expert_action_magnitude": float(np.mean(target_norm)),
        "mean_predicted_action_magnitude": float(np.mean(predicted_norm)),
        "mean_cosine_similarity": float(np.mean(cosine)) if len(cosine) else None,
    }
