"""Joining, fold validation, grouping, and descriptive diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from data.task_sampler import TaskConfig


def evaluation_task_from_trajectory(
    trajectory_id: str, trajectory: Mapping[str, Any]
) -> TaskConfig:
    """Reconstruct the exact demonstrated task configuration deterministically."""
    return TaskConfig(
        task_id=str(trajectory_id),
        seed=int(trajectory["task_seed"]),
        object_position=np.asarray(trajectory["task_object_position"], dtype=np.float64),
        goal_position=np.asarray(trajectory["goal_position"], dtype=np.float64),
        target_radius=float(trajectory["target_radius"]),
    )


def validate_checkpoint_for_fold(
    checkpoint: Mapping[str, Any], fold: Mapping[str, Any]
) -> None:
    training = set(checkpoint["training_trajectory_ids"])
    held_out = set(checkpoint["held_out_trajectory_ids"])
    expected_training = set(fold["training_trajectory_ids"])
    expected_held_out = set(fold["held_out_trajectory_ids"])
    if training != expected_training or held_out != expected_held_out:
        raise ValueError("OOF checkpoint does not match fold assignment")
    if training & held_out:
        raise ValueError("OOF checkpoint has training/held-out overlap")


def assert_held_out(checkpoint: Mapping[str, Any], trajectory_id: str) -> None:
    if trajectory_id in set(checkpoint["training_trajectory_ids"]):
        raise ValueError(f"Leakage: {trajectory_id} appears in checkpoint training IDs")
    if trajectory_id not in set(checkpoint["held_out_trajectory_ids"]):
        raise ValueError(f"{trajectory_id} is not held out by this checkpoint")


def join_by_trajectory_id(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    right_map = _unique_map(right)
    left_map = _unique_map(left)
    if set(left_map) != set(right_map):
        raise ValueError("Trajectory ID sets do not match")
    return [{**left_map[key], **right_map[key]} for key in sorted(left_map)]


def group_outcome_rates(
    rows: Sequence[Mapping[str, Any]], group_ids: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, float | int]]:
    outcome = {str(row["trajectory_id"]): bool(row["success"]) for row in rows}
    listed_ids = [str(trajectory_id) for ids in group_ids.values() for trajectory_id in ids]
    if set(listed_ids) != set(outcome) or len(listed_ids) != len(set(listed_ids)):
        raise ValueError("Difficulty groups do not cover rollout IDs")
    result = {}
    for group_name in ("easy", "medium", "hard"):
        ids = list(group_ids[group_name])
        successes = sum(outcome[trajectory_id] for trajectory_id in ids)
        result[group_name] = {
            "count": len(ids),
            "successes": successes,
            "failures": len(ids) - successes,
            "success_rate": successes / len(ids),
            "failure_rate": 1.0 - successes / len(ids),
        }
    return result


def summarize_fold_outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_fold: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_fold.setdefault(int(row["fold_id"]), []).append(row)
    result = {}
    for fold_id, fold_rows in sorted(by_fold.items()):
        successes = sum(bool(row["success"]) for row in fold_rows)
        result[str(fold_id)] = {
            "tasks": len(fold_rows),
            "successes": successes,
            "failures": len(fold_rows) - successes,
            "success_rate": successes / len(fold_rows),
        }
    return result


def failure_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(
            str(row["failure_reason"])
            for row in rows
            if not bool(row["success"])
        )
    )


def _unique_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in result:
            raise ValueError(f"Duplicate trajectory ID: {trajectory_id}")
        result[trajectory_id] = row
    return result
