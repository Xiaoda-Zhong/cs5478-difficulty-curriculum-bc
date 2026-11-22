"""Dataset-level metadata, integrity checks, and descriptive statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data.trajectory import (
    fingerprint_entry,
    load_trajectory,
    task_fingerprint,
    validate_trajectory,
)


class DatasetValidationError(ValueError):
    """Raised when files and metadata do not form a valid fixed dataset."""


def inspect_dataset(
    demonstration_dir: str | Path,
    metadata_path: str | Path,
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate a dataset and return raw descriptive statistics."""
    demo_dir = Path(demonstration_dir)
    metadata_file = Path(metadata_path)
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"Cannot read dataset metadata: {error}") from error

    configured_count = int(metadata["number_of_trajectories"])
    required_count = configured_count if expected_count is None else expected_count
    files = sorted(demo_dir.glob("traj_*.npz"))
    if len(files) != required_count or configured_count != required_count:
        raise DatasetValidationError(
            f"Expected {required_count} trajectories, found {len(files)} files "
            f"and metadata declares {configured_count}"
        )

    ranges = metadata["task_randomization"]
    lengths: list[int] = []
    path_lengths: list[float] = []
    object_goal_distances: list[float] = []
    ee_object_distances: list[float] = []
    radii: list[float] = []
    trajectory_ids: set[str] = set()
    fingerprint_entries: list[dict[str, Any]] = []

    for file_path in files:
        try:
            trajectory = load_trajectory(file_path)
            validate_trajectory(trajectory, require_success=True)
        except (OSError, ValueError) as error:
            raise DatasetValidationError(f"Malformed {file_path.name}: {error}") from error

        trajectory_id = str(np.asarray(trajectory["trajectory_id"]).item())
        if trajectory_id in trajectory_ids:
            raise DatasetValidationError(f"Duplicate trajectory ID: {trajectory_id}")
        trajectory_ids.add(trajectory_id)
        if file_path.stem != trajectory_id:
            raise DatasetValidationError(
                f"File {file_path.name} contains ID {trajectory_id!r}"
            )

        obj = np.asarray(trajectory["task_object_position"], dtype=float)
        goal = np.asarray(trajectory["goal_position"], dtype=float)
        radius = float(np.asarray(trajectory["target_radius"]).item())
        _check_range(obj[0], ranges["object_x_range"], "object x", file_path)
        _check_range(obj[1], ranges["object_y_range"], "object y", file_path)
        _check_range(goal[0], ranges["goal_x_range"], "goal x", file_path)
        _check_range(goal[1], ranges["goal_y_range"], "goal y", file_path)
        _check_range(radius, ranges["target_radius_range"], "target radius", file_path)
        object_goal_distance = float(np.linalg.norm(obj[:2] - goal[:2]))
        if object_goal_distance + 1e-12 < float(
            ranges["min_object_goal_distance"]
        ):
            raise DatasetValidationError(
                f"{file_path.name} violates minimum object-goal distance"
            )

        lengths.append(int(np.asarray(trajectory["trajectory_length"]).item()))
        path_lengths.append(float(np.asarray(trajectory["ee_path_length"]).item()))
        object_goal_distances.append(object_goal_distance)
        initial_ee = np.asarray(trajectory["initial_ee_position"], dtype=float)
        initial_object = np.asarray(
            trajectory["initial_object_position"], dtype=float
        )
        ee_object_distances.append(float(np.linalg.norm(initial_ee - initial_object)))
        radii.append(radius)
        fingerprint_entries.append(fingerprint_entry(trajectory))

    total_transitions = int(sum(lengths))
    if total_transitions != int(metadata["number_of_total_state_action_pairs"]):
        raise DatasetValidationError("Metadata total state-action count does not match files")
    computed_fingerprint = task_fingerprint(fingerprint_entries)
    if computed_fingerprint != metadata["dataset_fingerprint"]:
        raise DatasetValidationError("Dataset fingerprint does not match trajectory files")

    return {
        "number_of_trajectories": len(files),
        "total_transitions": total_transitions,
        "trajectory_length": _statistics(lengths),
        "ee_path_length": _statistics(path_lengths),
        "object_goal_distance": _statistics(object_goal_distances),
        "initial_ee_object_distance": _statistics(ee_object_distances),
        "target_radius": _statistics(radii),
        "dataset_fingerprint": computed_fingerprint,
    }


def _check_range(
    value: float, bounds: list[float], label: str, file_path: Path
) -> None:
    if not float(bounds[0]) <= value <= float(bounds[1]):
        raise DatasetValidationError(
            f"{file_path.name} has {label}={value} outside {bounds}"
        )


def _statistics(values: list[float] | list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
