#!/usr/bin/env python3
"""Validate and summarize the frozen demonstration dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.dataset import inspect_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    statistics = inspect_dataset(
        REPOSITORY_ROOT / "data" / "demonstrations",
        REPOSITORY_ROOT / "data" / "metadata" / "dataset_metadata.json",
        expected_count=args.expected_count,
    )

    print("Dataset integrity check: PASSED")
    print(f"Trajectories: {statistics['number_of_trajectories']}")
    print(f"Total transitions: {statistics['total_transitions']}")
    for label, key in (
        ("Trajectory length", "trajectory_length"),
        ("EE path length (m)", "ee_path_length"),
        ("Object-goal distance (m)", "object_goal_distance"),
        ("Initial EE-object distance (m)", "initial_ee_object_distance"),
        ("Target radius (m)", "target_radius"),
    ):
        values = statistics[key]
        print(
            f"{label}: mean={values['mean']:.4f}, std={values['std']:.4f}, "
            f"min={values['min']:.4f}, max={values['max']:.4f}"
        )
    print(f"Dataset fingerprint: {statistics['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
