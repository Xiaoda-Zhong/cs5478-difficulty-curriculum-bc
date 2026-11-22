#!/usr/bin/env python3
"""Compute task-geometry difficulty for the 240 BC training trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.trajectory import load_trajectory, validate_trajectory  # noqa: E402
from difficulty.normalization import (  # noqa: E402
    equal_weight_composite,
    fit_standardizer,
    standardize,
)
from difficulty.task import TASK_COMPONENT_NAMES, task_components  # noqa: E402


def main() -> None:
    split = json.loads(
        (REPOSITORY_ROOT / "data/metadata/bc_split_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    trajectory_ids = split["train_trajectory_ids"]
    records = []
    for trajectory_id in trajectory_ids:
        trajectory = load_trajectory(
            REPOSITORY_ROOT / "data/demonstrations" / f"{trajectory_id}.npz"
        )
        validate_trajectory(trajectory, require_success=True)
        records.append({"trajectory_id": trajectory_id, **task_components(trajectory)})

    raw = np.asarray(
        [[record[name] for name in TASK_COMPONENT_NAMES] for record in records],
        dtype=np.float64,
    )
    normalization = fit_standardizer(
        raw,
        TASK_COMPONENT_NAMES,
        float(config["difficulty"]["normalization_epsilon"]),
    )
    composite = equal_weight_composite(standardize(raw, normalization))
    for record, score in zip(records, composite, strict=True):
        record["task_difficulty"] = float(score)

    output = REPOSITORY_ROOT / "outputs/results/difficulty/task_difficulty_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset_fingerprint": split["dataset_fingerprint"],
                "trajectory_ids": trajectory_ids,
                "definition": "equal mean of training-standardized EE-object distance, object-goal distance, and inverse target radius",
                "normalization": normalization,
                "records": records,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Task difficulty: scored {len(records)} trajectories")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
