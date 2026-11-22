#!/usr/bin/env python3
"""Assemble, validate, rank, and summarize all M5 difficulty scores."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from difficulty.normalization import (  # noqa: E402
    deterministic_ranking,
    equal_weight_composite,
    spearman_correlation,
    standardize,
)


SCORE_COLUMNS = ("task_difficulty", "expert_difficulty", "learner_difficulty")
CSV_COLUMNS = (
    "trajectory_id",
    "task_ee_object_distance",
    "task_object_goal_distance",
    "task_inverse_target_radius",
    "task_difficulty",
    "expert_duration",
    "expert_ee_path_length",
    "expert_action_magnitude",
    "expert_difficulty",
    "learner_motion_error",
    "learner_gripper_error",
    "learner_difficulty",
    "learner_combined_difficulty",
)


def main() -> None:
    result_dir = REPOSITORY_ROOT / "outputs/results/difficulty"
    task_payload = _load(result_dir / "task_difficulty_v1.json")
    expert_payload = _load(result_dir / "expert_difficulty_v1.json")
    learner_payload = _load(result_dir / "learner_difficulty_v1.json")
    split = _load(REPOSITORY_ROOT / "data/metadata/bc_split_v1.json")
    payloads = (task_payload, expert_payload, learner_payload)
    fingerprints = {payload["dataset_fingerprint"] for payload in payloads}
    if fingerprints != {split["dataset_fingerprint"]}:
        raise ValueError("Difficulty result dataset fingerprints do not match")
    expected_ids = split["train_trajectory_ids"]
    if any(payload["trajectory_ids"] != expected_ids for payload in payloads):
        raise ValueError("Difficulty methods do not use the same ordered training IDs")

    maps = [
        {record["trajectory_id"]: record for record in payload["records"]}
        for payload in payloads
    ]
    if any(set(mapping) != set(expected_ids) for mapping in maps):
        raise ValueError("Difficulty results have missing or duplicate trajectory IDs")
    records = []
    for trajectory_id in expected_ids:
        record = {"trajectory_id": trajectory_id}
        for mapping in maps:
            record.update(mapping[trajectory_id])
        records.append(record)
    _validate_records(records, task_payload, expert_payload)

    csv_path = result_dir / "difficulty_scores_v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    rankings = {
        score_name.replace("_difficulty", ""): deterministic_ranking(
            {record["trajectory_id"]: float(record[score_name]) for record in records}
        )
        for score_name in SCORE_COLUMNS
    }
    rankings_path = result_dir / "difficulty_rankings_v1.json"
    rankings_path.write_text(
        json.dumps(
            {
                "dataset_fingerprint": split["dataset_fingerprint"],
                "trajectory_ids": expected_ids,
                "tie_breaker": "trajectory_id ascending",
                "ranking_direction": "ascending difficulty (easy to hard)",
                "rankings": rankings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    score_matrix = np.asarray(
        [[record[name] for name in SCORE_COLUMNS] for record in records], dtype=float
    )
    summary = {
        name: _summary(score_matrix[:, index])
        for index, name in enumerate(SCORE_COLUMNS)
    }
    correlation = np.eye(3, dtype=float)
    for row in range(3):
        for column in range(row + 1, 3):
            value = spearman_correlation(
                score_matrix[:, row], score_matrix[:, column]
            )
            correlation[row, column] = correlation[column, row] = value

    metadata = {
        "dataset_fingerprint": split["dataset_fingerprint"],
        "score_definition_version": 1,
        "date_generated_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_count": len(records),
        "ordered_trajectory_ids": expected_ids,
        "definitions": {
            "task": task_payload["definition"],
            "expert": expert_payload["definition"],
            "learner": learner_payload["definition"],
            "learner_gripper": learner_payload["gripper_definition"],
        },
        "normalization_parameters": {
            "task": task_payload["normalization"],
            "expert": expert_payload["normalization"],
            "learner_combined": learner_payload["combined_normalization"],
        },
        "oof": {
            "fold_assignment_file": learner_payload["fold_assignment_file"],
            "training_epochs": learner_payload["training_epochs"],
            "training_device": learner_payload["training_device"],
            "fold_seed": learner_payload["fold_seed"],
            "number_of_folds": learner_payload["number_of_folds"],
        },
        "expert_component_correlation": {
            "ee_path_vs_action_magnitude_pearson": expert_payload[
                "ee_path_action_magnitude_pearson_correlation"
            ]
        },
        "summary_statistics": summary,
        "spearman_score_order": list(SCORE_COLUMNS),
        "spearman_correlation_matrix": correlation.tolist(),
    }
    metadata_path = result_dir / "difficulty_metadata_v1.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    figure_path = REPOSITORY_ROOT / "outputs/figures/difficulty/correlation_heatmap.png"
    _plot_heatmap(correlation, figure_path)

    print("Difficulty integrity check: PASSED")
    print(f"Trajectories: {len(records)}")
    for score_name, statistics in summary.items():
        print(
            f"{score_name}: mean={statistics['mean']:.6f}, "
            f"std={statistics['std']:.6f}, median={statistics['median']:.6f}, "
            f"min={statistics['min']:.6f}, max={statistics['max']:.6f}"
        )
    print("Spearman correlation matrix:")
    print(correlation)
    for name, ranking in rankings.items():
        groups = ranking["groups"]
        print(
            f"{name}: easy={len(groups['easy'])}, medium={len(groups['medium'])}, "
            f"hard={len(groups['hard'])}"
        )
    print(f"Scores: {csv_path}")
    print(f"Rankings: {rankings_path}")
    print(f"Metadata: {metadata_path}")


def _validate_records(records: list[dict], task_payload: dict, expert_payload: dict) -> None:
    if len(records) != 240 or len({record["trajectory_id"] for record in records}) != 240:
        raise ValueError("Exactly 240 unique trajectories must be scored")
    numeric = np.asarray(
        [[float(record[column]) for column in CSV_COLUMNS[1:]] for record in records]
    )
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Difficulty scores and components must all be finite")
    task_raw = numeric[:, 0:3]
    expected_task = equal_weight_composite(
        standardize(task_raw, task_payload["normalization"])
    )
    np.testing.assert_allclose(expected_task, numeric[:, 3], atol=1e-10)
    expert_raw = numeric[:, 4:7]
    expected_expert = equal_weight_composite(
        standardize(expert_raw, expert_payload["normalization"])
    )
    np.testing.assert_allclose(expected_expert, numeric[:, 7], atol=1e-10)
    np.testing.assert_allclose(numeric[:, 10], numeric[:, 8], atol=0.0)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _plot_heatmap(correlation: np.ndarray, output_path: Path) -> None:
    labels = ["Task", "Expert", "Learner"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(3), labels=labels)
    axis.set_yticks(range(3), labels=labels)
    for row in range(3):
        for column in range(3):
            axis.text(column, row, f"{correlation[row, column]:.2f}", ha="center", va="center")
    axis.set_title("Difficulty score Spearman correlations")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
