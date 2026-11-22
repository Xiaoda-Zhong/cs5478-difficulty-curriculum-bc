#!/usr/bin/env python3
"""Analyze M5 difficulty scores as predictors of closed-loop OOF failure."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.difficulty_diagnostics import (  # noqa: E402
    failure_reason_counts, group_outcome_rates, join_by_trajectory_id,
    summarize_fold_outcomes, validate_checkpoint_for_fold,
)
from analysis.failure_prediction import (  # noqa: E402
    auprc, auroc, bootstrap_auroc, failure_labels,
    paired_bootstrap_auroc_differences,
)
from analysis.plotting import (  # noqa: E402
    auroc_comparison, correlation_heatmap, group_failure_bars,
    learner_vs_duration, pairwise_difficulty_scatter, phase_error_bars,
)
from data.trajectory import load_trajectory  # noqa: E402
from difficulty.normalization import (  # noqa: E402
    equal_weight_composite, spearman_correlation, standardize,
)
from models.bc_policy import load_policy_checkpoint  # noqa: E402
from training.train_bc import select_device  # noqa: E402

PRIMARY_SCORES = ("task_difficulty", "expert_difficulty", "learner_difficulty")
TASK_COMPONENTS = (
    "task_ee_object_distance", "task_object_goal_distance", "task_inverse_target_radius",
)


def main() -> None:
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["analysis"]["bootstrap_seed"])
    result_dir = REPOSITORY_ROOT / "outputs/results/difficulty_analysis"
    figure_dir = REPOSITORY_ROOT / "outputs/figures/difficulty_analysis"
    rollout_rows = _read_rollouts(result_dir / "oof_rollout_results_v1.csv")
    score_rows = _read_csv(REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_scores_v1.csv")
    rows = join_by_trajectory_id(score_rows, rollout_rows)
    if len(rows) != 240:
        raise ValueError("Analysis requires exactly 240 matched OOF rollout rows")
    metadata = _read_json(REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_metadata_v1.json")
    rankings = _read_json(REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json")
    rollout_metadata = _read_json(result_dir / "oof_rollout_metadata_v1.json")
    fingerprints = {metadata["dataset_fingerprint"], rankings["dataset_fingerprint"], rollout_metadata["dataset_fingerprint"]}
    if len(fingerprints) != 1:
        raise ValueError("Milestone 5 and rollout dataset fingerprints do not match")

    labels = failure_labels([row["success"] for row in rows])
    prediction = {}
    score_map = {}
    for name in PRIMARY_SCORES:
        values = np.asarray([float(row[name]) for row in rows])
        score_map[name] = values
        prediction[name] = {
            "auroc": bootstrap_auroc(labels, values, samples=samples, seed=seed),
            "auprc": auprc(labels, values),
            "per_fold_auroc": _per_fold_auroc(rows, labels, values),
        }
    paired = paired_bootstrap_auroc_differences(labels, score_map, samples=samples, seed=seed)
    group_rates = {
        name: group_outcome_rates(rows, rankings["rankings"][name]["groups"])
        for name in ("task", "expert", "learner")
    }

    duration = np.asarray([float(row["expert_duration"]) for row in rows])
    learner = score_map["learner_difficulty"]
    fold_learner = {}
    for fold_id in sorted({int(row["fold_id"]) for row in rows}):
        values = learner[np.asarray([int(row["fold_id"]) == fold_id for row in rows])]
        fold_learner[str(fold_id)] = _summary(values)
    top_outliers = sorted(rows, key=lambda row: (-float(row["learner_difficulty"]), row["trajectory_id"]))[:10]
    _write_top_outliers(result_dir / "top_learner_outliers_v1.csv", top_outliers)
    phase_rows = _phase_diagnostics(rows)
    _write_csv(result_dir / "phase_diagnostics_v1.csv", phase_rows)

    expert_norm = metadata["normalization_parameters"]["expert"]
    expert_simple = equal_weight_composite(standardize(
        np.asarray([[float(row["expert_duration"]), float(row["expert_ee_path_length"])] for row in rows]),
        {"mean": expert_norm["mean"][:2], "std": expert_norm["std"][:2], "epsilon": expert_norm["epsilon"]},
    ))
    task_norm = metadata["normalization_parameters"]["task"]
    geometry_only = equal_weight_composite(standardize(
        np.asarray([[float(row["task_ee_object_distance"]), float(row["task_object_goal_distance"])] for row in rows]),
        {"mean": task_norm["mean"][:2], "std": task_norm["std"][:2], "epsilon": task_norm["epsilon"]},
    ))
    sensitivity_aurocs = {
        "Expert simplified": bootstrap_auroc(labels, expert_simple, samples=samples, seed=seed),
        "Task geometry-only": bootstrap_auroc(labels, geometry_only, samples=samples, seed=seed),
    }
    component_correlations = {}
    for component in TASK_COMPONENTS:
        values = [float(row[component]) for row in rows]
        component_correlations[component] = {
            "with_expert": spearman_correlation(values, score_map["expert_difficulty"]),
            "with_learner": spearman_correlation(values, learner),
            "with_failure": spearman_correlation(values, labels),
        }

    matrix_names = [*PRIMARY_SCORES, "failure"]
    arrays = [score_map[name] for name in PRIMARY_SCORES] + [labels]
    correlation = np.asarray([[spearman_correlation(left, right) for right in arrays] for left in arrays])
    outcome_summary = {
        "tasks": len(rows), "successes": int((1 - labels).sum()), "failures": int(labels.sum()),
        "success_rate": float((1 - labels).mean()), "failure_prevalence": float(labels.mean()),
        "mean_rollout_steps": float(np.mean([float(row["rollout_steps"]) for row in rows])),
        "mean_final_object_goal_xy_error": float(np.mean([float(row["final_object_goal_xy_error"]) for row in rows])),
        "failure_reasons": failure_reason_counts(rows), "per_fold": summarize_fold_outcomes(rows),
    }
    report = {
        "version": 1, "date_generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": next(iter(fingerprints)),
        "bootstrap": {"samples": samples, "seed": seed, "method": "stratified percentile 95% CI"},
        "outcomes": outcome_summary, "failure_prediction": prediction,
        "paired_auroc_differences": paired, "difficulty_group_outcomes": group_rates,
        "negative_correlation_diagnostics": {
            "duration_vs_learner_spearman": spearman_correlation(duration, learner),
            "learner_by_fold": fold_learner, "phase_error": phase_rows,
            "top_10_learner_outlier_ids": [row["trajectory_id"] for row in top_outliers],
        },
        "expert_simplified_sensitivity": {
            "definition": "equal mean of M5-standardized duration and EE path length",
            "spearman_with_original_expert": spearman_correlation(expert_simple, score_map["expert_difficulty"]),
            "spearman_with_task": spearman_correlation(expert_simple, score_map["task_difficulty"]),
            "spearman_with_learner": spearman_correlation(expert_simple, learner),
            "failure_auroc": sensitivity_aurocs["Expert simplified"],
        },
        "task_component_sensitivity": {
            "component_correlations": component_correlations,
            "geometry_only_definition": "equal mean of M5-standardized EE-object and object-goal distances",
            "geometry_only_spearman_with_task": spearman_correlation(geometry_only, score_map["task_difficulty"]),
            "geometry_only_spearman_with_expert": spearman_correlation(geometry_only, score_map["expert_difficulty"]),
            "geometry_only_spearman_with_learner": spearman_correlation(geometry_only, learner),
            "geometry_only_failure_auroc": sensitivity_aurocs["Task geometry-only"],
            "observation_limitation": "Target radius is absent from the 10D policy observation, so inverse-radius difficulty cannot directly affect policy inputs.",
        },
        "spearman_order": matrix_names, "spearman_matrix": correlation.tolist(),
    }
    summary_path = result_dir / "analysis_summary_v1.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    correlation_heatmap(correlation, ["Task", "Expert", "Learner", "Failure"], figure_dir / "01_correlation_heatmap.png")
    pairwise_difficulty_scatter(rows, figure_dir / "02_pairwise_difficulty_scatter.png")
    group_failure_bars(group_rates, figure_dir / "03_task_group_failure_rates.png")
    plot_aurocs = {name.replace("_difficulty", "").title(): prediction[name]["auroc"] for name in PRIMARY_SCORES}
    auroc_comparison(plot_aurocs, figure_dir / "04_auroc_comparison.png")
    learner_vs_duration(rows, figure_dir / "05_learner_vs_duration.png")
    phase_error_bars(phase_rows, figure_dir / "06_phase_error.png")

    print(json.dumps(outcome_summary, indent=2, sort_keys=True))
    for name in PRIMARY_SCORES:
        print(f"{name}: AUROC={prediction[name]['auroc']['estimate']:.4f}, AUPRC={prediction[name]['auprc']:.4f}")
    print(f"Analysis: {summary_path}")
    print(f"Figures: {figure_dir}")


def _phase_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folds = _read_json(REPOSITORY_ROOT / "data/metadata/learner_oof_folds_v1.json")
    fold_map = {int(fold["fold_index"]): fold for fold in folds["folds"]}
    aggregate: dict[str, list[float]] = defaultdict(list)
    total_error = 0.0
    for fold_id in sorted(fold_map):
        checkpoint_path = REPOSITORY_ROOT / f"outputs/checkpoints/learner_oof/fold_{fold_id}.pt"
        policy, checkpoint = load_policy_checkpoint(checkpoint_path, "cpu")
        validate_checkpoint_for_fold(checkpoint, fold_map[fold_id])
        motion_scale = float(checkpoint["motion_scale"])
        for row in rows:
            if int(row["fold_id"]) != fold_id:
                continue
            trajectory = load_trajectory(REPOSITORY_ROOT / f"data/demonstrations/{row['trajectory_id']}.npz")
            predictions, _ = policy.predict_batch_with_logits(np.asarray(trajectory["observations"], dtype=np.float32))
            targets = np.asarray(trajectory["actions"], dtype=np.float64)[:, :3]
            errors = np.sum(((predictions - targets) / motion_scale) ** 2, axis=1)
            total_error += float(errors.sum())
            for phase, error in zip(np.asarray(trajectory["phases"]).astype(str), errors, strict=True):
                aggregate[phase].append(float(error))
    return [
        {"phase": phase, "steps": len(values), "mean_motion_error": float(np.mean(values)),
         "error_contribution": float(np.sum(values) / total_error)}
        for phase, values in sorted(aggregate.items())
    ]


def _per_fold_auroc(rows: list[dict[str, Any]], labels: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    result = {}
    folds = np.asarray([int(row["fold_id"]) for row in rows])
    for fold_id in sorted(set(folds.tolist())):
        mask = folds == fold_id
        result[str(fold_id)] = auroc(labels[mask], values[mask])
    return result


def _summary(values: np.ndarray) -> dict[str, float]:
    return {name: float(function(values)) for name, function in {
        "mean": np.mean, "median": np.median, "std": np.std, "min": np.min, "max": np.max,
    }.items()}


def _write_top_outliers(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["trajectory_id", "fold_id", "task_difficulty", "expert_difficulty", "learner_difficulty", "trajectory_length", "object_goal_distance", "target_radius"]
    formatted = []
    for row in rows:
        values = dict(row)
        values["trajectory_length"] = int(float(row["expert_duration"]))
        values["object_goal_distance"] = row["task_object_goal_distance"]
        formatted.append({key: values[key] for key in fields})
    _write_csv(path, formatted)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _read_rollouts(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    for row in rows:
        row["success"] = row["success"] == "1"
        row["failure"] = row["failure"] == "1"
        row["fold_id"] = int(row["fold_id"])
        row["rollout_steps"] = int(row["rollout_steps"])
        row["final_object_goal_xy_error"] = float(row["final_object_goal_xy_error"])
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
