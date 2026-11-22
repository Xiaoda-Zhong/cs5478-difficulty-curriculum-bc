#!/usr/bin/env python3
"""Validate, aggregate, tabulate, and plot the Milestone 7 experiment."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.train_curriculum import validate_training_history  # noqa: E402

DATASET_FINGERPRINT = "1b6cec9dda067778630268c307241c62cccbd49cfe849a031877ee00ea938bc0"
DISPLAY = {
    "uniform": "Uniform", "task": "Task", "expert": "Expert",
    "learner": "Learner", "reverse_learner": "Reverse Learner",
}


def main() -> None:
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/default.yaml").read_text())
    methods = list(config["curriculum"]["methods"])
    seeds = [int(seed) for seed in config["curriculum"]["seeds"]]
    result_root = REPOSITORY_ROOT / "outputs/results/curriculum_v1"
    evaluation_root = result_root / "evaluations"
    figure_root = REPOSITORY_ROOT / "outputs/figures/curriculum_v1"
    figure_root.mkdir(parents=True, exist_ok=True)
    groups = json.loads((REPOSITORY_ROOT / "data/metadata/id_eval_task_groups_v1.json").read_text())["groups"]
    group_by_task = {task_id: group for group, ids in groups.items() for task_id in ids}
    initial = {
        seed: _load_evaluation(evaluation_root / "initial" / f"seed_{seed}" / "step_000000.json")
        for seed in seeds
    }
    per_seed = []
    curves: dict[str, dict[int, list[dict[str, Any]]]] = {method: {} for method in methods}
    histories: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in methods}
    failures: dict[str, dict[str, dict[str, int]]] = {}
    for method in methods:
        failures[method] = {}
        for seed in seeds:
            history = json.loads((result_root / method / f"seed_{seed}" / "training_history.json").read_text())
            validate_training_history(history)
            histories[method][seed] = history
            evaluations = [initial[seed]] + [
                _load_evaluation(evaluation_root / method / f"seed_{seed}" / f"step_{int(step):06d}.json")
                for step in history["evaluation_steps"]
            ]
            _validate_curve(evaluations, history, method, seed)
            curves[method][seed] = evaluations
            progress = np.asarray([entry["optimizer_step"] / history["total_optimizer_updates"] for entry in evaluations])
            success = np.asarray([entry["success_rate"] for entry in evaluations])
            auc = float(np.trapezoid(success, progress))
            final = evaluations[-1]
            group_rates = _group_success(final["episodes"], group_by_task)
            record = {
                "method": method, "training_seed": seed,
                "final_successes": int(final["successes"]), "evaluation_tasks": int(final["tasks"]),
                "final_success_rate": float(final["success_rate"]),
                "learning_curve_auc": auc,
                "easy_success_rate": group_rates["easy"],
                "medium_success_rate": group_rates["medium"],
                "hard_success_rate": group_rates["hard"],
                "mean_rollout_length": float(final["mean_rollout_length"]),
                "median_rollout_length": float(final["median_rollout_length"]),
                "mean_final_goal_xy_error": float(final["mean_final_object_goal_xy_error"]),
                "final_validation_loss": float(history["records"][-1]["validation_total_loss"]),
                "initial_state_fingerprint": final["initial_state_fingerprint"],
            }
            per_seed.append(record)
            failures[method][str(seed)] = final["failure_reasons"]

    summary = []
    for method in methods:
        records = [row for row in per_seed if row["method"] == method]
        row: dict[str, Any] = {"method": method, "display_name": DISPLAY[method], "seeds": len(records)}
        for field in (
            "final_success_rate", "learning_curve_auc", "hard_success_rate",
            "mean_final_goal_xy_error", "final_validation_loss",
        ):
            values = np.asarray([float(record[field]) for record in records])
            row[f"mean_{field}"] = float(values.mean())
            row[f"std_{field}"] = float(values.std(ddof=1))
        summary.append(row)
    _write_csv(result_root / "per_seed_results_v1.csv", per_seed)
    _write_csv(result_root / "summary_table_v1.csv", summary)
    payload = {
        "version": 1, "dataset_fingerprint": DATASET_FINGERPRINT,
        "methods": methods, "seeds": seeds, "training_runs": len(per_seed),
        "optimizer_updates_per_run": int(config["curriculum"]["total_optimizer_updates"]),
        "batch_size": int(config["curriculum"]["batch_size"]),
        "learning_rate": float(config["curriculum"]["learning_rate"]),
        "stage_boundaries": config["curriculum"]["stage_boundaries"],
        "evaluation_checkpoint_fractions": [0.0, *config["curriculum"]["evaluation_checkpoint_fractions"]],
        "auc_definition": "trapezoidal integration of rollout success over normalized progress [0,1], including evaluated untrained initialization at progress 0",
        "across_seed_standard_deviation": "sample standard deviation (ddof=1)",
        "evaluation_task_group_sizes": {name: len(ids) for name, ids in groups.items()},
        "per_seed": per_seed, "summary": summary,
        "final_failure_reasons": failures,
    }
    (result_root / "curriculum_summary_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_success_curves(curves, methods, seeds, config, figure_root / "01_learning_curves.png")
    _plot_summary(summary, "mean_final_success_rate", "std_final_success_rate", "Final success rate", figure_root / "02_final_success.png")
    _plot_summary(summary, "mean_learning_curve_auc", "std_learning_curve_auc", "Learning-curve AUC", figure_root / "03_learning_curve_auc.png")
    _plot_group_success(per_seed, methods, figure_root / "04_difficulty_group_success.png")
    _plot_validation(histories, methods, seeds, config, figure_root / "05_validation_loss.png")
    print("Curriculum result integrity check: PASSED")
    for row in summary:
        print(
            f"{row['display_name']}: final={row['mean_final_success_rate']:.3f} +/- {row['std_final_success_rate']:.3f}, "
            f"AUC={row['mean_learning_curve_auc']:.3f}, hard={row['mean_hard_success_rate']:.3f}, "
            f"error={row['mean_mean_final_goal_xy_error']:.4f} m"
        )


def _load_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload["dataset_fingerprint"] != DATASET_FINGERPRINT or payload["tasks"] != 100:
        raise ValueError(f"Invalid curriculum evaluation: {path}")
    if len(payload["episodes"]) != 100 or len({episode["task_id"] for episode in payload["episodes"]}) != 100:
        raise ValueError(f"Incomplete curriculum evaluation: {path}")
    return payload


def _validate_curve(evaluations: list[dict[str, Any]], history: dict[str, Any], method: str, seed: int) -> None:
    expected = [0, *history["evaluation_steps"]]
    if [int(entry["optimizer_step"]) for entry in evaluations] != expected:
        raise ValueError("Evaluation curve uses inconsistent checkpoint positions")
    for entry in evaluations[1:]:
        if entry["method"] != method or int(entry["training_seed"]) != seed:
            raise ValueError("Evaluation method/seed mismatch")
    fingerprints = {entry["initial_state_fingerprint"] for entry in evaluations}
    if fingerprints != {history["initial_state_fingerprint"]}:
        raise ValueError("Evaluation curve does not share the declared initialization")


def _group_success(episodes: list[dict[str, Any]], group_by_task: dict[str, str]) -> dict[str, float]:
    totals = Counter(group_by_task[episode["task_id"]] for episode in episodes)
    successes = Counter(group_by_task[episode["task_id"]] for episode in episodes if episode["success"])
    return {group: successes[group] / totals[group] for group in ("easy", "medium", "hard")}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _plot_success_curves(curves, methods, seeds, config, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        matrix = np.asarray([[entry["success_rate"] for entry in curves[method][seed]] for seed in seeds])
        x = np.asarray([entry["optimizer_step"] / config["curriculum"]["total_optimizer_updates"] for entry in curves[method][seeds[0]]])
        mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        ax.plot(x, mean, marker="o", label=DISPLAY[method]); ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=.13)
    _stage_lines(ax, config); ax.set(xlabel="Normalized optimizer updates", ylabel="Rollout success rate", ylim=(0, 1.03), title="Curriculum learning curves (mean ± SD, 3 seeds)")
    ax.legend(frameon=False, ncol=2); _save(fig, path)


def _plot_summary(summary, mean_key: str, std_key: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(summary))
    means = [row[mean_key] for row in summary]; stds = [row[std_key] for row in summary]
    bars = ax.bar(x, means, yerr=stds, capsize=5)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in means])
    ax.set_xticks(x, [row["display_name"] for row in summary], rotation=20, ha="right")
    ax.set(ylabel=ylabel, ylim=(0, max(1.0, max(np.asarray(means) + np.asarray(stds)) * 1.15)), title=f"{ylabel} across training seeds")
    _save(fig, path)


def _plot_group_success(per_seed, methods, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(methods)); width = .24
    for offset, group in enumerate(("easy", "medium", "hard")):
        means, stds = [], []
        for method in methods:
            values = np.asarray([row[f"{group}_success_rate"] for row in per_seed if row["method"] == method])
            means.append(values.mean()); stds.append(values.std(ddof=1))
        ax.bar(x + (offset - 1) * width, means, width, yerr=stds, capsize=3, label=group.title())
    ax.set_xticks(x, [DISPLAY[method] for method in methods], rotation=20, ha="right")
    ax.set(ylabel="Final success rate", ylim=(0, 1.05), title="Final success by fixed evaluation-task difficulty group")
    ax.legend(frameon=False); _save(fig, path)


def _plot_validation(histories, methods, seeds, config, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        matrix = np.asarray([[record["validation_total_loss"] for record in histories[method][seed]["records"]] for seed in seeds])
        x = np.asarray(histories[method][seeds[0]]["evaluation_steps"]) / config["curriculum"]["total_optimizer_updates"]
        ax.plot(x, matrix.mean(axis=0), marker="o", label=DISPLAY[method])
    _stage_lines(ax, config); ax.set(xlabel="Normalized optimizer updates", ylabel="Validation loss", yscale="log", title="Held-out validation loss (mean, 3 seeds)")
    ax.legend(frameon=False, ncol=2); _save(fig, path)


def _stage_lines(ax, config) -> None:
    for boundary in config["curriculum"]["stage_boundaries"]:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=1)


def _save(fig, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
