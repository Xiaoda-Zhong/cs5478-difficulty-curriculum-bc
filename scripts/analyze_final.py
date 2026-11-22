#!/usr/bin/env python3
"""Validate and summarize the frozen Milestone 8 final benchmark."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.final_benchmark import (  # noqa: E402
    DATASET_FINGERPRINT, METHODS, SEEDS, paired_outcomes,
    paired_task_bootstrap, supported_failure_stage,
)

RESULT_ROOT = ROOT / "outputs/results/final_evaluation_v1"
FIGURE_ROOT = ROOT / "outputs/figures/final_evaluation_v1"
SPLITS = ("final_id", "precision_ood", "spatial_ood")
DISPLAY = {
    "uniform": "Uniform", "task": "Task", "expert": "Expert",
    "learner": "Learner", "reverse_learner": "Reverse Learner",
}
COLORS = {
    "uniform": "#4C78A8", "task": "#F58518", "expert": "#54A24B",
    "learner": "#E45756", "reverse_learner": "#7A5195",
}


def load_result(split: str, method: str, seed: int) -> dict[str, Any]:
    path = RESULT_ROOT / "evaluations" / split / method / f"seed_{seed}" / "final.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing final evaluation: {path}")
    result = json.loads(path.read_text())
    expected = (split, method, seed, DATASET_FINGERPRINT, 20600)
    observed = (
        result.get("evaluation_split"), result.get("method"),
        int(result.get("training_seed", -1)), result.get("dataset_fingerprint"),
        int(result.get("optimizer_step", -1)),
    )
    if observed != expected:
        raise ValueError(f"Final result identity mismatch in {path}: {observed}")
    episodes = result["episodes"]
    if len(episodes) != int(result["tasks"]):
        raise ValueError(f"Episode count mismatch in {path}")
    ids = [str(row["task_id"]) for row in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs in {path}")
    return result


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def style_axis(axis: Any, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)


def grouped_bar(path: Path, title: str, series: dict[str, list[float]], ylabel: str = "Success rate") -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    labels = list(series)
    x = np.arange(len(METHODS)); width = 0.8 / len(labels)
    for index, label in enumerate(labels):
        values = series[label]
        axis.bar(x + (index - (len(labels) - 1) / 2) * width, values, width, label=label, alpha=0.9)
    axis.set_xticks(x, [DISPLAY[m] for m in METHODS], rotation=15, ha="right")
    axis.set_ylim(0, 1.06); axis.set_title(title); style_axis(axis, ylabel)
    axis.legend(frameon=False, ncol=min(3, len(labels)))
    figure.savefig(path, dpi=220); plt.close(figure)


def method_mean_sd_bar(path: Path, title: str, means: list[float], sds: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.bar(np.arange(5), means, yerr=sds, capsize=4, color=[COLORS[m] for m in METHODS])
    axis.set_xticks(np.arange(5), [DISPLAY[m] for m in METHODS], rotation=15, ha="right")
    axis.set_ylim(0, 1.1); axis.set_title(title); style_axis(axis, "Success rate")
    figure.savefig(path, dpi=220); plt.close(figure)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    results = {(split, method, seed): load_result(split, method, seed)
               for split in SPLITS for method in METHODS for seed in SEEDS}

    # The pairing claim is validated against every one of the 15 evaluations per split.
    for split in SPLITS:
        expected_ids = [row["task_id"] for row in results[(split, "uniform", SEEDS[0])]["episodes"]]
        for method in METHODS:
            for seed in SEEDS:
                ids = [row["task_id"] for row in results[(split, method, seed)]["episodes"]]
                if ids != expected_ids:
                    raise ValueError(f"Task IDs or ordering differ for {split}/{method}/{seed}")

    groups_payload = json.loads((ROOT / "data/metadata/final_id_difficulty_groups_v1.json").read_text())
    groups = {name: set(ids) for name, ids in groups_payload["groups"].items()}
    per_seed_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    id_group_rates: dict[tuple[str, int, str], float] = {}
    for split in SPLITS:
        for method in METHODS:
            for seed in SEEDS:
                result = results[(split, method, seed)]; episodes = result["episodes"]
                raw_failures = Counter(row.get("failure_reason") or "none" for row in episodes if not row["success"])
                stages = Counter(supported_failure_stage(row) for row in episodes)
                row: dict[str, Any] = {
                    "method": method, "training_seed": seed, "evaluation_split": split,
                    "number_of_tasks": len(episodes),
                    "number_successful": sum(bool(ep["success"]) for ep in episodes),
                    "success_rate": result["success_rate"],
                    "mean_rollout_length": result["mean_rollout_length"],
                    "median_rollout_length": result["median_rollout_length"],
                    "mean_final_goal_error": result["mean_final_object_goal_xy_error"],
                    "failure_counts": json.dumps(dict(sorted(raw_failures.items())), sort_keys=True),
                    "supported_stage_counts": json.dumps(dict(sorted(stages.items())), sort_keys=True),
                }
                if split == "final_id":
                    by_id = {str(ep["task_id"]): ep for ep in episodes}
                    for group in ("easy", "medium", "hard"):
                        rate = float(np.mean([bool(by_id[task_id]["success"]) for task_id in groups[group]]))
                        row[f"{group}_success_rate"] = rate
                        id_group_rates[(method, seed, group)] = rate
                else:
                    row.update({f"{group}_success_rate": "" for group in ("easy", "medium", "hard")})
                per_seed_rows.append(row)
                for episode in episodes:
                    task_rows.append({
                        "method": method, "training_seed": seed, "evaluation_split": split,
                        "task_id": episode["task_id"], "success": int(bool(episode["success"])),
                        "steps": episode["steps"],
                        "final_goal_xy_error": episode["final_object_goal_xy_error"],
                        "failure_reason": episode.get("failure_reason") or "",
                        "supported_stage": supported_failure_stage(episode),
                    })
    save_csv(RESULT_ROOT / "per_seed_split_results_v1.csv", per_seed_rows)
    save_csv(RESULT_ROOT / "task_level_results_v1.csv", task_rows)

    paired_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        for method_index, method in enumerate(METHODS[1:], 1):
            for seed_index, seed in enumerate(SEEDS):
                method_eps = results[(split, method, seed)]["episodes"]
                uniform_eps = results[(split, "uniform", seed)]["episodes"]
                outcome = paired_outcomes(method_eps, uniform_eps)
                bootstrap = paired_task_bootstrap(
                    method_eps, uniform_eps, samples=1000,
                    seed=75478 + 100 * split_index + 10 * method_index + seed_index,
                )
                paired_rows.append({
                    "evaluation_split": split, "method": method, "reference_method": "uniform",
                    "training_seed": seed, **outcome,
                    "bootstrap_ci_lower": bootstrap["ci_lower"],
                    "bootstrap_ci_upper": bootstrap["ci_upper"],
                    "bootstrap_samples": bootstrap["samples"],
                    "uncertainty_scope": bootstrap["interpretation"],
                })
    save_csv(RESULT_ROOT / "paired_comparisons_v1.csv", paired_rows)

    dev_rows = list(csv.DictReader((ROOT / "outputs/results/curriculum_v1/per_seed_results_v1.csv").open()))
    dev = {(row["method"], int(row["training_seed"])): float(row["final_success_rate"]) for row in dev_rows}
    dev_final_rows: list[dict[str, Any]] = []
    degradation_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            final_id = float(results[("final_id", method, seed)]["success_rate"])
            dev_final_rows.append({
                "method": method, "training_seed": seed, "development_id_success_rate": dev[(method, seed)],
                "final_id_success_rate": final_id, "final_minus_development": final_id - dev[(method, seed)],
            })
            for split in SPLITS[1:]:
                ood = float(results[(split, method, seed)]["success_rate"])
                degradation_rows.append({
                    "method": method, "training_seed": seed, "evaluation_split": split,
                    "final_id_success_rate": final_id, "ood_success_rate": ood,
                    "id_minus_ood_degradation": final_id - ood,
                })
    save_csv(RESULT_ROOT / "development_vs_final_v1.csv", dev_final_rows)
    save_csv(RESULT_ROOT / "ood_degradation_v1.csv", degradation_rows)

    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        summary: dict[str, Any] = {"method": method, "display_name": DISPLAY[method], "training_seeds": 3}
        dev_values = [dev[(method, seed)] for seed in SEEDS]
        summary["development_id_mean"], summary["development_id_sample_sd"] = mean_sd(dev_values)
        for split in SPLITS:
            values = [float(results[(split, method, seed)]["success_rate"]) for seed in SEEDS]
            summary[f"{split}_mean"], summary[f"{split}_sample_sd"] = mean_sd(values)
        for group in ("easy", "medium", "hard"):
            values = [id_group_rates[(method, seed, group)] for seed in SEEDS]
            summary[f"final_id_{group}_mean"], summary[f"final_id_{group}_sample_sd"] = mean_sd(values)
        final_errors = [float(results[("final_id", method, seed)]["mean_final_object_goal_xy_error"])
                        for seed in SEEDS]
        summary["final_id_mean_final_goal_error"], summary["final_id_mean_final_goal_error_sample_sd"] = mean_sd(final_errors)
        errors = [float(results[(split, method, seed)]["mean_final_object_goal_xy_error"])
                  for split in SPLITS for seed in SEEDS]
        summary["all_splits_mean_final_goal_xy_error"] = float(np.mean(errors))
        summary["uncertainty"] = "sample standard deviation across 3 training seeds (ddof=1)"
        summary_rows.append(summary)
    save_csv(RESULT_ROOT / "summary_table_v1.csv", summary_rows)

    # Figures use seed-level variation; error bars are sample SD, not confidence intervals.
    means = [row["final_id_mean"] for row in summary_rows]; sds = [row["final_id_sample_sd"] for row in summary_rows]
    method_mean_sd_bar(FIGURE_ROOT / "final_id_success_v1.png", "Fresh held-out ID performance (mean ± sample SD, n=3 seeds)", means, sds)

    grouped_bar(FIGURE_ROOT / "final_id_difficulty_groups_v1.png", "Fresh ID performance by pre-frozen task difficulty", {
        group.title(): [next(row for row in summary_rows if row["method"] == method)[f"final_id_{group}_mean"] for method in METHODS]
        for group in ("easy", "medium", "hard")
    })
    grouped_bar(FIGURE_ROOT / "ood_success_v1.png", "Out-of-distribution performance", {
        "Precision OOD": [next(row for row in summary_rows if row["method"] == method)["precision_ood_mean"] for method in METHODS],
        "Spatial OOD": [next(row for row in summary_rows if row["method"] == method)["spatial_ood_mean"] for method in METHODS],
    })
    method_mean_sd_bar(
        FIGURE_ROOT / "precision_ood_success_v1.png", "Precision OOD performance (mean ± sample SD, n=3 seeds)",
        [next(row for row in summary_rows if row["method"] == method)["precision_ood_mean"] for method in METHODS],
        [next(row for row in summary_rows if row["method"] == method)["precision_ood_sample_sd"] for method in METHODS],
    )
    method_mean_sd_bar(
        FIGURE_ROOT / "spatial_ood_success_v1.png", "Spatial OOD performance (mean ± sample SD, n=3 seeds)",
        [next(row for row in summary_rows if row["method"] == method)["spatial_ood_mean"] for method in METHODS],
        [next(row for row in summary_rows if row["method"] == method)["spatial_ood_sample_sd"] for method in METHODS],
    )

    figure, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    offsets = np.linspace(-0.24, 0.24, len(METHODS) - 1)
    for offset, method in zip(offsets, METHODS[1:]):
        rows = [row for row in paired_rows if row["evaluation_split"] == "final_id" and row["method"] == method]
        values = [float(row["success_rate_difference"]) for row in rows]
        lower = [value - float(row["bootstrap_ci_lower"]) for value, row in zip(values, rows)]
        upper = [float(row["bootstrap_ci_upper"]) - value for value, row in zip(values, rows)]
        axis.errorbar(np.arange(3) + offset, values, yerr=[lower, upper], marker="o", capsize=3,
                      linestyle="none", label=DISPLAY[method], color=COLORS[method])
    axis.axhline(0, color="black", linewidth=0.9); axis.set_xticks(np.arange(3), [str(seed) for seed in SEEDS])
    axis.set_title("Paired fresh-ID difference from seed-matched Uniform")
    style_axis(axis, "Success-rate difference"); axis.legend(frameon=False, ncol=2)
    figure.savefig(FIGURE_ROOT / "paired_id_difference_v1.png", dpi=220); plt.close(figure)

    grouped_bar(FIGURE_ROOT / "development_vs_final_v1.png", "Development-task and fresh final-ID performance", {
        "Development ID": [next(row for row in summary_rows if row["method"] == method)["development_id_mean"] for method in METHODS],
        "Final ID": [next(row for row in summary_rows if row["method"] == method)["final_id_mean"] for method in METHODS],
    })

    stage_names = ["failed_to_reach_object", "grasp_or_lift_failure", "transport_failure",
                   "place_approach_failure", "release_failure", "timeout_unclassified", "object_left_workspace"]
    figure, axis = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    bottoms = np.zeros(len(METHODS))
    for stage in stage_names:
        values = []
        for method in METHODS:
            counts = Counter()
            for seed in SEEDS:
                counts.update(supported_failure_stage(ep) for ep in results[("final_id", method, seed)]["episodes"] if not ep["success"])
            values.append(counts[stage] / (200 * len(SEEDS)))
        axis.bar(np.arange(5), values, bottom=bottoms, label=stage.replace("_", " "))
        bottoms += np.asarray(values)
    axis.set_xticks(np.arange(5), [DISPLAY[m] for m in METHODS], rotation=15, ha="right")
    axis.set_title("Fresh-ID failures by supported physical stage")
    style_axis(axis, "Fraction of all rollouts"); axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(FIGURE_ROOT / "failure_stages_v1.png", dpi=220); plt.close(figure)

    final_summary = {
        "version": 1, "dataset_fingerprint": DATASET_FINGERPRINT,
        "checkpoint_selection": "M7 final.pt at optimizer update 20,600; no retraining",
        "task_sets_frozen_before_policy_evaluation": True,
        "spatial_ood_included_after_expert_feasibility_pilot": True,
        "rollout_horizon": 400, "training_seeds": list(SEEDS),
        "splits": {"final_id": 200, "precision_ood": 100, "spatial_ood": 100},
        "uncertainty": {
            "aggregate": "sample standard deviation across the three training seeds (ddof=1)",
            "paired": "fixed-seed 1,000-sample task-level paired bootstrap; conditional on each trained model and not training-seed uncertainty",
            "caution": "Three training seeds are insufficient for strong claims of method superiority.",
        },
        "summary": summary_rows,
        "paired_comparisons": paired_rows,
        "development_vs_final": dev_final_rows,
        "ood_degradation": degradation_rows,
        "failure_stage_definition": "Stages are inferred only from logged physical-event evidence; unsupported failures remain timeout_unclassified.",
        "representative_failure_traces": "Up to three failed trajectory traces are retained inside each per-checkpoint final.json.",
    }
    (RESULT_ROOT / "final_summary_v1.json").write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n")
    print(f"Validated 45 evaluations and {len(task_rows)} task-level rollouts")
    for row in summary_rows:
        print(f"{row['display_name']}: ID={row['final_id_mean']:.3f}±{row['final_id_sample_sd']:.3f}, "
              f"precision={row['precision_ood_mean']:.3f}±{row['precision_ood_sample_sd']:.3f}, "
              f"spatial={row['spatial_ood_mean']:.3f}±{row['spatial_ood_sample_sd']:.3f}")


if __name__ == "__main__":
    main()
