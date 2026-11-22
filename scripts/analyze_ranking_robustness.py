#!/usr/bin/env python3
"""Analyze the exploratory Milestone 9 learner-ranking robustness study."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.final_benchmark import paired_outcomes  # noqa: E402
from evaluation.ranking_robustness import (  # noqa: E402
    CORRUPTION_LEVELS, DATASET_FINGERPRINT, RANKING_SEED, TRAINING_SEEDS,
    sha256_file,
)
from training.train_curriculum import validate_training_history  # noqa: E402

LEVELS = (0, *CORRUPTION_LEVELS)
STEPS = (2060, 5150, 10300, 15450, 20600)
RESULT_ROOT = ROOT / "outputs/results/ranking_robustness_v1"
FIGURE_ROOT = ROOT / "outputs/figures/ranking_robustness_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def history(level: int, seed: int) -> dict[str, Any]:
    path = (ROOT / f"outputs/results/curriculum_v1/reverse_learner/seed_{seed}/training_history.json"
            if level == 0 else RESULT_ROOT / f"corruption_{level:03d}/seed_{seed}/training_history.json")
    result = read_json(path); validate_training_history(result)
    if int(result["total_optimizer_updates"]) != 20600 or list(result["evaluation_steps"]) != list(STEPS):
        raise ValueError(f"Mismatched training protocol: {path}")
    return result


def evaluation(level: int, seed: int, step: int) -> dict[str, Any]:
    path = (ROOT / f"outputs/results/curriculum_v1/evaluations/reverse_learner/seed_{seed}/step_{step:06d}.json"
            if level == 0 else RESULT_ROOT / f"evaluations/corruption_{level:03d}/seed_{seed}/step_{step:06d}.json")
    result = read_json(path)
    if result["dataset_fingerprint"] != DATASET_FINGERPRINT or int(result["optimizer_step"]) != step:
        raise ValueError(f"Invalid evaluation identity: {path}")
    if int(result["training_seed"]) != seed or int(result["tasks"]) != 100 or len(result["episodes"]) != 100:
        raise ValueError(f"Invalid evaluation size or seed: {path}")
    return result


def initial(seed: int) -> dict[str, Any]:
    return read_json(ROOT / f"outputs/results/curriculum_v1/evaluations/initial/seed_{seed}/step_000000.json")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def finish_plot(figure: Any, axis: Any, path: Path, ylabel: str) -> None:
    axis.set_xlabel("Learner-ranking corruption")
    axis.set_ylabel(ylabel); axis.set_xticks(LEVELS, [f"{level}%" for level in LEVELS])
    axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(path, dpi=220, bbox_inches="tight"); plt.close(figure)


def summary_plot(summary: list[dict[str, Any]], mean_key: str, sd_key: str, ylabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.7))
    means = [row[mean_key] for row in summary]; sds = [row[sd_key] for row in summary]
    axis.errorbar(LEVELS, means, yerr=sds, marker="o", linewidth=2, capsize=5, color="#7A5195")
    axis.set_title(f"{ylabel} vs learner-ranking corruption (mean ± sample SD)")
    if "success" in ylabel.lower(): axis.set_ylim(0, 1.05)
    finish_plot(figure, axis, path, ylabel)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True); FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    groups = read_json(ROOT / "data/metadata/id_eval_task_groups_v1.json")["groups"]
    hard_ids = set(groups["hard"])
    histories = {(level, seed): history(level, seed) for level in LEVELS for seed in TRAINING_SEEDS}
    curves = {(level, seed): [initial(seed), *[evaluation(level, seed, step) for step in STEPS]]
              for level in LEVELS for seed in TRAINING_SEEDS}
    per_seed: list[dict[str, Any]] = []
    for level in LEVELS:
        for seed in TRAINING_SEEDS:
            entries = curves[(level, seed)]
            fingerprints = {entry["initial_state_fingerprint"] for entry in entries}
            if fingerprints != {histories[(level, seed)]["initial_state_fingerprint"]}:
                raise ValueError("Evaluation curve initialization mismatch")
            progress = np.asarray([entry["optimizer_step"] / 20600 for entry in entries])
            rates = np.asarray([entry["success_rate"] for entry in entries])
            final = entries[-1]; by_id = {episode["task_id"]: episode for episode in final["episodes"]}
            if len(by_id) != 100 or not hard_ids <= set(by_id):
                raise ValueError("Development task IDs are incomplete")
            per_seed.append({
                "corruption_percentage": level, "ranking_seed": RANKING_SEED if level else "original",
                "training_seed": seed, "final_successes": final["successes"],
                "evaluation_tasks": final["tasks"], "final_success_rate": final["success_rate"],
                "learning_curve_auc": float(np.trapezoid(rates, progress)),
                "hard_success_rate": float(np.mean([bool(by_id[task_id]["success"]) for task_id in hard_ids])),
                "mean_final_goal_xy_error": final["mean_final_object_goal_xy_error"],
                "final_validation_loss": histories[(level, seed)]["records"][-1]["validation_total_loss"],
                "initial_state_fingerprint": final["initial_state_fingerprint"],
            })
    save_csv(RESULT_ROOT / "per_seed_results_v1.csv", per_seed)

    summary: list[dict[str, Any]] = []
    fields = ("final_success_rate", "learning_curve_auc", "hard_success_rate", "mean_final_goal_xy_error", "final_validation_loss")
    for level in LEVELS:
        rows = [row for row in per_seed if row["corruption_percentage"] == level]
        record: dict[str, Any] = {"corruption_percentage": level, "training_seeds": 3}
        for field in fields:
            values = np.asarray([float(row[field]) for row in rows])
            record[f"mean_{field}"] = float(values.mean()); record[f"std_{field}"] = float(values.std(ddof=1))
        summary.append(record)
    save_csv(RESULT_ROOT / "summary_table_v1.csv", summary)

    paired: list[dict[str, Any]] = []
    for level in CORRUPTION_LEVELS:
        for seed in TRAINING_SEEDS:
            candidate = curves[(level, seed)][-1]["episodes"]
            baseline = curves[(0, seed)][-1]["episodes"]
            comparison = paired_outcomes(candidate, baseline)
            paired.append({
                "corruption_percentage": level, "training_seed": seed,
                "reference": "original_reverse_learner_0_percent", **comparison,
            })
    save_csv(RESULT_ROOT / "paired_comparisons_v1.csv", paired)

    means = [row["mean_final_success_rate"] for row in summary]
    if all(a >= b for a, b in zip(means, means[1:])):
        pattern = "monotonic_nonincreasing"
    elif all(a <= b for a, b in zip(means, means[1:])):
        pattern = "monotonic_nondecreasing"
    else:
        pattern = "non_monotonic"
    manifest = read_json(ROOT / "data/metadata/ranking_robustness_v1/ranking_corruption_manifest_v1.json")
    payload = {
        "version": 1, "exploratory_follow_up": True,
        "dataset_fingerprint": DATASET_FINGERPRINT, "ranking_seed": RANKING_SEED,
        "zero_corruption_retrained": False, "new_training_runs": 9,
        "optimizer_updates_per_new_run": 20600,
        "corruption_manifest": manifest, "per_seed": per_seed, "summary": summary,
        "paired_comparisons": paired, "observed_mean_final_success_pattern": pattern,
        "uncertainty": "sample standard deviation across 3 training seeds (ddof=1); one corruption realization per level",
        "interpretation_limit": "Development-benchmark exploratory follow-up after M8; not independent final-test evidence.",
    }
    (RESULT_ROOT / "ranking_robustness_summary_v1.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    summary_plot(summary, "mean_final_success_rate", "std_final_success_rate", "Final success rate", FIGURE_ROOT / "01_final_success_vs_corruption.png")
    summary_plot(summary, "mean_learning_curve_auc", "std_learning_curve_auc", "Learning-curve AUC", FIGURE_ROOT / "02_auc_vs_corruption.png")
    summary_plot(summary, "mean_hard_success_rate", "std_hard_success_rate", "Hard-case success rate", FIGURE_ROOT / "03_hard_success_vs_corruption.png")
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    for level in LEVELS:
        matrix = np.asarray([[entry["success_rate"] for entry in curves[(level, seed)]] for seed in TRAINING_SEEDS])
        x = np.asarray([entry["optimizer_step"] / 20600 for entry in curves[(level, TRAINING_SEEDS[0])]])
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        axis.plot(x, mean, marker="o", label=f"{level}%"); axis.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), alpha=.10)
    for boundary in (1 / 3, 2 / 3): axis.axvline(boundary, color="gray", linestyle="--", linewidth=1)
    axis.set(xlabel="Normalized optimizer updates", ylabel="Success rate", ylim=(0, 1.03), title="Development-benchmark learning curves (mean ± sample SD)")
    axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", alpha=.25); axis.legend(title="Corruption", frameon=False)
    figure.tight_layout(); figure.savefig(FIGURE_ROOT / "04_learning_curves.png", dpi=220, bbox_inches="tight"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7.5, 4.7))
    for seed in TRAINING_SEEDS:
        values = [next(float(row["final_success_rate"]) for row in per_seed if row["corruption_percentage"] == level and row["training_seed"] == seed) for level in LEVELS]
        axis.plot(LEVELS, values, marker="o", label=str(seed))
    axis.set_ylim(0, 1.05); axis.set_title("Seed-level final-success variability"); axis.legend(title="Training seed", frameon=False)
    finish_plot(figure, axis, FIGURE_ROOT / "05_seed_level_variability.png", "Final success rate")
    print(f"Ranking robustness integrity check: PASSED; pattern={pattern}")
    for row in summary:
        print(f"{row['corruption_percentage']}%: final={row['mean_final_success_rate']:.3f}±{row['std_final_success_rate']:.3f}, "
              f"AUC={row['mean_learning_curve_auc']:.3f}±{row['std_learning_curve_auc']:.3f}, "
              f"hard={row['mean_hard_success_rate']:.3f}±{row['std_hard_success_rate']:.3f}")


if __name__ == "__main__":
    main()
