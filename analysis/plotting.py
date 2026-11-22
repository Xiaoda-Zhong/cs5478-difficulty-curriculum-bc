"""Static figures for Milestone 6 difficulty/failure analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def correlation_heatmap(matrix: np.ndarray, labels: Sequence[str], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    ax.set_title("Spearman correlations")
    fig.colorbar(image, ax=ax, shrink=0.8)
    _save(fig, path)


def pairwise_difficulty_scatter(rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    keys = ["task_difficulty", "expert_difficulty", "learner_difficulty"]
    labels = ["Task", "Expert", "Learner"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (left, right) in zip(axes, pairs):
        success = np.asarray([bool(row["success"]) for row in rows])
        x = np.asarray([float(row[keys[left]]) for row in rows])
        y = np.asarray([float(row[keys[right]]) for row in rows])
        ax.scatter(x[success], y[success], s=15, alpha=0.65, label="success")
        ax.scatter(x[~success], y[~success], s=15, alpha=0.65, label="failure")
        ax.set(xlabel=labels[left], ylabel=labels[right])
    axes[0].legend(frameon=False)
    fig.suptitle("Difficulty scores and OOF rollout outcomes")
    _save(fig, path)


def group_failure_bars(group_rates: Mapping[str, Mapping[str, Mapping[str, float]]], path: str | Path) -> None:
    groups = ["easy", "medium", "hard"]
    metrics = ["task", "expert", "learner"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(len(groups))
    width = 0.24
    largest = 0.0
    for index, metric in enumerate(metrics):
        values = [100 * float(group_rates[metric][group]["failure_rate"]) for group in groups]
        largest = max(largest, max(values))
        bars = ax.bar(positions + (index - 1) * width, values, width, label=metric.title())
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(positions, [group.title() for group in groups])
    ax.set(ylabel="Failure rate (%)", ylim=(0, largest * 1.25))
    ax.legend(frameon=False)
    ax.set_title("OOF failure rate by predefined difficulty group")
    _save(fig, path)


def auroc_comparison(metrics: Mapping[str, Mapping[str, float | int | None]], path: str | Path) -> None:
    names = list(metrics)
    estimates = np.asarray([float(metrics[name]["estimate"]) for name in names])
    lower = np.asarray([float(metrics[name]["ci_lower"]) for name in names])
    upper = np.asarray([float(metrics[name]["ci_upper"]) for name in names])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(names))
    ax.errorbar(positions, estimates, yerr=[estimates - lower, upper - estimates], fmt="o", capsize=5)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(positions, names, rotation=25, ha="right")
    ax.set(ylabel="AUROC (failure is positive)", ylim=(0, 1))
    ax.set_title("Failure-prediction discrimination (95% bootstrap CI)")
    _save(fig, path)


def learner_vs_duration(rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    success = np.asarray([bool(row["success"]) for row in rows])
    duration = np.asarray([float(row["expert_duration"]) for row in rows])
    learner = np.asarray([float(row["learner_difficulty"]) for row in rows])
    ax.scatter(duration[success], learner[success], s=18, alpha=0.7, label="success")
    ax.scatter(duration[~success], learner[~success], s=18, alpha=0.7, label="failure")
    ax.set(xlabel="Expert demonstration duration (steps)", ylabel="OOF learner motion error")
    ax.legend(frameon=False)
    ax.set_title("Learner error versus demonstration duration")
    _save(fig, path)


def phase_error_bars(phase_rows: Sequence[Mapping[str, object]], path: str | Path) -> None:
    ordered = sorted(phase_rows, key=lambda row: float(row["mean_motion_error"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [str(row["phase"]) for row in ordered]
    values = [float(row["mean_motion_error"]) for row in ordered]
    bars = ax.bar(names, values)
    ax.set(ylabel="Mean normalized squared motion error")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("OOF learner error by expert phase")
    for bar, row in zip(bars, ordered):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{100*float(row['error_contribution']):.0f}%", ha="center", va="bottom", fontsize=8)
    _save(fig, path)
