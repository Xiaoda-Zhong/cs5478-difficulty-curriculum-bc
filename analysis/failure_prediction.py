"""Dependency-light binary failure-prediction metrics and uncertainty."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def failure_labels(success: Sequence[bool | int]) -> np.ndarray:
    """Return 1 for failed rollouts and 0 for successful rollouts."""
    success_array = np.asarray(success, dtype=bool)
    return (~success_array).astype(np.int64)


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute AUROC with average ranks for ties; higher predicts failure."""
    y, values = _validated_arrays(labels, scores)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(values)
    positive_rank_sum = float(ranks[y == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute tie-invariant average precision for the failure class."""
    y, values = _validated_arrays(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    sorted_labels = y[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        true_positives += int(group.sum())
        false_positives += len(group) - int(group.sum())
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(average_precision)


def bootstrap_auroc(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Return a stratified percentile CI using a deterministic seed."""
    y, values = _validated_arrays(labels, scores)
    estimate = auroc(y, values)
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    if estimate is None or samples <= 0:
        return {
            "estimate": estimate,
            "ci_lower": None,
            "ci_upper": None,
            "valid_samples": 0,
        }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = np.concatenate(
            [
                rng.choice(positive_indices, len(positive_indices), replace=True),
                rng.choice(negative_indices, len(negative_indices), replace=True),
            ]
        )
        estimates[index] = float(auroc(y[sampled], values[sampled]))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "valid_samples": samples,
    }


def paired_bootstrap_auroc_differences(
    labels: Sequence[int],
    score_map: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    """Use shared stratified resamples for pairwise AUROC differences."""
    y = np.asarray(labels, dtype=np.int64)
    arrays = {name: np.asarray(values, dtype=float) for name, values in score_map.items()}
    if any(len(values) != len(y) for values in arrays.values()):
        raise ValueError("All score arrays must match labels")
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    names = list(arrays)
    result: dict[str, dict[str, float | int | None]] = {}
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return result
    rng = np.random.default_rng(seed)
    differences = {
        f"{names[left]}_minus_{names[right]}": []
        for left in range(len(names))
        for right in range(left + 1, len(names))
    }
    for _ in range(samples):
        sampled = np.concatenate(
            [
                rng.choice(positive_indices, len(positive_indices), replace=True),
                rng.choice(negative_indices, len(negative_indices), replace=True),
            ]
        )
        estimates = {
            name: float(auroc(y[sampled], values[sampled]))
            for name, values in arrays.items()
        }
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                key = f"{names[left]}_minus_{names[right]}"
                differences[key].append(estimates[names[left]] - estimates[names[right]])
    for key, values in differences.items():
        lower, upper = np.percentile(values, [2.5, 97.5])
        left_name, right_name = key.split("_minus_")
        point = float(auroc(y, arrays[left_name])) - float(auroc(y, arrays[right_name]))
        result[key] = {
            "estimate": point,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "valid_samples": samples,
        }
    return result


def _validated_arrays(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or values.shape != y.shape or len(y) == 0:
        raise ValueError("labels and scores must be non-empty matching vectors")
    if not np.all(np.isin(y, [0, 1])) or not np.all(np.isfinite(values)):
        raise ValueError("labels must be binary and scores finite")
    return y, values


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return ranks
