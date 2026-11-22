"""Shared standardization, ranking, and correlation helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def fit_standardizer(
    values: np.ndarray, component_names: Sequence[str], epsilon: float
) -> dict[str, Any]:
    """Fit per-component population statistics on training trajectories."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(component_names):
        raise ValueError("values and component_names have incompatible shapes")
    if not np.all(np.isfinite(array)):
        raise ValueError("standardization values must be finite")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return {
        "component_names": list(component_names),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "epsilon": float(epsilon),
    }


def standardize(values: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.asarray(parameters["mean"], dtype=np.float64)
    std = np.asarray(parameters["std"], dtype=np.float64)
    epsilon = float(parameters["epsilon"])
    result = (array - mean) / (std + epsilon)
    if not np.all(np.isfinite(result)):
        raise ValueError("standardized values must be finite")
    return result


def equal_weight_composite(standardized_components: np.ndarray) -> np.ndarray:
    array = np.asarray(standardized_components, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("standardized_components must be a non-empty matrix")
    return array.mean(axis=1)


def deterministic_ranking(
    score_by_id: Mapping[str, float]
) -> dict[str, Any]:
    """Sort low-to-high with trajectory ID tie-breaking and create thirds."""
    if len(score_by_id) == 0 or len(score_by_id) % 3 != 0:
        raise ValueError("trajectory count must be positive and divisible by three")
    if not all(np.isfinite(score) for score in score_by_id.values()):
        raise ValueError("ranking scores must be finite")
    ordered = sorted(score_by_id, key=lambda key: (score_by_id[key], key))
    group_size = len(ordered) // 3
    return {
        "ordered_trajectory_ids": ordered,
        "groups": {
            "easy": ordered[:group_size],
            "medium": ordered[group_size : 2 * group_size],
            "hard": ordered[2 * group_size :],
        },
    }


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_ranks = _average_ranks(np.asarray(x, dtype=np.float64))
    y_ranks = _average_ranks(np.asarray(y, dtype=np.float64))
    if np.std(x_ranks) == 0.0 or np.std(y_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks
