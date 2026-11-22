"""Tests for held-out rollout integrity and failure-prediction analysis."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.difficulty_diagnostics import (
    assert_held_out,
    evaluation_task_from_trajectory,
    group_outcome_rates,
    join_by_trajectory_id,
    validate_checkpoint_for_fold,
)
from analysis.failure_prediction import auprc, auroc, bootstrap_auroc, failure_labels


def test_checkpoint_fold_match_and_leakage_checks() -> None:
    fold = {"training_trajectory_ids": ["a", "b"], "held_out_trajectory_ids": ["c"]}
    checkpoint = dict(fold)
    validate_checkpoint_for_fold(checkpoint, fold)
    assert_held_out(checkpoint, "c")
    with pytest.raises(ValueError, match="Leakage"):
        assert_held_out(checkpoint, "a")
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_for_fold(
            {"training_trajectory_ids": ["a", "c"], "held_out_trajectory_ids": ["b"]}, fold
        )


def test_evaluation_task_reconstructs_exact_configuration() -> None:
    trajectory = {
        "task_seed": np.asarray(123),
        "task_object_position": np.asarray([0.4, -0.1, 0.025]),
        "goal_position": np.asarray([0.6, 0.2, 0.025]),
        "target_radius": np.asarray(0.055),
    }
    first = evaluation_task_from_trajectory("traj_0000", trajectory)
    second = evaluation_task_from_trajectory("traj_0000", trajectory)
    assert first.seed == second.seed == 123
    np.testing.assert_array_equal(first.object_position, second.object_position)
    np.testing.assert_array_equal(first.goal_position, second.goal_position)
    assert first.target_radius == second.target_radius == 0.055


def test_failure_labels_and_known_auc_values() -> None:
    labels = failure_labels([True, False, True, False])
    np.testing.assert_array_equal(labels, [0, 1, 0, 1])
    assert auroc(labels, [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert auroc(labels, [0.9, 0.1, 0.8, 0.2]) == 0.0
    assert auroc(labels, [1.0, 1.0, 1.0, 1.0]) == 0.5
    assert auprc(labels, [0.1, 0.9, 0.2, 0.8]) == 1.0


def test_bootstrap_is_deterministic_and_bounded() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.5, 0.4, 0.8, 0.9]
    first = bootstrap_auroc(labels, scores, samples=100, seed=7)
    second = bootstrap_auroc(labels, scores, samples=100, seed=7)
    assert first == second
    assert first["valid_samples"] == 100
    assert 0.0 <= first["ci_lower"] <= first["estimate"] <= first["ci_upper"] <= 1.0


def test_join_and_group_outcomes_are_id_stable() -> None:
    left = [{"trajectory_id": "b", "score": 2}, {"trajectory_id": "a", "score": 1}, {"trajectory_id": "c", "score": 3}]
    right = [{"trajectory_id": "c", "success": False}, {"trajectory_id": "b", "success": True}, {"trajectory_id": "a", "success": True}]
    joined = join_by_trajectory_id(left, right)
    assert [row["trajectory_id"] for row in joined] == ["a", "b", "c"]
    rates = group_outcome_rates(joined, {"easy": ["a"], "medium": ["b"], "hard": ["c"]})
    assert rates["easy"]["failure_rate"] == 0.0
    assert rates["hard"]["failure_rate"] == 1.0


def test_join_rejects_missing_ids() -> None:
    with pytest.raises(ValueError, match="sets do not match"):
        join_by_trajectory_id([{"trajectory_id": "a"}], [{"trajectory_id": "b"}])


def test_groups_reject_duplicate_assignment() -> None:
    rows = [{"trajectory_id": name, "success": True} for name in ("a", "b", "c")]
    with pytest.raises(ValueError, match="do not cover"):
        group_outcome_rates(rows, {"easy": ["a"], "medium": ["b"], "hard": ["a", "c"]})
