"""Unit tests for Milestone 5 trajectory difficulty metrics."""

from __future__ import annotations

import numpy as np

from difficulty.expert import expert_components
from difficulty.learner import generate_oof_folds, learner_errors, validate_oof_folds
from difficulty.normalization import (
    deterministic_ranking,
    equal_weight_composite,
    fit_standardizer,
    standardize,
)
from difficulty.task import task_components


def test_task_distance_and_precision_components() -> None:
    trajectory = {
        "initial_ee_position": np.array([0.0, 0.0, 1.0]),
        "initial_object_position": np.array([0.0, 0.0, 0.0]),
        "goal_position": np.array([3.0, 4.0, 0.0]),
        "target_radius": 0.05,
    }
    components = task_components(trajectory)
    assert components["task_ee_object_distance"] == 1.0
    assert components["task_object_goal_distance"] == 5.0
    assert components["task_inverse_target_radius"] == 20.0
    trajectory["target_radius"] = 0.025
    assert (
        task_components(trajectory)["task_inverse_target_radius"]
        > components["task_inverse_target_radius"]
    )


def test_standardized_composite_is_finite() -> None:
    values = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 6.0]])
    parameters = fit_standardizer(values, ["a", "b", "c"], 1e-8)
    score = equal_weight_composite(standardize(values, parameters))
    assert score.shape == (2,)
    assert np.all(np.isfinite(score))


def test_expert_components_use_executed_path_and_cartesian_actions() -> None:
    trajectory = {
        "trajectory_length": 2,
        "actions": np.array([[3.0, 4.0, 0.0, 1.0], [0.0, 0.0, 5.0, -1.0]]),
        "ee_positions": np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]]),
    }
    components = expert_components(trajectory)
    assert components["expert_duration"] == 2.0
    assert components["expert_ee_path_length"] == 17.0
    assert components["expert_action_magnitude"] == 10.0


def test_learner_motion_and_gripper_error() -> None:
    expert_actions = np.array(
        [[0.008, 0.0, 0.0, 1.0], [0.0, 0.008, 0.0, -1.0]]
    )
    predicted_motion = np.array([[0.0, 0.0, 0.0], [0.0, 0.008, 0.0]])
    logits = np.array([0.0, 0.0])
    errors = learner_errors(
        predicted_motion, logits, expert_actions, motion_scale=0.008
    )
    assert errors["learner_motion_error"] == 0.5
    np.testing.assert_allclose(errors["learner_gripper_error"], np.log(2.0))


def test_oof_fold_assignment_is_deterministic_and_disjoint() -> None:
    trajectory_ids = [f"traj_{index:04d}" for index in range(20)]
    first = generate_oof_folds(trajectory_ids, 5, 5478)
    second = generate_oof_folds(trajectory_ids, 5, 5478)
    assert first == second
    validate_oof_folds(first, trajectory_ids)
    held_out = []
    for fold in first:
        assert len(fold["held_out_trajectory_ids"]) == 4
        assert len(fold["training_trajectory_ids"]) == 16
        assert not set(fold["held_out_trajectory_ids"]) & set(
            fold["training_trajectory_ids"]
        )
        held_out.extend(fold["held_out_trajectory_ids"])
    assert sorted(held_out) == trajectory_ids


def test_ranking_is_stable_and_groups_are_equal() -> None:
    scores = {
        "traj_b": 0.0,
        "traj_a": 0.0,
        "traj_c": 1.0,
        "traj_d": 2.0,
        "traj_e": 3.0,
        "traj_f": 4.0,
    }
    ranking = deterministic_ranking(scores)
    assert ranking["ordered_trajectory_ids"][:2] == ["traj_a", "traj_b"]
    groups = ranking["groups"]
    assert [len(groups[name]) for name in ("easy", "medium", "hard")] == [2, 2, 2]
    assert set().union(*map(set, groups.values())) == set(scores)
