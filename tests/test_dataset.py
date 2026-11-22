"""Tests for randomized tasks and demonstration serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.dataset import DatasetValidationError, inspect_dataset
from data.task_sampler import TaskSampler
from data.trajectory import (
    fingerprint_entry,
    load_trajectory,
    rollout_expert,
    save_trajectory,
    task_fingerprint,
    validate_trajectory,
)
from envs.panda_pick_place import PandaPickPlaceEnv
from expert.scripted_expert import ScriptedExpert


def test_task_sampler_is_reproducible() -> None:
    first = TaskSampler(123)
    second = TaskSampler(123)
    for _ in range(10):
        task_a = first.sample()
        task_b = second.sample()
        assert task_a.task_id == task_b.task_id
        assert task_a.seed == task_b.seed
        np.testing.assert_array_equal(task_a.object_position, task_b.object_position)
        np.testing.assert_array_equal(task_a.goal_position, task_b.goal_position)
        assert task_a.target_radius == task_b.target_radius


def test_sampled_tasks_respect_bounds_and_minimum_distance() -> None:
    sampler = TaskSampler(5478)
    config = sampler.config
    for _ in range(100):
        task = sampler.sample()
        assert config["object_x_range"][0] <= task.object_position[0] <= config["object_x_range"][1]
        assert config["object_y_range"][0] <= task.object_position[1] <= config["object_y_range"][1]
        assert config["goal_x_range"][0] <= task.goal_position[0] <= config["goal_x_range"][1]
        assert config["goal_y_range"][0] <= task.goal_position[1] <= config["goal_y_range"][1]
        assert np.linalg.norm(task.object_position[:2] - task.goal_position[:2]) >= config["min_object_goal_distance"]


def test_reset_accepts_explicit_deterministic_task() -> None:
    sampler = TaskSampler(9)
    task = sampler.sample()
    with PandaPickPlaceEnv(mode="direct") as env:
        first = env.reset(
            task.object_position, task.goal_position, task.target_radius, task.seed
        )
        second = env.reset(
            task.object_position, task.goal_position, task.target_radius, task.seed
        )
        np.testing.assert_allclose(second, first, atol=1e-6)
        np.testing.assert_allclose(env.get_goal_position(), task.goal_position)
        assert env.target_radius == task.target_radius


def test_expert_solves_representative_randomized_tasks() -> None:
    sampler = TaskSampler(321)
    with PandaPickPlaceEnv(mode="direct") as env:
        expert = ScriptedExpert()
        for index in range(3):
            trajectory = rollout_expert(
                env, expert, sampler.sample(), f"representative_{index}"
            )
            assert trajectory["success"], trajectory["failure_reason"]


def test_trajectory_save_load_round_trip(tmp_path: Path) -> None:
    sampler = TaskSampler(44)
    with PandaPickPlaceEnv(mode="direct") as env:
        trajectory = rollout_expert(
            env, ScriptedExpert(), sampler.sample(), "traj_0000"
        )
    path = tmp_path / "traj_0000.npz"
    save_trajectory(path, trajectory)
    loaded = load_trajectory(path)
    validate_trajectory(loaded, require_success=True)
    np.testing.assert_array_equal(loaded["observations"], trajectory["observations"])
    np.testing.assert_array_equal(loaded["actions"], trajectory["actions"])
    assert loaded["observations"].shape[1] == 10
    assert loaded["actions"].shape[1] == 4


def test_dataset_inspection_detects_malformed_data(tmp_path: Path) -> None:
    demo_dir = tmp_path / "demonstrations"
    demo_dir.mkdir()
    malformed = {
        "trajectory_id": "traj_0000",
        "observations": np.zeros((2, 9), dtype=np.float32),
    }
    np.savez_compressed(demo_dir / "traj_0000.npz", **malformed)
    metadata = {
        "number_of_trajectories": 1,
        "number_of_total_state_action_pairs": 2,
        "task_randomization": TaskSampler(1).config,
        "dataset_fingerprint": task_fingerprint([]),
    }
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Malformed"):
        inspect_dataset(demo_dir, metadata_path)
