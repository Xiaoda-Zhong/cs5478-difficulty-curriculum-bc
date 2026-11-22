"""Lightweight invariants for the controlled curriculum experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from evaluation.rollout import load_or_create_evaluation_tasks
from training.curriculum_sampler import (
    CurriculumTransitionSampler, eligible_trajectory_ids, stage_for_update,
)
from training.train_curriculum import (
    checkpoint_steps, create_initial_state, state_dict_fingerprint,
    validate_training_history,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def experiment_data() -> tuple[dict, dict, dict]:
    split = json.loads((ROOT / "data/metadata/bc_split_v1.json").read_text())
    payload = json.loads((ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json").read_text())
    rankings = {name: value["groups"] for name, value in payload["rankings"].items()}
    return split, payload, rankings


def test_all_predefined_groups_are_80_and_cover_training(experiment_data) -> None:
    split, _, rankings = experiment_data
    expected = set(split["train_trajectory_ids"])
    for groups in rankings.values():
        assert {name: len(ids) for name, ids in groups.items()} == {"easy": 80, "medium": 80, "hard": 80}
        assert set().union(*(set(ids) for ids in groups.values())) == expected


@pytest.mark.parametrize("method", ["task", "expert", "learner"])
def test_forward_curriculum_selection(method, experiment_data) -> None:
    split, _, rankings = experiment_data
    train_ids = split["train_trajectory_ids"]
    assert set(eligible_trajectory_ids(method, 1, train_ids, rankings)) == set(rankings[method]["easy"])
    assert set(eligible_trajectory_ids(method, 2, train_ids, rankings)) == set(rankings[method]["easy"] + rankings[method]["medium"])
    assert set(eligible_trajectory_ids(method, 3, train_ids, rankings)) == set(train_ids)


def test_uniform_and_reverse_selection(experiment_data) -> None:
    split, _, rankings = experiment_data
    train_ids = split["train_trajectory_ids"]
    for stage in (1, 2, 3):
        assert set(eligible_trajectory_ids("uniform", stage, train_ids, rankings)) == set(train_ids)
    assert set(eligible_trajectory_ids("reverse_learner", 1, train_ids, rankings)) == set(rankings["learner"]["hard"])
    assert set(eligible_trajectory_ids("reverse_learner", 2, train_ids, rankings)) == set(rankings["learner"]["hard"] + rankings["learner"]["medium"])


def test_stage_boundaries_and_budget() -> None:
    total = 20600
    boundaries = [1 / 3, 2 / 3]
    assert stage_for_update(6866, total, boundaries) == 1
    assert stage_for_update(6867, total, boundaries) == 2
    assert stage_for_update(13733, total, boundaries) == 2
    assert stage_for_update(13734, total, boundaries) == 3
    assert checkpoint_steps(total, [.1, .25, .5, .75, 1]) == [2060, 5150, 10300, 15450, 20600]
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    assert total == config["curriculum"]["full_dataset_equivalent_epochs"] * math.ceil(52491 / config["curriculum"]["batch_size"])


def test_sampler_is_deterministic_transition_uniform(experiment_data) -> None:
    split, _, rankings = experiment_data
    ids = np.repeat(np.asarray(split["train_trajectory_ids"]), 2)
    first = CurriculumTransitionSampler(ids, split["train_trajectory_ids"], rankings, "task", 9)
    second = CurriculumTransitionSampler(ids, split["train_trajectory_ids"], rankings, "task", 9)
    for stage in (1, 2, 3):
        left, right = first.sample(64, stage), second.sample(64, stage)
        np.testing.assert_array_equal(left.indices, right.indices)
        assert left.eligible_trajectory_count == stage * 80
        assert left.eligible_transition_count == stage * 160


def test_no_validation_leakage(experiment_data) -> None:
    split, _, rankings = experiment_data
    validation = set(split["validation_trajectory_ids"])
    for method in ("uniform", "task", "expert", "learner", "reverse_learner"):
        for stage in (1, 2, 3):
            assert not validation & set(eligible_trajectory_ids(method, stage, split["train_trajectory_ids"], rankings))


def test_matching_initial_weights_across_methods() -> None:
    state = create_initial_state(5478, [128, 128])
    fingerprints = [state_dict_fingerprint(state) for _ in range(5)]
    assert len(set(fingerprints)) == 1
    assert state_dict_fingerprint(create_initial_state(5478, [128, 128])) == fingerprints[0]
    assert state_dict_fingerprint(create_initial_state(5479, [128, 128])) != fingerprints[0]


def test_checkpoint_metadata_shape() -> None:
    checkpoint = torch.load(ROOT / "outputs/checkpoints/uniform_bc_scaled/best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["motion_targets_normalized"] is True
    assert checkpoint["architecture"] == {"observation_dim": 10, "hidden_dims": [128, 128]}


def test_curriculum_checkpoint_metadata_and_matched_initialization() -> None:
    root = ROOT / "outputs/checkpoints/curriculum_v1"
    for seed in (5478, 5479, 5480):
        fingerprints = set()
        for method in ("uniform", "task", "expert", "learner", "reverse_learner"):
            checkpoint = torch.load(root / method / f"seed_{seed}" / "final.pt", map_location="cpu", weights_only=False)
            assert checkpoint["method_name"] == method
            assert checkpoint["training_seed"] == seed
            assert checkpoint["optimizer_step"] == 20600
            assert checkpoint["motion_targets_normalized"] is True
            assert checkpoint["training_configuration"]["optimizer"] == "AdamW"
            assert checkpoint["ranking_reference"] == "outputs/results/difficulty/difficulty_rankings_v1.json"
            fingerprints.add(checkpoint["initial_state_fingerprint"])
        assert len(fingerprints) == 1


def test_training_history_validation() -> None:
    history = {
        "method": "uniform", "training_seed": 1, "total_optimizer_updates": 1,
        "batch_size": 2, "learning_rate": .001, "stage_boundaries": [1/3, 2/3],
        "evaluation_steps": [1], "initial_state_fingerprint": "abc",
        "records": [{
            "optimizer_step": 1, "curriculum_stage": 3,
            "eligible_trajectory_count": 240, "eligible_transition_count": 52491,
            "training_total_loss": 1.0, "training_motion_mse": .5,
            "training_gripper_loss": .5, "validation_total_loss": 1.0,
            "validation_motion_mse": .5, "validation_gripper_accuracy": .5,
            "sampled_group_composition": {},
        }],
    }
    validate_training_history(history)
    history["records"][0]["validation_total_loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_training_history(history)


def test_completed_training_histories_are_valid() -> None:
    root = ROOT / "outputs/results/curriculum_v1"
    for method in ("uniform", "task", "expert", "learner", "reverse_learner"):
        for seed in (5478, 5479, 5480):
            history = json.loads((root / method / f"seed_{seed}" / "training_history.json").read_text())
            validate_training_history(history)


def test_existing_evaluation_tasks_load_exactly() -> None:
    payload = json.loads((ROOT / "data/metadata/id_eval_tasks_v1.json").read_text())
    tasks = load_or_create_evaluation_tasks(
        ROOT / "data/metadata/id_eval_tasks_v1.json", seed=15478, count=100,
        dataset_fingerprint=payload["dataset_fingerprint"], demonstration_tasks=[],
    )
    assert len(tasks) == 100
    assert tasks[0].task_id == "id_eval_0000"
    assert tasks[-1].task_id == "id_eval_0099"
