"""Fast integrity tests for Milestone 9 ranking robustness."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from evaluation.ranking_robustness import (
    CORRUPTION_LEVELS, RANKING_SEED, TRAINING_SEEDS, corrupt_ranking,
    ordered_learner_ids, sha256_file, validate_corrupted_payload,
)
from training.curriculum_sampler import eligible_trajectory_ids, stage_for_update
from training.train_curriculum import create_initial_state, state_dict_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def inputs() -> tuple[dict, dict, list[str]]:
    ranking = json.loads((ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json").read_text())
    split = json.loads((ROOT / "data/metadata/bc_split_v1.json").read_text())
    return ranking, split, ordered_learner_ids(ranking)


def test_corruptions_are_reproducible_complete_and_frozen() -> None:
    _, _, original_ids = inputs()
    manifest = json.loads((ROOT / "data/metadata/ranking_robustness_v1/ranking_corruption_manifest_v1.json").read_text())
    assert manifest["ranking_seed"] == RANKING_SEED
    for level, record in zip(CORRUPTION_LEVELS, manifest["rankings"]):
        path = ROOT / record["path"]
        payload = json.loads(path.read_text())
        validate_corrupted_payload(payload, original_ids, percentage=level)
        regenerated = corrupt_ranking(original_ids, percentage=level)
        for key, value in regenerated.items():
            assert payload[key] == value
        assert sha256_file(path) == record["sha256"]


def test_corruptions_keep_exact_group_sizes_and_training_ids() -> None:
    _, split, original_ids = inputs()
    assert set(original_ids) == set(split["train_trajectory_ids"])
    assert not set(original_ids) & set(split["validation_trajectory_ids"])
    for level in CORRUPTION_LEVELS:
        payload = json.loads((ROOT / f"data/metadata/ranking_robustness_v1/learner_ranking_corruption_{level:03d}.json").read_text())
        flattened = [value for group in ("easy", "medium", "hard") for value in payload["groups"][group]]
        assert len(flattened) == len(set(flattened)) == 240
        assert set(flattened) == set(split["train_trajectory_ids"])
        assert all(len(payload["groups"][group]) == 80 for group in ("easy", "medium", "hard"))


def test_reverse_schedule_uses_hard_then_hard_medium_then_all() -> None:
    _, split, original_ids = inputs()
    payload = json.loads((ROOT / "data/metadata/ranking_robustness_v1/learner_ranking_corruption_050.json").read_text())
    rankings = {"learner": payload["groups"]}
    stage_one = set(eligible_trajectory_ids("reverse_learner", 1, original_ids, rankings))
    stage_two = set(eligible_trajectory_ids("reverse_learner", 2, original_ids, rankings))
    stage_three = set(eligible_trajectory_ids("reverse_learner", 3, original_ids, rankings))
    assert stage_one == set(payload["groups"]["hard"])
    assert stage_two == set(payload["groups"]["hard"] + payload["groups"]["medium"])
    assert stage_three == set(split["train_trajectory_ids"])
    assert stage_for_update(6866, 20600, [1 / 3, 2 / 3]) == 1
    assert stage_for_update(6867, 20600, [1 / 3, 2 / 3]) == 2
    assert stage_for_update(13733, 20600, [1 / 3, 2 / 3]) == 2
    assert stage_for_update(13734, 20600, [1 / 3, 2 / 3]) == 3


def test_update_budget_matches_m7_protocol() -> None:
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["curriculum"]
    assert config["total_optimizer_updates"] == 20600
    assert config["batch_size"] == 256
    assert config["learning_rate"] == 0.001
    assert config["evaluation_checkpoint_fractions"] == [0.10, 0.25, 0.50, 0.75, 1.00]


def test_exact_m7_initializations_are_reproducible() -> None:
    hidden_dims = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["bc"]["hidden_dims"]
    for seed in TRAINING_SEEDS:
        payload = torch.load(ROOT / f"outputs/checkpoints/curriculum_v1/initial_seed_{seed}.pt", map_location="cpu", weights_only=False)
        generated = state_dict_fingerprint(create_initial_state(seed, hidden_dims))
        assert payload["fingerprint"] == generated
        assert state_dict_fingerprint(payload["model_state_dict"]) == generated


def test_original_ranking_is_unchanged() -> None:
    path = ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json"
    assert sha256_file(path) == "5c5a70465cbf6b5b51578661cefe21d8baad1bbf3114f49c61fb3a504e4505af"
