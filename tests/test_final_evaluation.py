"""Fast tests for frozen final tasks, checkpoint mapping, and paired analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.final_benchmark import (
    DATASET_FINGERPRINT, METHODS, SEEDS, final_checkpoint_path,
    generate_unique_tasks, paired_outcomes, paired_task_bootstrap,
    remaining_task_ids, success_rate, task_key, validate_final_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def test_final_id_generation_is_reproducible_and_unique() -> None:
    first = generate_unique_tasks(seed=25478, count=20, prefix="x", config=None, excluded_keys=set())
    second = generate_unique_tasks(seed=25478, count=20, prefix="x", config=None, excluded_keys=set())
    assert first == second
    assert len({task_key(task) for task in first}) == 20


def test_final_tasks_do_not_overlap_demo_or_development() -> None:
    demos = json.loads((ROOT / "data/metadata/dataset_metadata.json").read_text())["trajectory_summaries"]
    development = json.loads((ROOT / "data/metadata/id_eval_tasks_v1.json").read_text())["tasks"]
    excluded = {task_key(task) for task in [*demos, *development]}
    frozen = json.loads((ROOT / "data/metadata/final_id_eval_tasks_v1.json").read_text())["tasks"]
    regenerated = generate_unique_tasks(
        seed=25478, count=200, prefix="final_id", config=None, excluded_keys=excluded,
    )
    assert frozen == regenerated
    assert not {task_key(task) for task in frozen} & excluded


def test_final_checkpoint_mapping_and_loading() -> None:
    for method in METHODS:
        for seed in SEEDS:
            path = final_checkpoint_path(ROOT, method, seed)
            assert path.name == "final.pt" and path.exists()
            checkpoint = validate_final_checkpoint(path, method, seed)
            assert checkpoint["optimizer_step"] == 20600


def test_precision_and_spatial_bounds_when_frozen() -> None:
    precision = json.loads((ROOT / "data/metadata/final_precision_ood_v1.json").read_text())
    assert len(precision["tasks"]) == 100
    assert all(0.030 <= task["target_radius"] <= 0.044 for task in precision["tasks"])
    spatial_path = ROOT / "data/metadata/final_spatial_ood_v1.json"
    if spatial_path.exists():
        spatial = json.loads(spatial_path.read_text())
        assert all(0.390 <= task["object_position"][0] <= 0.415 for task in spatial["tasks"])
        assert all(task["object_position"][0] < 0.42 for task in spatial["tasks"])
        assert all(-0.28 <= task["object_position"][1] <= 0.12 for task in spatial["tasks"])
        assert all(0.42 <= task["goal_position"][0] <= 0.63 for task in spatial["tasks"])
        assert all(-0.12 <= task["goal_position"][1] <= 0.28 for task in spatial["tasks"])
        assert all(
            ((task["object_position"][0] - task["goal_position"][0]) ** 2
             + (task["object_position"][1] - task["goal_position"][1]) ** 2) ** 0.5 >= 0.18
            for task in spatial["tasks"]
        )


def test_frozen_hard_groups_are_reproducible_and_complete() -> None:
    payload = json.loads((ROOT / "data/metadata/final_id_difficulty_groups_v1.json").read_text())
    groups = payload["groups"]
    assert {name: len(ids) for name, ids in groups.items()} == {"easy": 67, "medium": 67, "hard": 66}
    ordered = sorted(payload["records"], key=lambda row: (row["task_difficulty"], row["task_id"]))
    assert groups["easy"] == [row["task_id"] for row in ordered[:67]]
    assert groups["hard"] == [row["task_id"] for row in ordered[134:]]


def test_success_rate_and_paired_comparison() -> None:
    uniform = [{"task_id": "a", "success": True}, {"task_id": "b", "success": False}, {"task_id": "c", "success": True}]
    method = [{"task_id": "c", "success": False}, {"task_id": "a", "success": True}, {"task_id": "b", "success": True}]
    assert success_rate(method) == pytest.approx(2 / 3)
    paired = paired_outcomes(method, uniform)
    assert paired == {"task_count": 3, "both_solve": 1, "only_method_solves": 1, "only_uniform_solves": 1, "neither_solves": 0, "success_rate_difference": 0.0}


def test_resume_selects_only_missing_identical_tasks() -> None:
    expected = ["a", "b", "c"]
    assert remaining_task_ids(expected, [{"task_id": "b"}]) == ["a", "c"]
    with pytest.raises(ValueError, match="unknown"):
        remaining_task_ids(expected, [{"task_id": "z"}])


def test_paired_bootstrap_is_fixed_seed_and_task_level() -> None:
    uniform = [{"task_id": str(i), "success": i % 2 == 0} for i in range(20)]
    method = [{"task_id": str(i), "success": i % 3 == 0} for i in reversed(range(20))]
    first = paired_task_bootstrap(method, uniform, samples=100, seed=75478)
    second = paired_task_bootstrap(method, uniform, samples=100, seed=75478)
    assert first == second
    assert first["samples"] == 100
    assert first["ci_lower"] <= first["estimate"] <= first["ci_upper"]


def test_completed_final_results_share_identical_task_ids() -> None:
    root = ROOT / "outputs/results/final_evaluation_v1/evaluations"
    if not root.exists():
        pytest.skip("Full final evaluation has not been run")
    for split in ("final_id", "precision_ood", "spatial_ood"):
        expected = None
        for method in METHODS:
            for seed in SEEDS:
                payload = json.loads((root / split / method / f"seed_{seed}" / "final.json").read_text())
                task_ids = [row["task_id"] for row in payload["episodes"]]
                assert len(task_ids) == len(set(task_ids))
                expected = task_ids if expected is None else expected
                assert task_ids == expected


def test_frozen_payload_fingerprint_and_task_ids() -> None:
    for name, count in (("final_id_eval_tasks_v1.json", 200), ("final_precision_ood_v1.json", 100)):
        payload = json.loads((ROOT / "data/metadata" / name).read_text())
        assert payload["dataset_fingerprint"] == DATASET_FINGERPRINT
        assert len(payload["tasks"]) == count
        assert len({task["task_id"] for task in payload["tasks"]}) == count
