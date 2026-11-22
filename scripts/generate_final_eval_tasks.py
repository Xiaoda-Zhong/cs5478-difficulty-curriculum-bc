#!/usr/bin/env python3
"""Generate and freeze M8 ID, precision-OOD, and feasible spatial-OOD tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.trajectory import rollout_expert  # noqa: E402
from difficulty.normalization import equal_weight_composite, standardize  # noqa: E402
from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from evaluation.final_benchmark import (  # noqa: E402
    DATASET_FINGERPRINT, generate_unique_tasks, sha256_file, task_key,
    tasks_from_payload, validate_task_payload,
)
from expert.scripted_expert import ScriptedExpert  # noqa: E402

ID_PATH = REPOSITORY_ROOT / "data/metadata/final_id_eval_tasks_v1.json"
PRECISION_PATH = REPOSITORY_ROOT / "data/metadata/final_precision_ood_v1.json"
SPATIAL_PATH = REPOSITORY_ROOT / "data/metadata/final_spatial_ood_v1.json"
GROUP_PATH = REPOSITORY_ROOT / "data/metadata/final_id_difficulty_groups_v1.json"
MANIFEST_PATH = REPOSITORY_ROOT / "data/metadata/final_benchmark_manifest_v1.json"
PILOT_PATH = REPOSITORY_ROOT / "outputs/results/final_evaluation_v1/spatial_feasibility_pilot.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-pilot", action="store_true")
    parser.add_argument("--include-spatial", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/default.yaml").read_text())
    base = dict(config["task_randomization"])
    demo_metadata = json.loads((REPOSITORY_ROOT / "data/metadata/dataset_metadata.json").read_text())
    dev_payload = json.loads((REPOSITORY_ROOT / "data/metadata/id_eval_tasks_v1.json").read_text())
    excluded = {task_key(summary) for summary in demo_metadata["trajectory_summaries"]}
    excluded |= {task_key(task) for task in dev_payload["tasks"]}
    if args.spatial_pilot:
        _run_spatial_pilot(base, excluded)
        return
    id_payload = _load_or_create(
        ID_PATH, seed=25478, count=200, prefix="final_id", config=None,
        distribution="fresh ID: original M3 task distribution", excluded=excluded,
    )
    id_keys = {task_key(task) for task in id_payload["tasks"]}
    precision_config = {**base, "target_radius_range": [0.030, 0.044]}
    precision_payload = _load_or_create(
        PRECISION_PATH, seed=35478, count=100, prefix="precision_ood",
        config=precision_config,
        distribution="precision OOD: original XY ranges; target radius [0.030, 0.044] m",
        excluded=excluded | id_keys,
    )
    _load_or_create_groups(id_payload)
    paths = [ID_PATH, PRECISION_PATH, GROUP_PATH]
    if args.include_spatial:
        if not PILOT_PATH.exists() or json.loads(PILOT_PATH.read_text())["success_rate"] < 0.95:
            raise SystemExit("A >=95% expert spatial feasibility pilot is required before freezing spatial OOD")
        spatial_config = _spatial_config(base)
        precision_keys = {task_key(task) for task in precision_payload["tasks"]}
        spatial_payload = _load_or_create(
            SPATIAL_PATH, seed=45478, count=100, prefix="spatial_ood",
            config=spatial_config,
            distribution="spatial OOD: object x [0.390, 0.415] m; all other ranges unchanged",
            excluded=excluded | id_keys | precision_keys,
        )
        if not all(float(task["object_position"][0]) < 0.42 for task in spatial_payload["tasks"]):
            raise ValueError("Every spatial-OOD task must leave the original object-x range")
        paths.append(SPATIAL_PATH)
    manifest = {
        "version": 1, "date_frozen_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "configuration_sha256": {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in paths},
        "spatial_ood_included": SPATIAL_PATH in paths,
        "policy_results_observed_before_freeze": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_or_create(path: Path, *, seed: int, count: int, prefix: str, config, distribution: str, excluded: set[str]) -> dict:
    if path.exists():
        payload = json.loads(path.read_text())
        validate_task_payload(payload, expected_count=count, expected_seed=seed, excluded_keys=excluded)
        return payload
    tasks = generate_unique_tasks(seed=seed, count=count, prefix=prefix, config=config, excluded_keys=excluded)
    payload = {
        "version": 1, "dataset_fingerprint": DATASET_FINGERPRINT,
        "seed": seed, "count": count, "distribution": distribution,
        "generation_config": config or "configs/default.yaml task_randomization",
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_task_payload(payload, expected_count=count, expected_seed=seed, excluded_keys=excluded)
    return payload


def _load_or_create_groups(id_payload: dict) -> dict:
    if GROUP_PATH.exists():
        payload = json.loads(GROUP_PATH.read_text())
        if payload["source_task_sha256"] != sha256_file(ID_PATH):
            raise ValueError("Final ID groups do not match the frozen task file")
        return payload
    difficulty = json.loads((REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_metadata_v1.json").read_text())
    normalization = difficulty["normalization_parameters"]["task"]
    records = []
    with PandaPickPlaceEnv(mode="direct") as env:
        for task in tasks_from_payload(id_payload):
            observation = env.reset(task.object_position, task.goal_position, task.target_radius, task.seed)
            components = np.asarray([[
                np.linalg.norm(observation[:3] - observation[3:6]),
                np.linalg.norm(observation[3:5] - observation[6:8]),
                1.0 / task.target_radius,
            ]])
            score = float(equal_weight_composite(standardize(components, normalization))[0])
            records.append({"task_id": task.task_id, "task_difficulty": score})
    ordered = sorted(records, key=lambda row: (row["task_difficulty"], row["task_id"]))
    payload = {
        "version": 1, "dataset_fingerprint": DATASET_FINGERPRINT,
        "source_task_file": str(ID_PATH.relative_to(REPOSITORY_ROOT)),
        "source_task_sha256": sha256_file(ID_PATH),
        "definition": "M5 task difficulty with frozen M5 normalization; no rollout outcomes",
        "remainder_handling": "200 tasks split 67 Easy, 67 Medium, 66 Hard; remainder assigned from easiest upward",
        "records": records,
        "groups": {
            "easy": [row["task_id"] for row in ordered[:67]],
            "medium": [row["task_id"] for row in ordered[67:134]],
            "hard": [row["task_id"] for row in ordered[134:]],
        },
    }
    GROUP_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _spatial_config(base: dict) -> dict:
    return {**base, "object_x_range": [0.390, 0.415]}


def _run_spatial_pilot(base: dict, excluded: set[str]) -> None:
    tasks = generate_unique_tasks(
        seed=45477, count=20, prefix="spatial_pilot", config=_spatial_config(base), excluded_keys=excluded
    )
    records = []; failures = Counter()
    with PandaPickPlaceEnv(mode="direct") as env:
        expert = ScriptedExpert()
        for task in tasks_from_payload({"tasks": tasks}):
            trajectory = rollout_expert(env, expert, task, f"pilot_{task.task_id}")
            success = bool(trajectory["success"])
            reason = str(trajectory["failure_reason"] or "")
            if not success: failures[reason] += 1
            records.append({"task_id": task.task_id, "success": success, "steps": int(trajectory["trajectory_length"]), "failure_reason": reason or None})
    successes = sum(row["success"] for row in records)
    report = {
        "purpose": "expert-only spatial-OOD feasibility pilot; configurations excluded from final benchmark",
        "seed": 45477, "count": len(tasks), "successes": successes,
        "success_rate": successes / len(tasks), "failure_reasons": dict(failures),
        "candidate_distribution": _spatial_config(base), "records": records,
    }
    PILOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PILOT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("count", "successes", "success_rate", "failure_reasons")}, indent=2))


if __name__ == "__main__":
    main()
