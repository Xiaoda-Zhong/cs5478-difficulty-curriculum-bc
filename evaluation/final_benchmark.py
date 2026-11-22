"""Frozen-task and paired-analysis utilities for the final benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data.task_sampler import TaskConfig, TaskSampler

DATASET_FINGERPRINT = "1b6cec9dda067778630268c307241c62cccbd49cfe849a031877ee00ea938bc0"
METHODS = ("uniform", "task", "expert", "learner", "reverse_learner")
SEEDS = (5478, 5479, 5480)


def task_key(task: Mapping[str, Any] | TaskConfig) -> str:
    if isinstance(task, TaskConfig):
        obj, goal, radius = task.object_position, task.goal_position, task.target_radius
    else:
        obj, goal, radius = task["object_position"], task["goal_position"], task["target_radius"]
    return json.dumps([list(obj), list(goal), float(radius)], separators=(",", ":"))


def generate_unique_tasks(
    *, seed: int, count: int, prefix: str, config: Mapping[str, Any] | None,
    excluded_keys: set[str],
) -> list[dict[str, Any]]:
    sampler = TaskSampler(seed, config=config) if config is not None else TaskSampler(seed)
    tasks: list[dict[str, Any]] = []
    seen = set(excluded_keys)
    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("Could not generate enough unique non-overlapping tasks")
        task = sampler.sample(task_id=f"{prefix}_{len(tasks):04d}").as_dict()
        key = task_key(task)
        if key in seen:
            continue
        seen.add(key); tasks.append(task)
    return tasks


def validate_task_payload(
    payload: Mapping[str, Any], *, expected_count: int, expected_seed: int,
    excluded_keys: set[str] | None = None,
) -> None:
    if payload["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise ValueError("Final task-set dataset fingerprint mismatch")
    if int(payload["count"]) != expected_count or int(payload["seed"]) != expected_seed:
        raise ValueError("Final task-set count or seed mismatch")
    tasks = payload["tasks"]
    keys = [task_key(task) for task in tasks]
    if len(tasks) != expected_count or len(set(keys)) != expected_count:
        raise ValueError("Final task configurations must be exactly unique")
    if excluded_keys and set(keys) & excluded_keys:
        raise ValueError("Final task set overlaps an excluded task set")


def tasks_from_payload(payload: Mapping[str, Any]) -> list[TaskConfig]:
    return [
        TaskConfig(
            task_id=item["task_id"], seed=int(item["seed"]),
            object_position=np.asarray(item["object_position"], dtype=np.float64),
            goal_position=np.asarray(item["goal_position"], dtype=np.float64),
            target_radius=float(item["target_radius"]),
        )
        for item in payload["tasks"]
    ]


def final_checkpoint_path(root: Path, method: str, seed: int) -> Path:
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Unknown final checkpoint method or seed")
    return root / "outputs/checkpoints/curriculum_v1" / method / f"seed_{seed}" / "final.pt"


def validate_final_checkpoint(path: Path, method: str, seed: int) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise ValueError("Final checkpoint fingerprint mismatch")
    if checkpoint["method_name"] != method or int(checkpoint["training_seed"]) != seed:
        raise ValueError("Final checkpoint method/seed mismatch")
    if int(checkpoint["optimizer_step"]) != 20600:
        raise ValueError("Final benchmark must use optimizer update 20,600")
    return checkpoint


def success_rate(episodes: Sequence[Mapping[str, Any]]) -> float:
    if not episodes:
        raise ValueError("Cannot calculate success rate for no episodes")
    return sum(bool(episode["success"]) for episode in episodes) / len(episodes)


def remaining_task_ids(
    expected_task_ids: Sequence[str], completed_episodes: Sequence[Mapping[str, Any]]
) -> list[str]:
    expected = [str(value) for value in expected_task_ids]
    completed = [str(row["task_id"]) for row in completed_episodes]
    if len(expected) != len(set(expected)) or len(completed) != len(set(completed)):
        raise ValueError("Resume task IDs must be unique")
    if not set(completed) <= set(expected):
        raise ValueError("Resume data contains unknown task IDs")
    done = set(completed)
    return [task_id for task_id in expected if task_id not in done]


def paired_outcomes(
    method_episodes: Sequence[Mapping[str, Any]],
    uniform_episodes: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    method = {str(row["task_id"]): bool(row["success"]) for row in method_episodes}
    uniform = {str(row["task_id"]): bool(row["success"]) for row in uniform_episodes}
    if len(method) != len(method_episodes) or len(uniform) != len(uniform_episodes) or set(method) != set(uniform):
        raise ValueError("Paired comparisons require identical unique task IDs")
    both = sum(method[key] and uniform[key] for key in method)
    only_method = sum(method[key] and not uniform[key] for key in method)
    only_uniform = sum(not method[key] and uniform[key] for key in method)
    neither = len(method) - both - only_method - only_uniform
    return {
        "task_count": len(method), "both_solve": both,
        "only_method_solves": only_method, "only_uniform_solves": only_uniform,
        "neither_solves": neither,
        "success_rate_difference": (only_method - only_uniform) / len(method),
    }


def paired_task_bootstrap(
    method_episodes: Sequence[Mapping[str, Any]], uniform_episodes: Sequence[Mapping[str, Any]],
    *, samples: int, seed: int,
) -> dict[str, float | int]:
    comparison = paired_outcomes(method_episodes, uniform_episodes)
    method = {str(row["task_id"]): int(bool(row["success"])) for row in method_episodes}
    uniform = {str(row["task_id"]): int(bool(row["success"])) for row in uniform_episodes}
    ids = sorted(method); differences = np.asarray([method[key] - uniform[key] for key in ids], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.asarray([np.mean(differences[rng.integers(0, len(ids), len(ids))]) for _ in range(samples)])
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": float(comparison["success_rate_difference"]),
        "ci_lower": float(lower), "ci_upper": float(upper), "samples": samples,
        "interpretation": "task-level paired bootstrap; does not represent training-seed uncertainty",
    }


def supported_failure_stage(episode: Mapping[str, Any]) -> str:
    if bool(episode["success"]):
        return "success"
    if episode.get("failure_reason") == "object_left_workspace":
        return "object_left_workspace"
    events = episode.get("physical_events", {})
    if not events.get("reaches_object_vicinity", False):
        return "failed_to_reach_object"
    if not events.get("gripper_closes", False) or not events.get("object_lifted", False):
        return "grasp_or_lift_failure"
    if not events.get("transported_toward_goal", False):
        return "transport_failure"
    if not events.get("reaches_goal_vicinity", False):
        return "place_approach_failure"
    if not events.get("releases_after_close", False):
        return "release_failure"
    return "timeout_unclassified"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
