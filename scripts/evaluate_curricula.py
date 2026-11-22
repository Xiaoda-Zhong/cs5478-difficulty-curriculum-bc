#!/usr/bin/env python3
"""Resumably evaluate curriculum checkpoints on the frozen 100 ID tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.task_sampler import TaskConfig  # noqa: E402
from difficulty.normalization import equal_weight_composite, standardize  # noqa: E402
from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from evaluation.rollout import rollout_bc_policy  # noqa: E402
from models.bc_policy import load_policy_checkpoint  # noqa: E402

DATASET_FINGERPRINT = "1b6cec9dda067778630268c307241c62cccbd49cfe849a031877ee00ea938bc0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/default.yaml").read_text())
    curriculum = config["curriculum"]
    methods = args.methods or list(curriculum["methods"])
    seeds = args.seeds or list(curriculum["seeds"])
    if args.smoke:
        seeds = [seeds[0]]
    root_name = "curriculum_v1_smoke" if args.smoke else "curriculum_v1"
    checkpoint_root = REPOSITORY_ROOT / "outputs/checkpoints" / root_name
    result_root = REPOSITORY_ROOT / "outputs/results" / root_name / "evaluations"
    result_root.mkdir(parents=True, exist_ok=True)
    task_limit = 3 if args.smoke else 100
    group_path = REPOSITORY_ROOT / "data/metadata/id_eval_task_groups_v1.json"
    groups = load_or_create_evaluation_groups(group_path)
    jobs: list[dict[str, Any]] = []
    if not args.smoke:
        for seed in seeds:
            initial = checkpoint_root / f"initial_eval_seed_{seed}.pt"
            if not initial.exists():
                raise SystemExit(f"Missing initial checkpoint: {initial}")
            jobs.append(_job(initial, result_root / "initial" / f"seed_{seed}" / "step_000000.json", "initial", seed, 0, task_limit, args.overwrite))
    for method in methods:
        for seed in seeds:
            run_dir = checkpoint_root / method / f"seed_{seed}"
            checkpoints = sorted(run_dir.glob("step_*.pt"))
            if args.smoke:
                checkpoints = checkpoints[-1:]
            if not checkpoints:
                raise SystemExit(f"No checkpoints found in {run_dir}")
            for checkpoint in checkpoints:
                step = int(checkpoint.stem.split("_")[1])
                output = result_root / method / f"seed_{seed}" / f"step_{step:06d}.json"
                jobs.append(_job(checkpoint, output, method, seed, step, task_limit, args.overwrite))
    pending = [job for job in jobs if job["overwrite"] or not Path(job["output_path"]).exists()]
    print(f"Evaluation jobs: total={len(jobs)}, pending={len(pending)}, workers={args.workers or curriculum['evaluation_workers']}")
    workers = max(1, int(args.workers or curriculum["evaluation_workers"]))
    if workers == 1:
        for index, job in enumerate(pending, start=1):
            result = evaluate_checkpoint_job(job)
            print(f"[{index}/{len(pending)}] {result}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(evaluate_checkpoint_job, job): job for job in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                print(f"[{index}/{len(pending)}] {future.result()}", flush=True)
    manifest = {
        "version": 1, "date_generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "evaluation_tasks": "data/metadata/id_eval_tasks_v1.json",
        "evaluation_task_groups": str(group_path.relative_to(REPOSITORY_ROOT)),
        "evaluation_seed": 15478, "task_count": task_limit,
        "rollout_horizon": int(config["bc"]["max_rollout_steps"]),
        "methods": methods, "seeds": seeds,
        "checkpoint_jobs": len(jobs), "task_groups": groups["groups"],
    }
    (result_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _job(checkpoint: Path, output: Path, method: str, seed: int, step: int, task_limit: int, overwrite: bool) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint), "output_path": str(output),
        "method": method, "seed": seed, "optimizer_step": step,
        "task_limit": task_limit, "overwrite": overwrite,
        "horizon": 400, "task_path": str(REPOSITORY_ROOT / "data/metadata/id_eval_tasks_v1.json"),
    }


def evaluate_checkpoint_job(job: dict[str, Any]) -> str:
    checkpoint_path = Path(job["checkpoint_path"])
    output_path = Path(job["output_path"])
    partial_path = output_path.with_suffix(".partial.jsonl")
    if output_path.exists() and not job["overwrite"]:
        return f"skip {job['method']} seed={job['seed']} step={job['optimizer_step']}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if job["overwrite"]:
        output_path.unlink(missing_ok=True); partial_path.unlink(missing_ok=True)
    tasks_payload = json.loads(Path(job["task_path"]).read_text())
    if tasks_payload["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise ValueError("Frozen evaluation-task fingerprint mismatch")
    tasks = [_task(item) for item in tasks_payload["tasks"][: int(job["task_limit"])]]
    completed: list[dict[str, Any]] = []
    if partial_path.exists():
        completed = [json.loads(line) for line in partial_path.read_text().splitlines() if line]
    completed_ids = {episode["task_id"] for episode in completed}
    retained_failure_traces = sum(
        not episode["success"] and "object_goal_xy_distance_trace" in episode
        for episode in completed
    )
    trace_limit = int(job.get("retain_failure_traces", 10**9))
    policy, checkpoint = load_policy_checkpoint(checkpoint_path, "cpu")
    if checkpoint["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise ValueError("Curriculum checkpoint dataset fingerprint mismatch")
    if int(checkpoint["training_seed"]) != int(job["seed"]):
        raise ValueError("Curriculum checkpoint seed mismatch")
    with partial_path.open("a", encoding="utf-8") as output, PandaPickPlaceEnv(mode="direct") as env:
        for task in tasks:
            if task.task_id in completed_ids:
                continue
            result = rollout_bc_policy(env, policy, task, int(job["horizon"]))
            episode = result.as_dict()
            if episode["success"] or retained_failure_traces >= trace_limit:
                episode.pop("object_goal_xy_distance_trace", None)
            else:
                retained_failure_traces += 1
            output.write(json.dumps(episode, sort_keys=True) + "\n"); output.flush()
            completed.append(episode)
    order = {task.task_id: index for index, task in enumerate(tasks)}
    completed.sort(key=lambda episode: order[episode["task_id"]])
    if len(completed) != len(tasks) or {episode["task_id"] for episode in completed} != set(order):
        raise RuntimeError("Incomplete or duplicate curriculum evaluation results")
    successes = sum(bool(episode["success"]) for episode in completed)
    summary = {
        "version": 1, "dataset_fingerprint": DATASET_FINGERPRINT,
        "method": job["method"], "training_seed": int(job["seed"]),
        **({"evaluation_split": job["evaluation_split"]} if "evaluation_split" in job else {}),
        "optimizer_step": int(job["optimizer_step"]),
        "checkpoint": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
        "initial_state_fingerprint": checkpoint["initial_state_fingerprint"],
        "tasks": len(completed), "successes": successes,
        "success_rate": successes / len(completed),
        "mean_rollout_length": float(np.mean([episode["steps"] for episode in completed])),
        "median_rollout_length": float(np.median([episode["steps"] for episode in completed])),
        "mean_final_object_goal_xy_error": float(np.mean([episode["final_object_goal_xy_error"] for episode in completed])),
        "failure_reasons": _failure_counts(completed), "episodes": completed,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_path.unlink()
    return f"{job['method']} seed={job['seed']} step={job['optimizer_step']} success={successes}/{len(completed)}"


def load_or_create_evaluation_groups(path: Path) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["dataset_fingerprint"] != DATASET_FINGERPRINT:
            raise ValueError("Evaluation group fingerprint mismatch")
        return payload
    tasks_payload = json.loads((REPOSITORY_ROOT / "data/metadata/id_eval_tasks_v1.json").read_text())
    difficulty_metadata = json.loads((REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_metadata_v1.json").read_text())
    normalization = difficulty_metadata["normalization_parameters"]["task"]
    records = []
    with PandaPickPlaceEnv(mode="direct") as env:
        for item in tasks_payload["tasks"]:
            task = _task(item)
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
        "definition": "M5 predefined task-difficulty formula and normalization; no rollout outcomes used",
        "tie_breaker": "task_id ascending", "records": records,
        "groups": {
            "easy": [row["task_id"] for row in ordered[:33]],
            "medium": [row["task_id"] for row in ordered[33:66]],
            "hard": [row["task_id"] for row in ordered[66:]],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _task(item: dict[str, Any]) -> TaskConfig:
    return TaskConfig(
        task_id=item["task_id"], seed=int(item["seed"]),
        object_position=np.asarray(item["object_position"], dtype=np.float64),
        goal_position=np.asarray(item["goal_position"], dtype=np.float64),
        target_radius=float(item["target_radius"]),
    )


def _failure_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for episode in episodes:
        if not episode["success"]:
            reason = str(episode["failure_reason"])
            result[reason] = result.get(reason, 0) + 1
    return result


if __name__ == "__main__":
    main()
