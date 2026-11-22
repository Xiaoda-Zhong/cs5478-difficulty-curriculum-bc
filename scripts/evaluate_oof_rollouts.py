#!/usr/bin/env python3
"""Evaluate every training demonstration task with its held-out OOF policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.difficulty_diagnostics import (  # noqa: E402
    assert_held_out,
    evaluation_task_from_trajectory,
    failure_reason_counts,
    summarize_fold_outcomes,
    validate_checkpoint_for_fold,
)
from data.trajectory import load_trajectory  # noqa: E402
from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from evaluation.rollout import rollout_bc_policy  # noqa: E402
from models.bc_policy import load_policy_checkpoint  # noqa: E402
from training.train_bc import select_device  # noqa: E402

FIELDS = [
    "trajectory_id", "fold_id", "oof_checkpoint", "task_difficulty",
    "expert_difficulty", "learner_difficulty", "success", "failure",
    "rollout_steps", "final_object_goal_xy_error", "failure_reason",
    "object_position", "goal_position", "target_radius", "physical_events",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/default.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run one held-out task per fold without writing outputs")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    horizon = int(config["analysis"]["rollout_horizon"])
    folds_payload = _read_json(REPOSITORY_ROOT / "data/metadata/learner_oof_folds_v1.json")
    scores = _read_scores(REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_scores_v1.csv")
    expected_fingerprint = str(folds_payload["dataset_fingerprint"])
    device = select_device(args.device)
    output_dir = REPOSITORY_ROOT / "outputs/results/difficulty_analysis"
    final_path = output_dir / "oof_rollout_results_v1.csv"
    partial_path = output_dir / "oof_rollout_results_v1.partial.csv"
    metadata_path = output_dir / "oof_rollout_metadata_v1.json"

    if args.smoke_test:
        rows, _ = _evaluate(folds_payload, scores, expected_fingerprint, device, horizon, smoke_test=True)
        print(f"Smoke test passed: {len(rows)} held-out rollouts, one per fold")
        return
    if args.overwrite:
        for path in (final_path, partial_path, metadata_path):
            if path.exists():
                path.unlink()
    elif final_path.exists():
        raise SystemExit(f"Completed results already exist at {final_path}; pass --overwrite to replace")

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_rows = _read_partial(partial_path)
    completed_ids = {row["trajectory_id"] for row in completed_rows}
    if completed_ids - set(scores):
        raise SystemExit("Partial output contains unknown trajectory IDs")
    mode = "a" if partial_path.exists() else "w"
    with partial_path.open(mode, newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        rows, checkpoint_metadata = _evaluate(
            folds_payload, scores, expected_fingerprint, device, horizon,
            skip_ids=completed_ids, writer=writer, output_file=output_file,
        )
    all_rows = _read_partial(partial_path)
    expected_ids = set(scores)
    if len(all_rows) != len(expected_ids) or {row["trajectory_id"] for row in all_rows} != expected_ids:
        raise RuntimeError("Evaluation did not produce exactly one row for every scored trajectory")
    partial_path.replace(final_path)
    parsed_rows = [_parse_row(row) for row in all_rows]
    successes = sum(bool(row["success"]) for row in parsed_rows)
    metadata = {
        "version": 1,
        "date_generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": expected_fingerprint,
        "fold_assignment_file": "data/metadata/learner_oof_folds_v1.json",
        "fold_configuration": {
            "fold_seed": int(folds_payload["fold_seed"]),
            "number_of_folds": int(folds_payload["number_of_folds"]),
            "training_split_fingerprint": folds_payload["training_split_fingerprint"],
            "held_out_tasks_per_fold": [len(fold["held_out_trajectory_ids"]) for fold in folds_payload["folds"]],
        },
        "difficulty_scores_file": "outputs/results/difficulty/difficulty_scores_v1.csv",
        "rollout_horizon": horizon,
        "device": str(device),
        "environment_config": config,
        "checkpoints": checkpoint_metadata,
        "summary": {
            "tasks": len(parsed_rows), "successes": successes,
            "failures": len(parsed_rows) - successes,
            "success_rate": successes / len(parsed_rows),
            "mean_rollout_steps": float(np.mean([row["rollout_steps"] for row in parsed_rows])),
            "mean_final_object_goal_xy_error": float(np.mean([row["final_object_goal_xy_error"] for row in parsed_rows])),
            "failure_reasons": failure_reason_counts(parsed_rows),
            "per_fold": summarize_fold_outcomes(parsed_rows),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2, sort_keys=True))
    print(f"Results: {final_path}")


def _evaluate(
    folds_payload: dict[str, Any], scores: dict[str, dict[str, str]], fingerprint: str,
    device: object, horizon: int, *, smoke_test: bool = False,
    skip_ids: set[str] | None = None, writer: csv.DictWriter | None = None,
    output_file: Any = None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    skip_ids = skip_ids or set()
    rows: list[dict[str, str]] = []
    checkpoint_metadata = []
    total = sum(len(fold["held_out_trajectory_ids"]) for fold in folds_payload["folds"])
    processed = len(skip_ids)
    with PandaPickPlaceEnv(mode="direct") as env:
        for fold in folds_payload["folds"]:
            fold_id = int(fold["fold_index"])
            checkpoint_path = REPOSITORY_ROOT / f"outputs/checkpoints/learner_oof/fold_{fold_id}.pt"
            policy, checkpoint = load_policy_checkpoint(checkpoint_path, device)
            if checkpoint["dataset_fingerprint"] != fingerprint:
                raise ValueError(f"Fold {fold_id} checkpoint fingerprint mismatch")
            validate_checkpoint_for_fold(checkpoint, fold)
            checkpoint_metadata.append({
                "fold_id": fold_id, "path": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
                "sha256": _sha256(checkpoint_path), "epoch": int(checkpoint["epoch"]),
                "held_out_count": len(fold["held_out_trajectory_ids"]),
            })
            ids = fold["held_out_trajectory_ids"][:1] if smoke_test else fold["held_out_trajectory_ids"]
            for trajectory_id in ids:
                if trajectory_id in skip_ids:
                    continue
                assert_held_out(checkpoint, trajectory_id)
                trajectory = load_trajectory(REPOSITORY_ROOT / f"data/demonstrations/{trajectory_id}.npz")
                task = evaluation_task_from_trajectory(trajectory_id, trajectory)
                result = rollout_bc_policy(env, policy, task, horizon)
                score = scores[trajectory_id]
                row = {
                    "trajectory_id": trajectory_id, "fold_id": str(fold_id),
                    "oof_checkpoint": str(checkpoint_path.relative_to(REPOSITORY_ROOT)),
                    "task_difficulty": score["task_difficulty"],
                    "expert_difficulty": score["expert_difficulty"],
                    "learner_difficulty": score["learner_difficulty"],
                    "success": str(int(result.success)), "failure": str(int(not result.success)),
                    "rollout_steps": str(result.steps),
                    "final_object_goal_xy_error": repr(result.final_object_goal_xy_error),
                    "failure_reason": result.failure_reason or "",
                    "object_position": json.dumps(task.object_position.tolist()),
                    "goal_position": json.dumps(task.goal_position.tolist()),
                    "target_radius": repr(task.target_radius),
                    "physical_events": json.dumps(result.physical_events, sort_keys=True),
                }
                rows.append(row)
                if writer is not None:
                    writer.writerow(row)
                    output_file.flush()
                processed += 1
                if smoke_test or processed % 10 == 0:
                    print(f"Evaluated {processed} / {total}: {trajectory_id} fold={fold_id} success={result.success}", flush=True)
    return rows, checkpoint_metadata


def _read_scores(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    result = {row["trajectory_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate IDs in difficulty scores")
    return result


def _read_partial(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != FIELDS:
            raise ValueError("Partial output schema mismatch")
        rows = list(reader)
    if len({row["trajectory_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs in partial output")
    return rows


def _parse_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = dict(row)
    parsed["success"] = row["success"] == "1"
    parsed["failure"] = row["failure"] == "1"
    parsed["fold_id"] = int(row["fold_id"])
    parsed["rollout_steps"] = int(row["rollout_steps"])
    parsed["final_object_goal_xy_error"] = float(row["final_object_goal_xy_error"])
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
