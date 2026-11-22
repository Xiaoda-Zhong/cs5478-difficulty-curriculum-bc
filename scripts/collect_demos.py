#!/usr/bin/env python3
"""Collect and freeze successful expert demonstrations in DIRECT mode."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.task_sampler import TaskSampler  # noqa: E402
from data.trajectory import (  # noqa: E402
    fingerprint_entry,
    rollout_expert,
    save_trajectory,
    task_fingerprint,
)
from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from expert.scripted_expert import ScriptedExpert  # noqa: E402


DEMONSTRATION_DIR = REPOSITORY_ROOT / "data" / "demonstrations"
METADATA_DIR = REPOSITORY_ROOT / "data" / "metadata"
METADATA_PATH = METADATA_DIR / "dataset_metadata.json"
COLLECTION_LOG_PATH = METADATA_DIR / "collection_log.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trajectories", type=int, default=300)
    parser.add_argument("--seed", type=int, default=5478)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_trajectories <= 0:
        parser.error("--num-trajectories must be positive")

    _prepare_output(args.overwrite)
    sampler = TaskSampler(args.seed)
    collection_log: list[dict[str, Any]] = []
    trajectory_summaries: list[dict[str, Any]] = []
    fingerprint_entries: list[dict[str, Any]] = []
    lengths: list[int] = []
    path_lengths: list[float] = []
    failures: Counter[str] = Counter()
    successful = 0
    attempted = 0
    allowed_failures = max(5, math.ceil(args.num_trajectories * 0.05))
    max_attempts = args.num_trajectories + allowed_failures

    with PandaPickPlaceEnv(mode="direct") as env:
        expert = ScriptedExpert()
        while successful < args.num_trajectories and attempted < max_attempts:
            task = sampler.sample()
            trajectory_id = f"traj_{successful:04d}"
            trajectory = rollout_expert(env, expert, task, trajectory_id)
            attempted += 1
            success = bool(trajectory["success"])
            log_entry = {
                "attempt": attempted,
                **task.as_dict(),
                "success": success,
                "failure_reason": trajectory["failure_reason"] or None,
                "steps": int(trajectory["trajectory_length"]),
                "final_object_position": np.asarray(
                    trajectory["final_object_position"]
                ).tolist(),
            }
            collection_log.append(log_entry)

            if not success:
                failures[str(trajectory["failure_reason"] or "unknown")] += 1
                continue

            save_trajectory(
                DEMONSTRATION_DIR / f"{trajectory_id}.npz", trajectory
            )
            entry = fingerprint_entry(trajectory)
            fingerprint_entries.append(entry)
            length = int(trajectory["trajectory_length"])
            path_length = float(trajectory["ee_path_length"])
            lengths.append(length)
            path_lengths.append(path_length)
            trajectory_summaries.append(
                {
                    **entry,
                    "trajectory_length": length,
                    "ee_path_length": path_length,
                    "cumulative_action_magnitude": float(
                        trajectory["cumulative_action_magnitude"]
                    ),
                }
            )
            successful += 1
            if successful % 50 == 0 or successful == args.num_trajectories:
                print(
                    f"Collected {successful} / {args.num_trajectories} "
                    "successful trajectories"
                )

        config = env.config

    _write_json(COLLECTION_LOG_PATH, collection_log)
    if successful < args.num_trajectories:
        raise SystemExit(
            f"Stopped after {attempted} attempts with {successful} successes; "
            "failure rate exceeded the 5% collection threshold"
        )

    fingerprint = task_fingerprint(fingerprint_entries)
    metadata = {
        "dataset_version": int(config["dataset"]["version"]),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "global_dataset_seed": args.seed,
        "number_of_trajectories": successful,
        "number_of_total_state_action_pairs": int(sum(lengths)),
        "observation_dimension": 10,
        "action_dimension": 4,
        "task_randomization": config["task_randomization"],
        "expert_configuration": config["expert"],
        "environment_configuration_summary": {
            key: config[key] for key in ("physics", "robot", "task", "control")
        },
        "success_filtering_rule": (
            "Only rollouts with expert success=True and env.is_success()=True are saved"
        ),
        "attempted_episodes": attempted,
        "failed_episodes": attempted - successful,
        "failure_reasons": dict(failures),
        "fingerprint_basis": "SHA256 of ordered successful task configurations",
        "dataset_fingerprint": fingerprint,
        "trajectory_summaries": trajectory_summaries,
    }
    _write_json(METADATA_PATH, metadata)

    length_array = np.asarray(lengths, dtype=float)
    path_array = np.asarray(path_lengths, dtype=float)
    print("\nDataset collection complete")
    print(f"Successful trajectories: {successful}")
    print(f"Attempted trajectories: {attempted}")
    print(f"Failed trajectories: {attempted - successful}")
    print(f"Total state-action pairs: {sum(lengths)}")
    print(
        "Trajectory length: "
        f"mean={length_array.mean():.2f}, std={length_array.std():.2f}, "
        f"min={int(length_array.min())}, max={int(length_array.max())}"
    )
    print(
        "EE path length (m): "
        f"mean={path_array.mean():.4f}, std={path_array.std():.4f}, "
        f"min={path_array.min():.4f}, max={path_array.max():.4f}"
    )
    print(f"Output directory: {DEMONSTRATION_DIR}")
    print(f"Dataset fingerprint: {fingerprint}")


def _prepare_output(overwrite: bool) -> None:
    DEMONSTRATION_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(DEMONSTRATION_DIR.glob("traj_*.npz"))
    existing.extend(
        path for path in (METADATA_PATH, COLLECTION_LOG_PATH) if path.exists()
    )
    if existing and not overwrite:
        raise SystemExit(
            "Dataset output already exists; pass --overwrite to regenerate it explicitly"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
