#!/usr/bin/env python3
"""Evaluate scripted-expert reliability on reproducibly sampled tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.task_sampler import TaskSampler  # noqa: E402
from data.trajectory import rollout_expert  # noqa: E402
from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from expert.scripted_expert import ScriptedExpert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5478)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs"
        / "results"
        / "randomized_expert_evaluation.json",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    sampler = TaskSampler(args.seed)
    successes = 0
    lengths: list[int] = []
    path_lengths: list[float] = []
    failures: Counter[str] = Counter()
    records: list[dict[str, object]] = []

    with PandaPickPlaceEnv(mode="direct") as env:
        expert = ScriptedExpert()
        for index in range(args.episodes):
            task = sampler.sample()
            trajectory = rollout_expert(env, expert, task, f"eval_{index:04d}")
            success = bool(trajectory["success"])
            successes += int(success)
            lengths.append(int(trajectory["trajectory_length"]))
            path_lengths.append(float(trajectory["ee_path_length"]))
            if not success:
                failures[str(trajectory["failure_reason"] or "unknown")] += 1
            records.append(
                {
                    **task.as_dict(),
                    "success": success,
                    "steps": int(trajectory["trajectory_length"]),
                    "failure_reason": trajectory["failure_reason"] or None,
                    "final_object_position": np.asarray(
                        trajectory["final_object_position"]
                    ).tolist(),
                }
            )
            if (index + 1) % 20 == 0:
                print(f"Evaluated {index + 1} / {args.episodes} tasks")

    length_array = np.asarray(lengths, dtype=float)
    path_array = np.asarray(path_lengths, dtype=float)
    print("\nRandomized expert evaluation")
    print(f"Episodes: {args.episodes}")
    print(f"Success: {successes}")
    print(f"Success rate: {successes / args.episodes:.1%}")
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
    print(f"Failure reasons: {dict(failures) if failures else 'none'}")
    report = {
        "seed": args.seed,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "trajectory_length": {
            "mean": float(length_array.mean()),
            "std": float(length_array.std()),
            "min": int(length_array.min()),
            "max": int(length_array.max()),
        },
        "ee_path_length": {
            "mean": float(path_array.mean()),
            "std": float(path_array.std()),
            "min": float(path_array.min()),
            "max": float(path_array.max()),
        },
        "failure_reasons": dict(failures),
        "tasks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation records: {args.output}")
    if successes / args.episodes < 0.95:
        raise SystemExit("Expert reliability is below 95%; do not collect the dataset")


if __name__ == "__main__":
    main()
