#!/usr/bin/env python3
"""Evaluate Uniform BC autonomously on frozen fresh ID tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from evaluation.rollout import (  # noqa: E402
    load_or_create_evaluation_tasks,
    rollout_bc_policy,
)
from models.bc_policy import load_policy_checkpoint  # noqa: E402
from training.train_bc import select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/checkpoints/uniform_bc/best.pt",
    )
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/default.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bc_config = config["bc"]
    device = select_device(args.device)
    policy, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    metadata = json.loads(
        (REPOSITORY_ROOT / "data/metadata/dataset_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if checkpoint["dataset_fingerprint"] != metadata["dataset_fingerprint"]:
        raise SystemExit("Checkpoint and frozen dataset fingerprints do not match")
    tasks = load_or_create_evaluation_tasks(
        REPOSITORY_ROOT / "data/metadata/id_eval_tasks_v1.json",
        seed=int(bc_config["evaluation_seed"]),
        count=int(bc_config["evaluation_tasks"]),
        dataset_fingerprint=metadata["dataset_fingerprint"],
        demonstration_tasks=metadata["trajectory_summaries"],
    )
    run_name = args.run_name or args.checkpoint.parent.name
    output_path = REPOSITORY_ROOT / "outputs/results" / run_name / "rollout_evaluation.json"
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Result exists at {output_path}; pass --overwrite explicitly")

    results = []
    with PandaPickPlaceEnv(mode="direct") as env:
        for index, task in enumerate(tasks, start=1):
            result = rollout_bc_policy(
                env, policy, task, int(bc_config["max_rollout_steps"])
            )
            results.append(result)
            if index % 20 == 0:
                print(f"Evaluated {index} / {len(tasks)} tasks")

    successes = sum(result.success for result in results)
    lengths = np.asarray([result.steps for result in results], dtype=float)
    errors = np.asarray(
        [result.final_object_goal_xy_error for result in results], dtype=float
    )
    failures = Counter(
        result.failure_reason for result in results if not result.success
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "evaluation_seed": int(bc_config["evaluation_seed"]),
        "tasks": len(tasks),
        "successful_episodes": successes,
        "success_rate": successes / len(tasks),
        "mean_rollout_length": float(lengths.mean()),
        "median_rollout_length": float(np.median(lengths)),
        "failure_reasons": dict(failures),
        "mean_final_object_goal_xy_error": float(errors.mean()),
        "episodes": [result.as_dict() for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_failed_rollouts(
        results,
        REPOSITORY_ROOT / "outputs/figures" / run_name / "rollout_diagnostics.png",
    )
    print("\nUniform BC closed-loop evaluation")
    print(f"Success: {successes} / {len(tasks)} ({successes / len(tasks):.1%})")
    print(f"Mean rollout length: {lengths.mean():.2f}")
    print(f"Median rollout length: {np.median(lengths):.2f}")
    print(f"Failure reasons: {dict(failures) if failures else 'none'}")
    print(f"Mean final object-goal XY error: {errors.mean():.4f} m")
    print(f"Results: {output_path}")


def _plot_failed_rollouts(results, output_path: Path) -> None:
    failed = [result for result in results if not result.success][:5]
    if not failed:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    for result in failed:
        axis.plot(result.object_goal_xy_distance_trace, label=result.task_id)
    axis.set(
        xlabel="Rollout step",
        ylabel="Object-goal XY distance (m)",
        title="Failed rollout diagnostics",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
