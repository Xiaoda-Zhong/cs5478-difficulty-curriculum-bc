#!/usr/bin/env python3
"""Visualize autonomous Uniform BC rollouts in the PyBullet GUI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (REPOSITORY_ROOT / "data/metadata/dataset_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = load_or_create_evaluation_tasks(
        REPOSITORY_ROOT / "data/metadata/id_eval_tasks_v1.json",
        seed=int(config["bc"]["evaluation_seed"]),
        count=int(config["bc"]["evaluation_tasks"]),
        dataset_fingerprint=metadata["dataset_fingerprint"],
        demonstration_tasks=metadata["trajectory_summaries"],
    )
    if not 0 <= args.task_index < len(tasks):
        parser.error(f"--task-index must be between 0 and {len(tasks) - 1}")
    device = select_device(args.device)
    policy, _ = load_policy_checkpoint(args.checkpoint, device)

    with PandaPickPlaceEnv(mode="gui") as env:
        for offset in range(args.episodes):
            task = tasks[(args.task_index + offset) % len(tasks)]
            result = rollout_bc_policy(
                env,
                policy,
                task,
                int(config["bc"]["max_rollout_steps"]),
                step_callback=lambda _step, _observation, _action: time.sleep(0.02),
            )
            print(
                f"{task.task_id}: {'SUCCESS' if result.success else 'FAILURE'} | "
                f"steps={result.steps} | reason={result.failure_reason} | "
                f"final_xy_error={result.final_object_goal_xy_error:.4f} m"
            )
            time.sleep(max(0.0, args.hold_seconds))


if __name__ == "__main__":
    main()
