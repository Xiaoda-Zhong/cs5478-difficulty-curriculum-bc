#!/usr/bin/env python3
"""Evaluate S2 expert on a fixed conservative randomized task distribution."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_stacking import PandaStackingEnv  # noqa: E402
from expert.stacking_expert import ScriptedStackingExpert, rollout_stacking_expert  # noqa: E402

SOURCE_X = (0.46, 0.64)
SOURCE_Y = (-0.24, -0.08)
SUPPORT_X = (0.46, 0.64)
SUPPORT_Y = (0.08, 0.24)
EVALUATION_SEED = 25478


def sample_tasks(count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    tasks = []
    for index in range(count):
        task_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        task_rng = np.random.default_rng(task_seed)
        source = [task_rng.uniform(*SOURCE_X), task_rng.uniform(*SOURCE_Y), 0.025]
        support = [task_rng.uniform(*SUPPORT_X), task_rng.uniform(*SUPPORT_Y), 0.025]
        tasks.append({"task_id": f"stacking_eval_{index:04d}", "seed": task_seed,
                      "source_position": source, "support_position": support})
    return tasks


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    args = parser.parse_args()
    tasks = sample_tasks(args.tasks, args.seed)
    records = []
    with PandaStackingEnv(mode="direct") as env:
        for index, task in enumerate(tasks):
            env.reset(task["source_position"], task["support_position"], seed=task["seed"])
            trace = rollout_stacking_expert(env, ScriptedStackingExpert())
            result = trace["result"]
            ee = np.asarray(trace["ee_positions"])
            actions = np.asarray(trace["actions"])
            record = {
                **task, "success": result.success, "failure_reason": result.failure_reason,
                "steps": result.steps,
                "ee_path_length": float(np.linalg.norm(np.diff(ee, axis=0), axis=1).sum()),
                "cumulative_action_magnitude": float(np.linalg.norm(actions[:, :3], axis=1).sum()),
                "final_alignment_error": result.final_diagnostics["horizontal_alignment_error"],
                "maximum_support_displacement": result.maximum_support_displacement,
                "stable_steps": result.final_diagnostics["stable_steps"],
            }
            records.append(record)
            if (index + 1) % 10 == 0 or not result.success:
                print(f"[{index + 1:03d}/{args.tasks}] success={sum(r['success'] for r in records)} "
                      f"last={result.success} steps={result.steps} reason={result.failure_reason}", flush=True)
    successes = sum(row["success"] for row in records)
    summary = {
        "version": 1, "task": "object_stacking", "evaluation_seed": args.seed,
        "tasks": args.tasks, "successes": successes, "success_rate": successes / args.tasks,
        "distribution": {
            "source_x_range": list(SOURCE_X), "source_y_range": list(SOURCE_Y),
            "support_x_range": list(SUPPORT_X), "support_y_range": list(SUPPORT_Y),
            "cube_z": 0.025,
            "note": "disjoint y bands ensure safe initial separation; no task is filtered by outcome",
        },
        "steps": _stats([row["steps"] for row in records]),
        "ee_path_length": _stats([row["ee_path_length"] for row in records]),
        "cumulative_action_magnitude": _stats([row["cumulative_action_magnitude"] for row in records]),
        "failure_reasons": dict(Counter(row["failure_reason"] for row in records if not row["success"])),
        "records": records,
    }
    output = ROOT / "outputs/results/stacking_s2/randomized_expert_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSuccess: {successes}/{args.tasks} ({successes / args.tasks:.1%})")
    print(f"Steps: {summary['steps']}")
    print(f"EE path length: {summary['ee_path_length']}")
    print(f"Failures: {summary['failure_reasons']}")
    print(f"Saved: {output}")
    if successes / args.tasks < 0.95:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
