#!/usr/bin/env python3
"""Watch one complete deterministic stacking-expert trajectory in PyBullet GUI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_stacking import PandaStackingEnv  # noqa: E402
from expert.stacking_expert import ScriptedStackingExpert  # noqa: E402
from scripts.evaluate_randomized_stacking_expert import sample_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--seed", type=int, default=25478)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    args = parser.parse_args()
    with PandaStackingEnv(mode="gui") as env:
        if args.randomized:
            task = sample_tasks(args.task_index + 1, args.seed)[args.task_index]
            observation = env.reset(task["source_position"], task["support_position"],
                                    seed=task["seed"])
            print(f"Task: {task}")
        else:
            observation = env.reset()
            print("Task: default")
        expert = ScriptedStackingExpert()
        expert.reset(env)
        previous = None
        while not expert.is_done:
            if expert.current_phase != previous:
                print(expert.current_phase.value, flush=True)
                previous = expert.current_phase
            action = expert.act(observation)
            observation, _ = env.step(action)
            time.sleep(0.01)
        result = expert.result()
        print("SUCCESS" if result.success else f"FAILURE: {result.failure_reason}")
        print(result.final_diagnostics)
        time.sleep(max(0.0, args.hold_seconds))


if __name__ == "__main__":
    main()
