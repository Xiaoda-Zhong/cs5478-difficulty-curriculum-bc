#!/usr/bin/env python3
"""Visually inspect deterministic expert rollouts on sampled GUI tasks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.task_sampler import TaskSampler  # noqa: E402
from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from expert.scripted_expert import ScriptedExpert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--control-delay", type=float, default=0.02)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    sampler = TaskSampler(args.seed)
    with PandaPickPlaceEnv(mode="gui") as env:
        expert = ScriptedExpert()
        for episode in range(1, args.episodes + 1):
            task = sampler.sample()
            observation = env.reset(
                object_position=task.object_position,
                goal_position=task.goal_position,
                target_radius=task.target_radius,
                seed=task.seed,
            )
            expert.reset(env)
            print(
                f"Episode {episode}: object={task.object_position[:2]}, "
                f"goal={task.goal_position[:2]}, radius={task.target_radius:.3f}"
            )
            while not expert.is_done:
                observation, _ = env.step(expert.act(observation))
                time.sleep(max(0.0, args.control_delay))
            result = expert.result()
            print(
                f"  {'SUCCESS' if result.success else 'FAILURE'} | "
                f"steps={result.steps} | reason={result.failure_reason}"
            )
            time.sleep(max(0.0, args.hold_seconds))


if __name__ == "__main__":
    main()
