#!/usr/bin/env python3
"""Evaluate the deterministic scripted expert in DIRECT mode."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from expert.scripted_expert import ScriptedExpert  # noqa: E402


def run_episode(env: PandaPickPlaceEnv, expert: ScriptedExpert):
    observation = env.reset()
    expert.reset(env)
    while not expert.is_done:
        action = expert.act(observation)
        observation, _ = env.step(action)
    return expert.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    successes = 0
    failures: Counter[str] = Counter()
    with PandaPickPlaceEnv(mode="direct") as env:
        expert = ScriptedExpert()
        for episode in range(1, args.episodes + 1):
            result = run_episode(env, expert)
            status = "SUCCESS" if result.success else "FAILURE"
            print(f"Episode {episode:02d}: {status} | steps={result.steps}")
            if result.success:
                successes += 1
            else:
                failures[result.failure_reason or "unknown"] += 1

    print(f"\nSuccess rate: {successes} / {args.episodes}")
    print(f"Failure reasons: {dict(failures) if failures else 'none'}")
    if successes != args.episodes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
