#!/usr/bin/env python3
"""Run one scripted-expert trajectory in the PyBullet GUI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402
from expert.scripted_expert import ScriptedExpert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-delay", type=float, default=0.02)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    args = parser.parse_args()

    with PandaPickPlaceEnv(mode="gui") as env:
        observation = env.reset()
        expert = ScriptedExpert()
        expert.reset(env)
        previous_phase = expert.current_phase
        print(previous_phase.value)

        while not expert.is_done:
            action = expert.act(observation)
            if expert.current_phase != previous_phase:
                print(f"{previous_phase.value} -> {expert.current_phase.value}")
                previous_phase = expert.current_phase
            observation, _ = env.step(action)
            time.sleep(max(0.0, args.control_delay))

        result = expert.result()
        print("SUCCESS" if result.success else f"FAILURE: {result.failure_reason}")
        print(f"steps={result.steps}, final_object_position={result.final_object_position}")
        time.sleep(max(0.0, args.hold_seconds))


if __name__ == "__main__":
    main()
