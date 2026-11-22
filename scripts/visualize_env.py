#!/usr/bin/env python3
"""Open the deterministic environment in PyBullet GUI for visual inspection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-motion",
        action="store_true",
        help="apply a few small Cartesian movements before holding the GUI open",
    )
    args = parser.parse_args()

    env = PandaPickPlaceEnv(mode="gui")
    try:
        observation = env.reset()
        print(f"Environment ready. Observation shape: {observation.shape}")
        print("Close the PyBullet window or press Ctrl+C to exit.")

        if args.demo_motion:
            for action in ([0.02, 0.0, 0.0, -1.0], [0.0, 0.02, 0.0, -1.0]):
                env.step(action)
                time.sleep(0.4)

        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
