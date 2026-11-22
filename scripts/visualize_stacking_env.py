#!/usr/bin/env python3
"""Open the S1 stacking scene in PyBullet GUI without autonomous motion."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_stacking import PandaStackingEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="Close after N seconds; 0 waits for Ctrl-C")
    args = parser.parse_args()
    with PandaStackingEnv(mode="gui") as env:
        print("Stacking scene ready. Red=source, blue=support. Ctrl-C to close.")
        print(env.get_stack_diagnostics())
        started = time.monotonic()
        try:
            while args.seconds <= 0 or time.monotonic() - started < args.seconds:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
