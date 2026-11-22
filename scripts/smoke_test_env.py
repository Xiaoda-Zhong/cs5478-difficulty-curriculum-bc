#!/usr/bin/env python3
"""Run lightweight end-to-end checks in headless PyBullet mode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.panda_pick_place import PandaPickPlaceEnv  # noqa: E402


def main() -> None:
    env = PandaPickPlaceEnv(mode="direct")
    try:
        observations = [env.reset() for _ in range(20)]
        assert all(obs.shape == (env.observation_dim,) for obs in observations)
        assert all(np.all(np.isfinite(obs)) for obs in observations)

        ee_before = env.get_ee_position()
        object_position = env.get_object_position()
        goal_position = env.get_goal_position()
        assert all(
            np.all(np.isfinite(position))
            for position in (ee_before, object_position, goal_position)
        )

        _, motion_info = env.step([0.02, 0.0, 0.0, -1.0])
        ee_after = env.get_ee_position()
        motion_distance = float(np.linalg.norm(ee_after - ee_before))
        assert motion_distance > 1e-4, "End effector did not respond to a small action"

        env.step([0.0, 0.0, 0.0, 1.0])
        env.step([0.0, 0.0, 0.0, -1.0])

        max_delta = float(env.config["control"]["max_cartesian_delta"])
        _, clipping_info = env.step([10.0, -10.0, 10.0, 5.0])
        assert np.all(np.abs(clipping_info["applied_action"][:3]) <= max_delta)
        bounds = np.asarray(env.config["control"]["workspace_bounds"], dtype=float)
        target = clipping_info["ee_target_position"]
        assert np.all(target >= bounds[:, 0]) and np.all(target <= bounds[:, 1])

        print("DIRECT-mode smoke test passed")
        print(f"  repeated resets: 20")
        print(f"  observation shape: {observations[0].shape}")
        print(f"  EE motion from 2 cm command: {motion_distance:.4f} m")
        print(f"  clipped Cartesian action: {clipping_info['applied_action'][:3]}")
        print(f"  success at current state: {motion_info['is_success']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
