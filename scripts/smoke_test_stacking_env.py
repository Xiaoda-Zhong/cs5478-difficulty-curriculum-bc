#!/usr/bin/env python3
"""Headless smoke checks for the S1 Panda stacking environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.panda_stacking import PandaStackingEnv  # noqa: E402


def main() -> None:
    with PandaStackingEnv(mode="direct") as env:
        reference = env.reset()
        assert reference.shape == (29,) and np.all(np.isfinite(reference))
        assert env.source_cube_id is not None and env.support_cube_id is not None
        assert p.getDynamicsInfo(env.source_cube_id, -1, physicsClientId=env.client_id)[0] > 0
        assert p.getDynamicsInfo(env.support_cube_id, -1, physicsClientId=env.client_id)[0] > 0
        assert not p.getClosestPoints(env.source_cube_id, env.support_cube_id, 0.0,
                                      physicsClientId=env.client_id)
        for _ in range(100):
            np.testing.assert_allclose(env.reset(), reference, atol=1e-6)
        before = env.get_ee_position()
        observation, info = env.step([0.02, 0.0, 0.0, -1.0])
        assert observation.shape == (29,) and np.all(np.isfinite(observation))
        assert 1e-4 < np.linalg.norm(env.get_ee_position() - before) < 0.08
        maximum = float(env.config["control"]["max_cartesian_delta"])
        _, clipped = env.step([100.0, -100.0, 100.0, 1.0])
        np.testing.assert_allclose(clipped["applied_action"],
                                   [maximum, -maximum, maximum, 1.0])
        bounds = np.asarray(env.config["control"]["workspace_bounds"])
        assert np.all(clipped["ee_target_position"] >= bounds[:, 0])
        assert np.all(clipped["ee_target_position"] <= bounds[:, 1])
        assert isinstance(info["stack_diagnostics"], dict)
    print("Stacking environment smoke test: PASSED (100 deterministic resets)")


if __name__ == "__main__":
    main()
