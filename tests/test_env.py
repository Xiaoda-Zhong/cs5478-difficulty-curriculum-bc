"""Tests for the deterministic Milestone 1 environment."""

from __future__ import annotations

import numpy as np
import pytest

from envs.panda_pick_place import PandaPickPlaceEnv


@pytest.fixture
def env() -> PandaPickPlaceEnv:
    environment = PandaPickPlaceEnv(mode="direct")
    yield environment
    environment.close()


def test_reset_is_deterministic(env: PandaPickPlaceEnv) -> None:
    first = env.reset()
    env.step([0.02, 0.0, 0.0, 1.0])
    second = env.reset()
    np.testing.assert_allclose(second, first, atol=1e-6)


def test_observation_is_ten_dimensional_and_finite(env: PandaPickPlaceEnv) -> None:
    observation = env.reset()
    assert observation.shape == (10,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))


def test_cube_and_goal_positions_are_fixed(env: PandaPickPlaceEnv) -> None:
    env.reset()
    expected_cube = np.asarray(env.config["task"]["cube_position"], dtype=float)
    expected_goal = np.asarray(env.config["task"]["goal_position"], dtype=float)
    np.testing.assert_allclose(env.get_object_position(), expected_cube, atol=2e-3)
    np.testing.assert_allclose(env.get_goal_position(), expected_goal, atol=1e-12)


def test_action_is_clipped_and_target_stays_in_workspace(env: PandaPickPlaceEnv) -> None:
    max_delta = float(env.config["control"]["max_cartesian_delta"])
    _, info = env.step([100.0, -100.0, 100.0, 100.0])
    np.testing.assert_allclose(info["applied_action"], [max_delta, -max_delta, max_delta, 1.0])

    bounds = np.asarray(env.config["control"]["workspace_bounds"], dtype=float)
    assert np.all(info["ee_target_position"] >= bounds[:, 0])
    assert np.all(info["ee_target_position"] <= bounds[:, 1])


@pytest.mark.parametrize(
    "bad_action",
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, np.nan]),
)
def test_invalid_action_is_rejected(env: PandaPickPlaceEnv, bad_action: list[float]) -> None:
    with pytest.raises(ValueError):
        env.step(bad_action)


def test_small_cartesian_action_moves_end_effector(env: PandaPickPlaceEnv) -> None:
    before = env.get_ee_position()
    env.step([0.02, 0.0, 0.0, -1.0])
    after = env.get_ee_position()
    displacement = float(np.linalg.norm(after - before))
    assert 1e-4 < displacement < 0.08


def test_repeated_reset_is_stable(env: PandaPickPlaceEnv) -> None:
    reference = env.reset()
    for _ in range(20):
        observation = env.reset()
        assert observation.shape == (10,)
        assert np.all(np.isfinite(observation))
        np.testing.assert_allclose(observation, reference, atol=1e-6)


def test_close_is_idempotent() -> None:
    environment = PandaPickPlaceEnv(mode="direct")
    environment.close()
    environment.close()
    with pytest.raises(RuntimeError):
        environment.get_observation()
