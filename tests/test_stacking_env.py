"""S1 tests for deterministic physics and persistent stacking success."""

from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from envs.panda_stacking import PandaStackingEnv


@pytest.fixture
def env() -> PandaStackingEnv:
    environment = PandaStackingEnv(mode="direct")
    yield environment
    environment.close()


def _place(env: PandaStackingEnv, *, source_offset=(0.0, 0.0, 0.0),
           source_velocity=(0.0, 0.0, 0.0),
           support_orientation=(0.0, 0.0, 0.0, 1.0)) -> None:
    support = np.array([0.60, 0.18, 0.025])
    source = support + np.array([0.0, 0.0, env.cube_edge_length]) + source_offset
    p.resetBasePositionAndOrientation(env.support_cube_id, support, support_orientation,
                                      physicsClientId=env.client_id)
    p.resetBaseVelocity(env.support_cube_id, [0.0] * 3, [0.0] * 3,
                        physicsClientId=env.client_id)
    p.resetBasePositionAndOrientation(env.source_cube_id, source, [0, 0, 0, 1],
                                      physicsClientId=env.client_id)
    p.resetBaseVelocity(env.source_cube_id, source_velocity, [0.0] * 3,
                        physicsClientId=env.client_id)


def _settle(env: PandaStackingEnv, actions: int) -> None:
    for _ in range(actions):
        env.step([0.0, 0.0, 0.0, -1.0])


def test_observation_action_and_deterministic_reset(env: PandaStackingEnv) -> None:
    first = env.reset()
    assert first.shape == (29,) and first.dtype == np.float32
    assert np.all(np.isfinite(first))
    for _ in range(20):
        np.testing.assert_allclose(env.reset(), first, atol=1e-6)
    with pytest.raises(ValueError):
        env.step([0.0, 0.0, 0.0])


def test_explicit_numpy_reset_and_invalid_overlap(env: PandaStackingEnv) -> None:
    source = np.array([0.50, -0.15, 0.025])
    support = np.array([0.60, 0.15, 0.025])
    assert env.reset(source, support).shape == (29,)
    with pytest.raises(ValueError, match="overlap"):
        env.reset(source, source)


def test_two_cubes_are_dynamic_and_initially_separate(env: PandaStackingEnv) -> None:
    for body in (env.source_cube_id, env.support_cube_id):
        assert p.getDynamicsInfo(body, -1, physicsClientId=env.client_id)[0] > 0
    assert not p.getClosestPoints(env.source_cube_id, env.support_cube_id, 0.0,
                                  physicsClientId=env.client_id)


def test_aligned_stack_is_instantaneously_physical_after_settling(env: PandaStackingEnv) -> None:
    _place(env)
    _settle(env, 5)
    assert env.is_stack_geometrically_aligned()
    assert env.is_stack_physically_stable()
    assert not env.is_success()


@pytest.mark.parametrize("offset", [(0.04, 0.0, 0.0), (0.0, 0.0, 0.03)])
def test_beside_or_above_without_contact_is_not_stable(env: PandaStackingEnv, offset) -> None:
    _place(env, source_offset=offset)
    assert not env.is_stack_physically_stable()


def test_source_on_table_is_not_stack(env: PandaStackingEnv) -> None:
    p.resetBasePositionAndOrientation(env.support_cube_id, [0.60, 0.18, 0.025],
                                      [0, 0, 0, 1], physicsClientId=env.client_id)
    p.resetBasePositionAndOrientation(env.source_cube_id, [0.67, 0.18, 0.025],
                                      [0, 0, 0, 1], physicsClientId=env.client_id)
    _settle(env, 3)
    assert env.get_stack_diagnostics()["source_table_contact"]
    assert not env.is_stack_physically_stable()


def test_temporary_contact_and_counter_reset(env: PandaStackingEnv) -> None:
    _place(env)
    _settle(env, 5)
    assert env.stable_steps > 0
    _place(env, source_offset=(0.04, 0.0, 0.0))
    env.step([0.0, 0.0, 0.0, -1.0])
    assert env.stable_steps == 0 and not env.is_success()


def test_source_moving_too_fast_is_not_stable(env: PandaStackingEnv) -> None:
    _place(env, source_velocity=(0.1, 0.0, 0.0))
    assert not env.is_stack_physically_stable()


def test_tipped_support_is_not_stable(env: PandaStackingEnv) -> None:
    _place(env, support_orientation=p.getQuaternionFromEuler([np.pi / 4, 0, 0]))
    assert not env.is_stack_physically_stable()


def test_closed_gripper_rejects_stack(env: PandaStackingEnv) -> None:
    _place(env)
    for index in env.finger_joint_indices:
        p.resetJointState(env.robot_id, index, 0.0, physicsClientId=env.client_id)
    assert not env.is_stack_physically_stable()


def test_insufficient_gripper_retreat_rejects_stack(env: PandaStackingEnv,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _place(env)
    monkeypatch.setattr(env, "get_ee_position", lambda: env.get_source_position().copy())
    assert not env.is_stack_physically_stable()


def test_persistent_stack_requires_full_window(env: PandaStackingEnv) -> None:
    _place(env)
    required = int(env.config["success"]["settling_control_steps"])
    _settle(env, required - 1)
    assert env.stable_steps == required - 1 and not env.is_success()
    _settle(env, 1)
    assert env.stable_steps == required and env.is_success()
