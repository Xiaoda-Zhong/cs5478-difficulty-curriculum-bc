"""Tests for the deterministic finite-state-machine expert."""

from __future__ import annotations

import numpy as np
import pytest

from envs.panda_pick_place import PandaPickPlaceEnv
from expert.scripted_expert import ExpertPhase, ScriptedExpert


@pytest.fixture
def env() -> PandaPickPlaceEnv:
    environment = PandaPickPlaceEnv(mode="direct")
    yield environment
    environment.close()


def run_episode(env: PandaPickPlaceEnv, expert: ScriptedExpert):
    observation = env.reset()
    expert.reset(env)
    while not expert.is_done:
        observation, _ = env.step(expert.act(observation))
    return expert.result()


def test_expert_starts_in_raise_phase(env: PandaPickPlaceEnv) -> None:
    expert = ScriptedExpert()
    expert.reset(env)
    assert expert.current_phase == ExpertPhase.RAISE
    assert not expert.is_done


def test_action_is_finite_four_vector(env: PandaPickPlaceEnv) -> None:
    expert = ScriptedExpert()
    expert.reset(env)
    action = expert.act(env.get_observation())
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))


def test_cartesian_step_respects_configured_size(env: PandaPickPlaceEnv) -> None:
    expert = ScriptedExpert()
    expert.reset(env)
    action = expert.act(env.get_observation())
    assert np.linalg.norm(action[:3]) <= env.config["expert"]["step_size"] + 1e-12


def test_fsm_progresses_and_fixed_task_succeeds(env: PandaPickPlaceEnv) -> None:
    expert = ScriptedExpert()
    result = run_episode(env, expert)
    assert result.success, result.failure_reason
    assert expert.current_phase == ExpertPhase.DONE
    assert expert.phase_history == list(ExpertPhase)
    assert env.is_success()


def test_repeated_episode_resets_fsm_and_is_deterministic(
    env: PandaPickPlaceEnv,
) -> None:
    expert = ScriptedExpert()
    first = run_episode(env, expert)
    first_history = expert.phase_history.copy()
    second = run_episode(env, expert)
    assert first.success and second.success
    assert first.steps == second.steps
    assert expert.phase_history == first_history
    np.testing.assert_allclose(
        second.final_object_position, first.final_object_position, atol=1e-6
    )
