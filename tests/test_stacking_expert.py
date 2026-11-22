"""Tests for the deterministic physical stacking expert."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from envs.panda_stacking import PandaStackingEnv
from expert.stacking_expert import (
    ScriptedStackingExpert,
    StackingPhase,
    rollout_stacking_expert,
)
from expert.stacking_waypoints import build_stacking_waypoints


@pytest.fixture
def env() -> PandaStackingEnv:
    environment = PandaStackingEnv(mode="direct")
    yield environment
    environment.close()


def test_expert_initialization_and_starting_phase(env: PandaStackingEnv) -> None:
    expert = ScriptedStackingExpert()
    assert expert.is_done and expert.failure_reason == "not_reset"
    expert.reset(env)
    assert expert.current_phase == StackingPhase.RAISE
    assert not expert.is_done and expert.steps == 0


def test_action_is_finite_bounded_four_vector(env: PandaStackingEnv) -> None:
    expert = ScriptedStackingExpert()
    expert.reset(env)
    action = expert.act(env.get_observation())
    assert action.shape == (4,) and np.all(np.isfinite(action))
    assert np.linalg.norm(action[:3]) <= env.config["expert"]["step_size"] + 1e-12
    assert action[3] == -1.0


def test_waypoint_geometry_uses_cube_edge_and_grasp_offset(env: PandaStackingEnv) -> None:
    config = env.config["expert"]
    waypoints = build_stacking_waypoints(
        env.get_ee_position(), env.get_source_position(), env.get_support_position(),
        cube_edge_length=env.cube_edge_length, safe_height=config["safe_height"],
        grasp_height_offset=config["grasp_height_offset"],
        transport_clearance=config["transport_clearance"],
        precontact_clearance=config["precontact_clearance"],
        contact_descent_margin=config["contact_descent_margin"],
    )
    expected_contact = (env.get_support_position()[2] + env.cube_edge_length
                        + config["grasp_height_offset"])
    assert waypoints.grasp[2] == pytest.approx(
        env.get_source_position()[2] + config["grasp_height_offset"])
    assert waypoints.contact[2] == pytest.approx(expected_contact - config["contact_descent_margin"])
    np.testing.assert_allclose(waypoints.above_support[:2], env.get_support_position()[:2])


def test_invalid_observation_failure(env: PandaStackingEnv) -> None:
    expert = ScriptedStackingExpert()
    expert.reset(env)
    action = expert.act(np.zeros(28))
    assert action.shape == (4,) and expert.is_done
    assert expert.failure_reason == "invalid_observation"


def test_timeout_failure_classification() -> None:
    config = PandaStackingEnv._load_config(None, None)
    config = deepcopy(config)
    config["expert"]["max_episode_steps"] = 1
    with PandaStackingEnv(config=config) as environment:
        expert = ScriptedStackingExpert()
        expert.reset(environment)
        observation = environment.get_observation()
        observation, _ = environment.step(expert.act(observation))
        expert.act(observation)
        assert expert.is_done and expert.failure_reason == "expert_timeout"


def test_complete_fixed_task_and_phase_progression(env: PandaStackingEnv) -> None:
    trace = rollout_stacking_expert(env, ScriptedStackingExpert())
    result = trace["result"]
    assert result.success, result.failure_reason
    assert result.phase_history == tuple(phase.value for phase in StackingPhase)
    assert env.is_success() and env.is_stack_physically_stable()
    assert result.final_diagnostics["source_support_contact"]
    assert result.final_diagnostics["gripper_retreat_clearance"] >= 0.10
    assert result.maximum_support_displacement < 0.01
    assert set(trace) >= {"observations", "actions", "ee_positions", "source_positions",
                          "support_positions", "source_orientations", "support_orientations",
                          "phases", "stacking_diagnostics", "success", "failure_reason"}


def test_grasp_and_lift_are_physically_verified(env: PandaStackingEnv) -> None:
    trace = rollout_stacking_expert(env, ScriptedStackingExpert())
    phases = trace["phases"].tolist()
    lift_indices = [i for i, phase in enumerate(phases) if phase == "LIFT"]
    assert lift_indices
    source = trace["source_positions"]
    assert source[lift_indices[-1], 2] - source[0, 2] >= env.config["expert"]["min_lift_height"]


def test_repeated_episodes_are_deterministic(env: PandaStackingEnv) -> None:
    first = rollout_stacking_expert(env, ScriptedStackingExpert())
    env.reset()
    second = rollout_stacking_expert(env, ScriptedStackingExpert())
    assert first["success"] and second["success"]
    assert first["result"].steps == second["result"].steps
    assert first["result"].phase_history == second["result"].phase_history
    np.testing.assert_allclose(first["source_positions"], second["source_positions"], atol=1e-6)
