"""Deterministic finite-state-machine expert for Panda pick-and-place."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from envs.panda_pick_place import PandaPickPlaceEnv
from expert.waypoints import ExpertWaypoints, build_waypoints


class ExpertPhase(str, Enum):
    """Ordered phases of one scripted pick-and-place trajectory."""

    RAISE = "RAISE"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    DONE = "DONE"


@dataclass(frozen=True)
class ExpertResult:
    """Terminal status for one expert episode."""

    success: bool
    failure_reason: str | None
    steps: int
    final_object_position: np.ndarray


class ScriptedExpert:
    """Geometry-based closed-loop controller that emits environment actions."""

    def __init__(self) -> None:
        self.current_phase = ExpertPhase.DONE
        self.is_done = True
        self.success = False
        self.failure_reason: str | None = "not_reset"
        self.steps = 0
        self.phase_history: list[ExpertPhase] = []
        self.waypoints: ExpertWaypoints | None = None
        self._config: Mapping[str, Any] = {}
        self._workspace_bounds = np.empty((3, 2), dtype=np.float64)
        self._initial_object_position = np.zeros(3, dtype=np.float64)
        self._goal_position = np.zeros(3, dtype=np.float64)
        self._final_object_position = np.zeros(3, dtype=np.float64)
        self._grasp_counter = 0
        self._release_counter = 0

    def reset(self, env: PandaPickPlaceEnv) -> None:
        """Reset FSM state and derive waypoints from the current environment."""
        observation = env.get_observation()
        self._config = env.config["expert"]
        self._workspace_bounds = np.asarray(
            env.config["control"]["workspace_bounds"], dtype=np.float64
        )
        self._initial_object_position = observation[3:6].astype(np.float64)
        self._goal_position = observation[6:9].astype(np.float64)
        self._final_object_position = self._initial_object_position.copy()
        self._target_radius = env.target_radius
        self._target_height_tolerance = float(
            env.config["task"]["target_height_tolerance"]
        )
        self.waypoints = build_waypoints(
            observation[:3],
            self._initial_object_position,
            self._goal_position,
            safe_height=float(self._config["safe_height"]),
            grasp_height_offset=float(self._config["grasp_height_offset"]),
            place_height_offset=float(self._config["place_height_offset"]),
        )
        self.current_phase = ExpertPhase.RAISE
        self.is_done = False
        self.success = False
        self.failure_reason = None
        self.steps = 0
        self.phase_history = [self.current_phase]
        self._grasp_counter = 0
        self._release_counter = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return the next closed-loop ``[dx, dy, dz, gripper]`` action."""
        if self.is_done:
            raise RuntimeError("Expert is done; call reset() before requesting actions")

        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (10,) or not np.all(np.isfinite(obs)):
            return self._fail("invalid_state")
        self._final_object_position = obs[3:6].copy()

        if self.steps >= int(self._config["max_episode_steps"]):
            return self._fail("timeout")
        if self._object_outside_workspace(self._final_object_position):
            return self._fail("object_left_workspace")
        if self.current_phase == ExpertPhase.TRANSPORT and self._object_is_dropped(obs):
            return self._fail("dropped_object")

        while True:
            if self.current_phase == ExpertPhase.RAISE:
                action = self._movement_action(obs[:3], self._waypoint("safe_current"), -1.0)
                if action is not None:
                    return self._emit(action)
                self._transition(ExpertPhase.APPROACH)

            elif self.current_phase == ExpertPhase.APPROACH:
                action = self._movement_action(obs[:3], self._waypoint("above_object"), -1.0)
                if action is not None:
                    return self._emit(action)
                self._transition(ExpertPhase.DESCEND)

            elif self.current_phase == ExpertPhase.DESCEND:
                action = self._movement_action(obs[:3], self._waypoint("grasp"), -1.0)
                if action is not None:
                    return self._emit(action)
                self._transition(ExpertPhase.GRASP)

            elif self.current_phase == ExpertPhase.GRASP:
                if self._grasp_counter < int(self._config["grasp_settle_steps"]):
                    self._grasp_counter += 1
                    return self._emit(self._stationary_action(close=True))
                self._transition(ExpertPhase.LIFT)

            elif self.current_phase == ExpertPhase.LIFT:
                action = self._movement_action(obs[:3], self._waypoint("lift"), 1.0)
                if action is not None:
                    return self._emit(action)
                if not self._grasp_verified(obs):
                    return self._fail("failed_grasp")
                self._transition(ExpertPhase.TRANSPORT)

            elif self.current_phase == ExpertPhase.TRANSPORT:
                action = self._movement_action(obs[:3], self._waypoint("above_goal"), 1.0)
                if action is not None:
                    return self._emit(action)
                self._transition(ExpertPhase.PLACE)

            elif self.current_phase == ExpertPhase.PLACE:
                action = self._movement_action(obs[:3], self._waypoint("place"), 1.0)
                if action is not None:
                    return self._emit(action)
                self._transition(ExpertPhase.RELEASE)

            elif self.current_phase == ExpertPhase.RELEASE:
                if self._release_counter < int(self._config["release_settle_steps"]):
                    self._release_counter += 1
                    return self._emit(self._stationary_action(close=False))
                self._transition(ExpertPhase.RETREAT)

            elif self.current_phase == ExpertPhase.RETREAT:
                action = self._movement_action(obs[:3], self._waypoint("retreat"), -1.0)
                if action is not None:
                    return self._emit(action)
                return self._finish(obs)

            else:
                return self._fail("invalid_phase")

    def result(self) -> ExpertResult:
        """Return current or terminal episode status."""
        return ExpertResult(
            success=self.success,
            failure_reason=self.failure_reason,
            steps=self.steps,
            final_object_position=self._final_object_position.copy(),
        )

    def _movement_action(
        self, current: np.ndarray, target: np.ndarray, gripper_command: float
    ) -> np.ndarray | None:
        displacement = target - current
        distance = float(np.linalg.norm(displacement))
        if distance <= float(self._config["position_tolerance"]):
            return None
        magnitude = min(distance, float(self._config["step_size"]))
        delta = displacement / distance * magnitude
        return np.concatenate([delta, np.asarray([gripper_command])])

    @staticmethod
    def _stationary_action(*, close: bool) -> np.ndarray:
        return np.asarray([0.0, 0.0, 0.0, 1.0 if close else -1.0], dtype=np.float64)

    def _grasp_verified(self, observation: np.ndarray) -> bool:
        object_position = observation[3:6]
        ee_position = observation[:3]
        height_gain = object_position[2] - self._initial_object_position[2]
        distance = float(np.linalg.norm(object_position - ee_position))
        return bool(
            height_gain >= float(self._config["min_lift_height"])
            and distance <= float(self._config["max_grasp_distance"])
        )

    def _object_is_dropped(self, observation: np.ndarray) -> bool:
        height_gain = observation[5] - self._initial_object_position[2]
        return bool(height_gain < float(self._config["min_lift_height"]))

    def _object_outside_workspace(self, object_position: np.ndarray) -> bool:
        margin = 0.10
        lower = self._workspace_bounds[:, 0] - margin
        upper = self._workspace_bounds[:, 1] + margin
        return bool(np.any(object_position < lower) or np.any(object_position > upper))

    def _finish(self, observation: np.ndarray) -> np.ndarray:
        object_position = observation[3:6]
        xy_error = float(np.linalg.norm(object_position[:2] - self._goal_position[:2]))
        height_error = abs(float(object_position[2] - self._goal_position[2]))
        self.success = bool(
            xy_error < self._target_radius
            and height_error < self._target_height_tolerance
        )
        self.failure_reason = None if self.success else "placement_failed"
        self.is_done = True
        self._transition(ExpertPhase.DONE)
        return self._emit(self._stationary_action(close=False))

    def _fail(self, reason: str) -> np.ndarray:
        self.success = False
        self.failure_reason = reason
        self.is_done = True
        self._transition(ExpertPhase.DONE)
        return self._stationary_action(close=False)

    def _emit(self, action: np.ndarray) -> np.ndarray:
        self.steps += 1
        return np.asarray(action, dtype=np.float64)

    def _transition(self, phase: ExpertPhase) -> None:
        self.current_phase = phase
        self.phase_history.append(phase)

    def _waypoint(self, name: str) -> np.ndarray:
        if self.waypoints is None:
            raise RuntimeError("Expert has not been reset")
        return getattr(self.waypoints, name)
