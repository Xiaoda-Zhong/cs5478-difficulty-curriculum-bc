"""Deterministic finite-state-machine expert for physical cube stacking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from envs.panda_stacking import PandaStackingEnv
from expert.stacking_waypoints import StackingWaypoints, build_stacking_waypoints


class StackingPhase(str, Enum):
    RAISE = "RAISE"
    APPROACH_SOURCE = "APPROACH_SOURCE"
    DESCEND_SOURCE = "DESCEND_SOURCE"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSPORT_ABOVE_SUPPORT = "TRANSPORT_ABOVE_SUPPORT"
    ALIGN = "ALIGN"
    LOWER_TO_PRECONTACT = "LOWER_TO_PRECONTACT"
    LOWER_TO_CONTACT = "LOWER_TO_CONTACT"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    VERIFY_STACK = "VERIFY_STACK"
    DONE = "DONE"


@dataclass(frozen=True)
class StackingExpertResult:
    success: bool
    failure_reason: str | None
    steps: int
    phase_history: tuple[str, ...]
    final_source_position: np.ndarray
    final_support_position: np.ndarray
    maximum_support_displacement: float
    final_diagnostics: dict[str, Any]


class ScriptedStackingExpert:
    """Closed-loop Cartesian expert using only normal environment actions."""

    def __init__(self) -> None:
        self.current_phase = StackingPhase.DONE
        self.is_done = True
        self.success = False
        self.failure_reason: str | None = "not_reset"
        self.step_count = 0
        self.phase_step_count = 0
        self.phase_history: list[StackingPhase] = []
        self.waypoints: StackingWaypoints | None = None
        self.env: PandaStackingEnv | None = None
        self._config: dict[str, Any] = {}
        self._initial_source = np.zeros(3)
        self._initial_support = np.zeros(3)
        self._maximum_support_displacement = 0.0
        self._grasp_counter = 0
        self._release_counter = 0

    def reset(self, env: PandaStackingEnv) -> None:
        self.env = env
        self._config = dict(env.config["expert"])
        self._initial_source = env.get_source_position()
        self._initial_support = env.get_support_position()
        self._maximum_support_displacement = 0.0
        self.waypoints = self._build_waypoints()
        self.current_phase = StackingPhase.RAISE
        self.is_done = False
        self.success = False
        self.failure_reason = None
        self.step_count = 0
        self.phase_step_count = 0
        self.phase_history = [self.current_phase]
        self._grasp_counter = 0
        self._release_counter = 0

    @property
    def steps(self) -> int:
        return self.step_count

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.is_done or self.env is None:
            raise RuntimeError("Expert is done; call reset() before requesting actions")
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (29,) or not np.all(np.isfinite(obs)):
            return self._fail("invalid_observation")
        if self.step_count >= int(self._config["max_episode_steps"]):
            return self._fail("expert_timeout")
        failure = self._physical_failure()
        if failure is not None:
            return self._fail(failure)
        if self.phase_step_count >= self._phase_timeout():
            return self._fail(self._timeout_reason())

        while True:
            ee = self.env.get_ee_position()
            if self.current_phase == StackingPhase.RAISE:
                action = self._movement(ee, self._waypoint("safe_current"), close=False)
                if action is not None:
                    return self._emit(action)
                self._transition(StackingPhase.APPROACH_SOURCE)

            elif self.current_phase == StackingPhase.APPROACH_SOURCE:
                action = self._movement(ee, self._waypoint("above_source"), close=False)
                if action is not None:
                    return self._emit(action)
                self._transition(StackingPhase.DESCEND_SOURCE)

            elif self.current_phase == StackingPhase.DESCEND_SOURCE:
                action = self._movement(ee, self._waypoint("grasp"), close=False)
                if action is not None:
                    return self._emit(action)
                self._transition(StackingPhase.GRASP)

            elif self.current_phase == StackingPhase.GRASP:
                if self._grasp_counter < int(self._config["grasp_settle_steps"]):
                    self._grasp_counter += 1
                    return self._emit(self._stationary(close=True))
                self._transition(StackingPhase.LIFT)

            elif self.current_phase == StackingPhase.LIFT:
                action = self._movement(ee, self._waypoint("lift"), close=True)
                if action is not None:
                    return self._emit(action)
                if not self._grasp_verified():
                    return self._fail("failed_grasp_or_lift")
                self._transition(StackingPhase.TRANSPORT_ABOVE_SUPPORT)

            elif self.current_phase == StackingPhase.TRANSPORT_ABOVE_SUPPORT:
                action = self._movement(ee, self._dynamic_waypoint("above_support"), close=True)
                if action is not None:
                    return self._emit(action)
                self._transition(StackingPhase.ALIGN)

            elif self.current_phase == StackingPhase.ALIGN:
                xy_error = self.env.get_support_position()[:2] - self.env.get_source_position()[:2]
                if np.linalg.norm(xy_error) > float(self._config["alignment_tolerance"]):
                    target = ee.copy()
                    target[:2] += xy_error
                    action = self._movement(ee, target, close=True)
                    if action is not None:
                        return self._emit(action)
                self._transition(StackingPhase.LOWER_TO_PRECONTACT)

            elif self.current_phase == StackingPhase.LOWER_TO_PRECONTACT:
                target = self._dynamic_waypoint("precontact")
                xy_error = self.env.get_support_position()[:2] - self.env.get_source_position()[:2]
                target[:2] = ee[:2] + xy_error
                action = self._movement(ee, target, close=True)
                if action is not None:
                    return self._emit(action)
                if self._source_support_xy_error() > float(self._config["alignment_tolerance"]):
                    return self._fail("failed_stacking_alignment")
                self._transition(StackingPhase.LOWER_TO_CONTACT)

            elif self.current_phase == StackingPhase.LOWER_TO_CONTACT:
                diagnostics = self.env.get_stack_diagnostics()
                if diagnostics["source_support_contact"]:
                    if not self.env.is_stack_geometrically_aligned():
                        return self._fail("failed_placement")
                    self._transition(StackingPhase.RELEASE)
                    continue
                target = self._dynamic_waypoint("contact")
                target[:2] = (
                    ee[:2]
                    + self.env.get_support_position()[:2]
                    - self.env.get_source_position()[:2]
                )
                action = self._movement(
                    ee, target, close=True,
                    step_size=float(self._config["near_contact_step_size"]),
                )
                if action is None:
                    return self._fail("failed_to_establish_contact")
                return self._emit(action)

            elif self.current_phase == StackingPhase.RELEASE:
                if self._release_counter < int(self._config["release_settle_steps"]):
                    self._release_counter += 1
                    return self._emit(self._stationary(close=False))
                d = self.env.get_stack_diagnostics()
                if not d["source_support_contact"] or d["source_table_contact"]:
                    return self._fail("unstable_after_release")
                self._transition(StackingPhase.RETREAT)

            elif self.current_phase == StackingPhase.RETREAT:
                action = self._movement(ee, self._dynamic_waypoint("retreat"), close=False)
                if action is not None:
                    return self._emit(action)
                self._transition(StackingPhase.VERIFY_STACK)

            elif self.current_phase == StackingPhase.VERIFY_STACK:
                if self.env.is_success():
                    return self._finish()
                return self._emit(self._stationary(close=False))

            else:
                return self._fail("invalid_phase")

    def result(self) -> StackingExpertResult:
        if self.env is None:
            raise RuntimeError("Expert has not been reset")
        return StackingExpertResult(
            success=self.success,
            failure_reason=self.failure_reason,
            steps=self.step_count,
            phase_history=tuple(phase.value for phase in self.phase_history),
            final_source_position=self.env.get_source_position(),
            final_support_position=self.env.get_support_position(),
            maximum_support_displacement=self._maximum_support_displacement,
            final_diagnostics=self.env.get_stack_diagnostics(),
        )

    def _build_waypoints(self) -> StackingWaypoints:
        assert self.env is not None
        return build_stacking_waypoints(
            self.env.get_ee_position(), self.env.get_source_position(),
            self.env.get_support_position(), cube_edge_length=self.env.cube_edge_length,
            safe_height=float(self._config["safe_height"]),
            grasp_height_offset=float(self._config["grasp_height_offset"]),
            transport_clearance=float(self._config["transport_clearance"]),
            precontact_clearance=float(self._config["precontact_clearance"]),
            contact_descent_margin=float(self._config["contact_descent_margin"]),
        )

    def _dynamic_waypoint(self, name: str) -> np.ndarray:
        refreshed = self._build_waypoints()
        return getattr(refreshed, name)

    def _waypoint(self, name: str) -> np.ndarray:
        if self.waypoints is None:
            raise RuntimeError("Expert has not been reset")
        return getattr(self.waypoints, name)

    def _movement(self, current: np.ndarray, target: np.ndarray, *, close: bool,
                  step_size: float | None = None) -> np.ndarray | None:
        displacement = target - current
        distance = float(np.linalg.norm(displacement))
        if distance <= float(self._config["position_tolerance"]):
            return None
        maximum = float(step_size if step_size is not None else self._config["step_size"])
        delta = displacement / distance * min(distance, maximum)
        return np.concatenate([delta, [1.0 if close else -1.0]])

    @staticmethod
    def _stationary(*, close: bool) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0 if close else -1.0])

    def _grasp_verified(self) -> bool:
        assert self.env is not None
        source = self.env.get_source_position()
        height_gain = source[2] - self._initial_source[2]
        distance = float(np.linalg.norm(source - self.env.get_ee_position()))
        return bool(height_gain >= float(self._config["min_lift_height"])
                    and distance <= float(self._config["max_grasp_distance"]))

    def _source_support_xy_error(self) -> float:
        assert self.env is not None
        return float(np.linalg.norm(
            self.env.get_source_position()[:2] - self.env.get_support_position()[:2]))

    def _physical_failure(self) -> str | None:
        assert self.env is not None
        source = self.env.get_source_position()
        support = self.env.get_support_position()
        displacement = float(np.linalg.norm(support - self._initial_support))
        self._maximum_support_displacement = max(self._maximum_support_displacement, displacement)
        if source[2] < -0.05:
            return "source_left_workspace"
        if displacement > float(self._config["max_support_displacement"]):
            return "support_displaced_excessively"
        diagnostics = self.env.get_stack_diagnostics()
        if diagnostics["support_uprightness"] < float(self._config["min_allowed_uprightness"]):
            return "support_tipped"
        carrying = self.current_phase in {
            StackingPhase.TRANSPORT_ABOVE_SUPPORT, StackingPhase.ALIGN,
            StackingPhase.LOWER_TO_PRECONTACT,
        }
        if carrying and source[2] < self._initial_source[2] + 0.04:
            return "dropped_source"
        return None

    def _phase_timeout(self) -> int:
        if self.current_phase == StackingPhase.GRASP:
            return int(self._config["grasp_phase_timeout"])
        if self.current_phase == StackingPhase.LOWER_TO_CONTACT:
            return int(self._config["contact_phase_timeout"])
        if self.current_phase == StackingPhase.RELEASE:
            return int(self._config["release_phase_timeout"])
        if self.current_phase == StackingPhase.VERIFY_STACK:
            return int(self._config["verify_phase_timeout"])
        return int(self._config["movement_phase_timeout"])

    def _timeout_reason(self) -> str:
        return {
            StackingPhase.GRASP: "failed_grasp",
            StackingPhase.LIFT: "failed_lift",
            StackingPhase.ALIGN: "failed_stacking_alignment",
            StackingPhase.LOWER_TO_CONTACT: "failed_placement",
            StackingPhase.RELEASE: "unstable_after_release",
            StackingPhase.RETREAT: "failed_retreat",
            StackingPhase.VERIFY_STACK: "success_condition_never_reached",
        }.get(self.current_phase, "phase_timeout")

    def _emit(self, action: np.ndarray) -> np.ndarray:
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            return self._fail("invalid_action")
        self.step_count += 1
        self.phase_step_count += 1
        return action.astype(np.float64)

    def _transition(self, phase: StackingPhase) -> None:
        self.current_phase = phase
        self.phase_step_count = 0
        self.phase_history.append(phase)

    def _finish(self) -> np.ndarray:
        self.success = True
        self.failure_reason = None
        self.is_done = True
        self._transition(StackingPhase.DONE)
        return self._emit(self._stationary(close=False))

    def _fail(self, reason: str) -> np.ndarray:
        self.success = False
        self.failure_reason = reason
        self.is_done = True
        self._transition(StackingPhase.DONE)
        return self._stationary(close=False)


def rollout_stacking_expert(env: PandaStackingEnv,
                            expert: ScriptedStackingExpert) -> dict[str, Any]:
    """Run one episode and return an in-memory trace; no serialization occurs."""
    observation = env.get_observation()
    expert.reset(env)
    trace: dict[str, list[Any]] = {
        "observations": [], "actions": [], "ee_positions": [],
        "source_positions": [], "support_positions": [],
        "source_orientations": [], "support_orientations": [],
        "phases": [], "stacking_diagnostics": [],
    }
    while not expert.is_done:
        trace["observations"].append(observation.copy())
        trace["ee_positions"].append(env.get_ee_position())
        trace["source_positions"].append(env.get_source_position())
        trace["support_positions"].append(env.get_support_position())
        trace["source_orientations"].append(env.get_source_pose()[1])
        trace["support_orientations"].append(env.get_support_pose()[1])
        trace["phases"].append(expert.current_phase.value)
        trace["stacking_diagnostics"].append(env.get_stack_diagnostics())
        action = expert.act(observation)
        trace["actions"].append(action.copy())
        observation, _ = env.step(action)
    result = expert.result()
    return {
        **{key: np.asarray(value) if key != "stacking_diagnostics" else value
           for key, value in trace.items()},
        "success": result.success, "failure_reason": result.failure_reason,
        "result": result,
    }
