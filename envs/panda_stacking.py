"""Deterministic Franka Panda environment for physical object stacking.

World coordinates are metres in PyBullet's right-handed frame with +z upward.
The 29-D float32 observation is ordered as::

    [ee_xyz,
     source_xyz, source_up, source_linear_velocity, source_angular_velocity,
     support_xyz, support_up, support_linear_velocity, support_angular_velocity,
     gripper_opening, placement_tolerance]

Cube ``up`` vectors are local +z axes transformed into world coordinates. This
task state is not claimed to be a complete Markov state because robot joint
velocities and other simulator variables are omitted.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pybullet as p
import pybullet_data
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "stacking_v1.yaml"


class PandaStackingEnv:
    """State-only two-cube stacking environment with persistent success."""

    observation_dim = 29
    action_dim = 4

    def __init__(self, mode: str = "direct", config_path: str | Path | None = None,
                 config: Mapping[str, Any] | None = None) -> None:
        normalized_mode = mode.lower()
        if normalized_mode not in {"gui", "direct"}:
            raise ValueError("mode must be either 'gui' or 'direct'")
        if config_path is not None and config is not None:
            raise ValueError("Provide config_path or config, not both")
        self.mode = normalized_mode
        self.config = self._load_config(config_path, config)
        self._validate_config()
        self.rng = np.random.default_rng(int(self.config["seed"]))
        self.client_id = p.connect(p.GUI if self.mode == "gui" else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError(f"Could not connect to PyBullet in {self.mode!r} mode")
        self.robot_id: int | None = None
        self.table_id: int | None = None
        self.source_cube_id: int | None = None
        self.support_cube_id: int | None = None
        self.arm_joint_indices: list[int] = []
        self.finger_joint_indices: list[int] = []
        self.ee_link_index: int | None = None
        self._source_position = np.zeros(3)
        self._support_position = np.zeros(3)
        self._placement_tolerance = 0.0
        self._task_seed: int | None = None
        self._ee_target_position = np.zeros(3)
        self._last_applied_action = np.zeros(4)
        self._stable_steps = 0
        self._success = False
        self._last_diagnostics: dict[str, Any] = {}
        self._closed = False
        try:
            self.reset()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _load_config(config_path: str | Path | None,
                     config: Mapping[str, Any] | None) -> dict[str, Any]:
        if config is not None:
            return deepcopy(dict(config))
        path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        with path.expanduser().open("r", encoding="utf-8") as source:
            loaded = yaml.safe_load(source)
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a mapping: {path}")
        return loaded

    def _validate_config(self) -> None:
        if len(self.config["robot"]["initial_joint_positions"]) != 7:
            raise ValueError("robot.initial_joint_positions must contain 7 values")
        if len(self.config["robot"]["fixed_ee_orientation"]) != 4:
            raise ValueError("robot.fixed_ee_orientation must contain 4 values")
        bounds = np.asarray(self.config["control"]["workspace_bounds"], dtype=float)
        if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("control.workspace_bounds must have shape (3, 2)")
        if float(self.config["cubes"]["edge_length"]) <= 0.0:
            raise ValueError("cubes.edge_length must be positive")
        if int(self.config["success"]["settling_control_steps"]) <= 0:
            raise ValueError("success.settling_control_steps must be positive")

    def reset(self, source_position: Sequence[float] | None = None,
              support_position: Sequence[float] | None = None,
              placement_tolerance: float | None = None,
              seed: int | None = None) -> np.ndarray:
        """Reset to one explicit, deterministic, collision-free task."""
        self._ensure_open()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._task_seed = seed
        half = self.cube_edge_length / 2.0
        self._source_position = self._validate_cube_position(
            source_position if source_position is not None
            else self.config["task"]["source_position"],
            "source_position", half)
        self._support_position = self._validate_cube_position(
            support_position if support_position is not None
            else self.config["task"]["support_position"],
            "support_position", half)
        separation = float(np.linalg.norm(self._source_position[:2] - self._support_position[:2]))
        minimum = max(self.cube_edge_length, float(self.config["task"]["min_initial_xy_separation"]))
        if separation < minimum:
            raise ValueError("source and support cubes overlap or are too close initially")
        self._placement_tolerance = float(
            placement_tolerance if placement_tolerance is not None
            else self.config["task"]["placement_tolerance"])
        if not 0.0 < self._placement_tolerance < self.cube_edge_length / 2.0:
            raise ValueError("placement_tolerance must be between zero and half the cube edge")

        client = self.client_id
        p.resetSimulation(physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        p.setTimeStep(float(self.config["physics"]["timestep"]), physicsClientId=client)
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1,
            numSolverIterations=int(self.config["physics"]["solver_iterations"]),
            physicsClientId=client)
        p.loadURDF("plane.urdf", [0.0, 0.0, -0.65], physicsClientId=client)
        self.table_id = p.loadURDF("table/table.urdf", self.config["table"]["base_position"],
                                   useFixedBase=True, physicsClientId=client)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0],
                                   useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION,
                                   physicsClientId=client)
        self._discover_robot_joints()
        self._reset_robot()
        self.source_cube_id = self._create_cube(
            self._source_position, float(self.config["cubes"]["source_mass"]),
            self.config["cubes"]["source_color"])
        self.support_cube_id = self._create_cube(
            self._support_position, float(self.config["cubes"]["support_mass"]),
            self.config["cubes"]["support_color"])
        for _ in range(int(self.config["physics"]["reset_settle_steps"])):
            p.stepSimulation(physicsClientId=client)
        self._ee_target_position = self.get_ee_position()
        self._last_applied_action = np.zeros(4)
        self._stable_steps = 0
        self._success = False
        self._last_diagnostics = self.get_stack_diagnostics()
        return self.get_observation()

    def _validate_cube_position(self, value: Sequence[float], name: str,
                                half: float) -> np.ndarray:
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name} must be a finite 3-D position")
        expected_z = float(self.config["table"]["surface_height"]) + half
        if not np.isclose(position[2], expected_z, atol=1e-9):
            raise ValueError(f"{name} z must place the cube on the table at {expected_z}")
        for coordinate, key in zip(position[:2], ("x_bounds", "y_bounds"), strict=True):
            lower, upper = map(float, self.config["table"][key])
            if not lower + half <= coordinate <= upper - half:
                raise ValueError(f"{name} lies outside the usable table bounds")
        return position.copy()

    def _discover_robot_joints(self) -> None:
        assert self.robot_id is not None
        self.arm_joint_indices, self.finger_joint_indices = [], []
        self.ee_link_index = None
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
            joint_name, link_name = info[1].decode(), info[12].decode()
            if joint_name in {f"panda_joint{i}" for i in range(1, 8)}:
                self.arm_joint_indices.append(index)
            if joint_name in {"panda_finger_joint1", "panda_finger_joint2"}:
                self.finger_joint_indices.append(index)
            if link_name == "panda_grasptarget":
                self.ee_link_index = index
        if len(self.arm_joint_indices) != 7 or len(self.finger_joint_indices) != 2:
            raise RuntimeError("Unexpected Franka Panda URDF joint layout")
        if self.ee_link_index is None:
            raise RuntimeError("Could not find the Panda grasp target link")

    def _reset_robot(self) -> None:
        assert self.robot_id is not None
        joints = self.config["robot"]["initial_joint_positions"]
        for index, value in zip(self.arm_joint_indices, joints, strict=True):
            p.resetJointState(self.robot_id, index, float(value), physicsClientId=self.client_id)
        opening = float(self.config["control"]["gripper_open_position"])
        for index in self.finger_joint_indices:
            p.resetJointState(self.robot_id, index, opening, physicsClientId=self.client_id)
        self._command_arm(joints)
        self._command_gripper(opening)

    def _create_cube(self, position: np.ndarray, mass: float, color: Sequence[float]) -> int:
        half = self.cube_edge_length / 2.0
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3,
                                           physicsClientId=self.client_id)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3,
                                     rgbaColor=color, physicsClientId=self.client_id)
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision,
                                 baseVisualShapeIndex=visual, basePosition=position,
                                 baseOrientation=[0, 0, 0, 1], physicsClientId=self.client_id)
        p.changeDynamics(body, -1,
                         lateralFriction=float(self.config["cubes"]["lateral_friction"]),
                         rollingFriction=float(self.config["cubes"]["rolling_friction"]),
                         spinningFriction=float(self.config["cubes"]["spinning_friction"]),
                         physicsClientId=self.client_id)
        return body

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply one Cartesian/gripper action and advance physical stability."""
        self._ensure_open()
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (4,) or not np.all(np.isfinite(array)):
            raise ValueError("action must be a finite vector with shape (4,)")
        maximum = float(self.config["control"]["max_cartesian_delta"])
        delta, gripper = np.clip(array[:3], -maximum, maximum), float(np.clip(array[3], -1, 1))
        bounds = np.asarray(self.config["control"]["workspace_bounds"], dtype=float)
        desired = np.clip(self._ee_target_position + delta, bounds[:, 0], bounds[:, 1])
        self._command_arm(self._calculate_ik(desired))
        gripper_target = float(self.config["control"][
            "gripper_closed_position" if gripper > 0 else "gripper_open_position"])
        self._command_gripper(gripper_target)
        for _ in range(int(self.config["physics"]["sim_steps_per_action"])):
            p.stepSimulation(physicsClientId=self.client_id)
        self._ee_target_position = desired
        self._last_applied_action = np.concatenate([delta, [gripper]])
        self._update_stability()
        info = {
            "is_success": self._success,
            "ee_target_position": desired.copy(),
            "applied_action": self._last_applied_action.copy(),
            "stack_alignment_error": self._last_diagnostics["horizontal_alignment_error"],
            "physical_contact": self._last_diagnostics["source_support_contact"],
            "stable_steps": self._stable_steps,
            "stack_diagnostics": dict(self._last_diagnostics),
        }
        return self.get_observation(), info

    def _calculate_ik(self, desired: np.ndarray) -> tuple[float, ...]:
        assert self.robot_id is not None and self.ee_link_index is not None
        result = p.calculateInverseKinematics(
            self.robot_id, self.ee_link_index, targetPosition=desired,
            targetOrientation=self.config["robot"]["fixed_ee_orientation"],
            maxNumIterations=100, residualThreshold=1e-5,
            physicsClientId=self.client_id)
        return tuple(float(result[index]) for index in self.arm_joint_indices)

    def _command_arm(self, targets: Sequence[float]) -> None:
        assert self.robot_id is not None
        p.setJointMotorControlArray(self.robot_id, self.arm_joint_indices, p.POSITION_CONTROL,
                                    targetPositions=list(targets), forces=[200.0] * 7,
                                    positionGains=[0.2] * 7, physicsClientId=self.client_id)

    def _command_gripper(self, target: float) -> None:
        assert self.robot_id is not None
        p.setJointMotorControlArray(self.robot_id, self.finger_joint_indices, p.POSITION_CONTROL,
                                    targetPositions=[target] * 2, forces=[40.0] * 2,
                                    physicsClientId=self.client_id)

    def _body_state(self, body_id: int | None) -> tuple[np.ndarray, ...]:
        self._ensure_open()
        if body_id is None:
            raise RuntimeError("Cube body has not been created")
        position, orientation = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
        linear, angular = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
        rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        return tuple(np.asarray(value, dtype=np.float64)
                     for value in (position, orientation, linear, angular, rotation[:, 2]))

    def get_observation(self) -> np.ndarray:
        sp, _, sl, sa, su = self._body_state(self.source_cube_id)
        tp, _, tl, ta, tu = self._body_state(self.support_cube_id)
        observation = np.concatenate([self.get_ee_position(), sp, su, sl, sa,
                                      tp, tu, tl, ta,
                                      [self.get_gripper_opening(), self._placement_tolerance]]).astype(np.float32)
        if observation.shape != (29,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("Stacking observation is invalid")
        return observation

    def get_ee_position(self) -> np.ndarray:
        self._ensure_open()
        assert self.robot_id is not None and self.ee_link_index is not None
        state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True,
                               physicsClientId=self.client_id)
        return np.asarray(state[4], dtype=np.float64)

    def get_source_position(self) -> np.ndarray:
        return self.get_source_pose()[0]

    def get_support_position(self) -> np.ndarray:
        return self.get_support_pose()[0]

    def get_source_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation, *_ = self._body_state(self.source_cube_id)
        return position, orientation

    def get_support_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation, *_ = self._body_state(self.support_cube_id)
        return position, orientation

    def get_source_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        _, _, linear, angular, _ = self._body_state(self.source_cube_id)
        return linear, angular

    def get_support_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        _, _, linear, angular, _ = self._body_state(self.support_cube_id)
        return linear, angular

    def get_gripper_opening(self) -> float:
        assert self.robot_id is not None
        states = p.getJointStates(self.robot_id, self.finger_joint_indices,
                                  physicsClientId=self.client_id)
        return float(np.mean([state[0] for state in states]))

    def get_stack_diagnostics(self) -> dict[str, Any]:
        source, _, source_linear, source_angular, source_up = self._body_state(self.source_cube_id)
        support, _, support_linear, support_angular, support_up = self._body_state(self.support_cube_id)
        vertical = float(source[2] - support[2])
        return {
            "source_above_support": bool(vertical > 0),
            "horizontal_alignment_error": float(np.linalg.norm(source[:2] - support[:2])),
            "vertical_separation": vertical,
            "vertical_stacking_error": abs(vertical - self.cube_edge_length),
            "source_support_contact": self._has_contact(self.source_cube_id, self.support_cube_id),
            "source_table_contact": self._has_contact(self.source_cube_id, self.table_id),
            "source_linear_speed": float(np.linalg.norm(source_linear)),
            "source_angular_speed": float(np.linalg.norm(source_angular)),
            "support_linear_speed": float(np.linalg.norm(support_linear)),
            "support_angular_speed": float(np.linalg.norm(support_angular)),
            "source_uprightness": float(source_up[2]),
            "support_uprightness": float(support_up[2]),
            "gripper_opening": self.get_gripper_opening(),
            "gripper_retreat_clearance": float(np.linalg.norm(self.get_ee_position() - source)),
            "stable_steps": self._stable_steps,
            "is_success": self._success,
        }

    def is_stack_geometrically_aligned(self) -> bool:
        d, s = self.get_stack_diagnostics(), self.config["success"]
        return bool(d["source_above_support"]
                    and d["horizontal_alignment_error"] <= self._placement_tolerance
                    and d["vertical_stacking_error"] <= float(s["vertical_tolerance"])
                    and d["source_uprightness"] >= float(s["upright_min_cosine"])
                    and d["support_uprightness"] >= float(s["upright_min_cosine"]))

    def is_stack_physically_stable(self) -> bool:
        """Return the instantaneous physical-stack condition, without persistence."""
        d, s = self.get_stack_diagnostics(), self.config["success"]
        return bool(d["source_above_support"]
                    and d["horizontal_alignment_error"] <= self._placement_tolerance
                    and d["vertical_stacking_error"] <= float(s["vertical_tolerance"])
                    and d["source_uprightness"] >= float(s["upright_min_cosine"])
                    and d["support_uprightness"] >= float(s["upright_min_cosine"])
                    and (d["source_support_contact"] or not s["require_source_support_contact"])
                    and (not d["source_table_contact"] or not s["require_no_source_table_contact"])
                    and d["source_linear_speed"] <= float(s["source_max_linear_speed"])
                    and d["source_angular_speed"] <= float(s["source_max_angular_speed"])
                    and d["support_linear_speed"] <= float(s["support_max_linear_speed"])
                    and d["support_angular_speed"] <= float(s["support_max_angular_speed"])
                    and d["gripper_opening"] >= float(s["gripper_open_threshold"])
                    and d["gripper_retreat_clearance"] >= float(s["gripper_retreat_clearance"]))

    def _update_stability(self) -> None:
        self._stable_steps = self._stable_steps + 1 if self.is_stack_physically_stable() else 0
        self._success = self._stable_steps >= int(self.config["success"]["settling_control_steps"])
        self._last_diagnostics = self.get_stack_diagnostics()

    def _has_contact(self, body_a: int | None, body_b: int | None) -> bool:
        return bool(body_a is not None and body_b is not None and
                    p.getContactPoints(bodyA=body_a, bodyB=body_b,
                                       physicsClientId=self.client_id))

    def is_success(self) -> bool:
        return self._success

    @property
    def cube_edge_length(self) -> float:
        return float(self.config["cubes"]["edge_length"])

    @property
    def placement_tolerance(self) -> float:
        return self._placement_tolerance

    @property
    def stable_steps(self) -> int:
        return self._stable_steps

    @property
    def ee_target_position(self) -> np.ndarray:
        return self._ee_target_position.copy()

    @property
    def last_applied_action(self) -> np.ndarray:
        return self._last_applied_action.copy()

    def _ensure_open(self) -> None:
        if self._closed or not p.isConnected(self.client_id):
            raise RuntimeError("Environment is closed")

    def close(self) -> None:
        if not self._closed and p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
        self._closed = True

    def __enter__(self) -> "PandaStackingEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
