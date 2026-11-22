"""Minimal deterministic Franka Panda pick-and-place environment.

The public observation is a 10-element NumPy array, ordered as
``[ee_x, ee_y, ee_z, object_x, object_y, object_z,
goal_x, goal_y, goal_z, gripper_state]``. Positions and gripper state are in
metres. ``gripper_state`` is the mean position of the two finger joints and is
approximately 0.04 when open and 0.0 when closed.

The public action is ``[delta_x, delta_y, delta_z, gripper_command]``.
Cartesian deltas are in metres and are clipped per axis. A positive gripper
command closes the fingers; zero or a negative command opens them.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pybullet as p
import pybullet_data
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class PandaPickPlaceEnv:
    """A small state-only PyBullet environment for deterministic experiments."""

    observation_dim = 10
    action_dim = 4

    def __init__(
        self,
        mode: str = "direct",
        config_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_mode = mode.lower()
        if normalized_mode not in {"gui", "direct"}:
            raise ValueError("mode must be either 'gui' or 'direct'")

        self.mode = normalized_mode
        self.config = self._load_config(config_path, config)
        self.rng = np.random.default_rng(int(self.config["seed"]))
        self._validate_config()

        connection_mode = p.GUI if self.mode == "gui" else p.DIRECT
        self.client_id = p.connect(connection_mode)
        if self.client_id < 0:
            raise RuntimeError(f"Could not connect to PyBullet in {self.mode!r} mode")

        self.robot_id: int | None = None
        self.cube_id: int | None = None
        self.target_id: int | None = None
        self.arm_joint_indices: list[int] = []
        self.finger_joint_indices: list[int] = []
        self.ee_link_index: int | None = None
        self._ee_target_position = np.zeros(3, dtype=np.float64)
        self._last_applied_action = np.zeros(4, dtype=np.float64)
        self._object_position = np.zeros(3, dtype=np.float64)
        self._goal_position = np.zeros(3, dtype=np.float64)
        self._target_radius = 0.0
        self._task_seed: int | None = None
        self._closed = False

        try:
            self.reset()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _load_config(
        config_path: str | Path | None, config: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if config_path is not None and config is not None:
            raise ValueError("Provide config_path or config, not both")
        if config is not None:
            return deepcopy(dict(config))

        path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        with path.expanduser().open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a mapping: {path}")
        return loaded

    def _validate_config(self) -> None:
        joints = self.config["robot"]["initial_joint_positions"]
        orientation = self.config["robot"]["fixed_ee_orientation"]
        bounds = np.asarray(self.config["control"]["workspace_bounds"], dtype=float)
        if len(joints) != 7:
            raise ValueError("robot.initial_joint_positions must contain 7 values")
        if len(orientation) != 4:
            raise ValueError("robot.fixed_ee_orientation must contain 4 values")
        if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("control.workspace_bounds must have shape (3, 2)")
        if float(self.config["control"]["max_cartesian_delta"]) <= 0:
            raise ValueError("control.max_cartesian_delta must be positive")

    def reset(
        self,
        object_position: Sequence[float] | np.ndarray | None = None,
        goal_position: Sequence[float] | np.ndarray | None = None,
        target_radius: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Restore a specified task, or the fixed configured task by default."""
        self._ensure_open()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._task_seed = seed
        self._object_position = self._validate_position(
            object_position
            if object_position is not None
            else self.config["task"]["cube_position"],
            "object_position",
        )
        self._goal_position = self._validate_position(
            goal_position
            if goal_position is not None
            else self.config["task"]["goal_position"],
            "goal_position",
        )
        self._target_radius = float(
            target_radius
            if target_radius is not None
            else self.config["task"]["target_radius"]
        )
        if not np.isfinite(self._target_radius) or self._target_radius <= 0.0:
            raise ValueError("target_radius must be a positive finite value")

        client = self.client_id
        p.resetSimulation(physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        p.setTimeStep(
            float(self.config["physics"]["timestep"]), physicsClientId=client
        )
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1,
            numSolverIterations=100,
            physicsClientId=client,
        )

        p.loadURDF("plane.urdf", [0.0, 0.0, -0.65], physicsClientId=client)
        p.loadURDF(
            "table/table.urdf",
            # This places the stock table URDF's upper surface at world z=0.
            [0.5, 0.0, -0.625],
            useFixedBase=True,
            physicsClientId=client,
        )
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            [0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=client,
        )
        self._discover_robot_joints()
        self._reset_robot()
        self.cube_id = self._create_cube(self._object_position)
        self.target_id = self._create_target_marker(
            self._goal_position, self._target_radius
        )

        for _ in range(int(self.config["physics"]["reset_settle_steps"])):
            p.stepSimulation(physicsClientId=client)

        self._ee_target_position = self.get_ee_position().astype(np.float64)
        self._last_applied_action = np.zeros(4, dtype=np.float64)
        return self.get_observation()

    def _discover_robot_joints(self) -> None:
        assert self.robot_id is not None
        self.arm_joint_indices = []
        self.finger_joint_indices = []
        self.ee_link_index = None

        for joint_index in range(
            p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        ):
            info = p.getJointInfo(
                self.robot_id, joint_index, physicsClientId=self.client_id
            )
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            if joint_name in {f"panda_joint{i}" for i in range(1, 8)}:
                self.arm_joint_indices.append(joint_index)
            if joint_name in {"panda_finger_joint1", "panda_finger_joint2"}:
                self.finger_joint_indices.append(joint_index)
            if link_name == "panda_grasptarget":
                self.ee_link_index = joint_index

        if len(self.arm_joint_indices) != 7 or len(self.finger_joint_indices) != 2:
            raise RuntimeError("Unexpected Franka Panda URDF joint layout")
        if self.ee_link_index is None:
            raise RuntimeError("Could not find the Panda grasp target link")

    def _reset_robot(self) -> None:
        assert self.robot_id is not None
        initial_joints = self.config["robot"]["initial_joint_positions"]
        for joint_index, joint_position in zip(
            self.arm_joint_indices, initial_joints, strict=True
        ):
            p.resetJointState(
                self.robot_id,
                joint_index,
                float(joint_position),
                physicsClientId=self.client_id,
            )

        open_position = float(self.config["control"]["gripper_open_position"])
        for joint_index in self.finger_joint_indices:
            p.resetJointState(
                self.robot_id,
                joint_index,
                open_position,
                physicsClientId=self.client_id,
            )
        self._command_arm(initial_joints)
        self._command_gripper(open_position)

    @staticmethod
    def _validate_position(
        value: Sequence[float] | np.ndarray, name: str
    ) -> np.ndarray:
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name} must be a finite 3-D position")
        return position.copy()

    def _create_cube(self, position: np.ndarray) -> int:
        half_extent = 0.025
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_extent] * 3,
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_extent] * 3,
            rgbaColor=[0.85, 0.15, 0.10, 1.0],
            physicsClientId=self.client_id,
        )
        cube_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position.tolist(),
            physicsClientId=self.client_id,
        )
        p.changeDynamics(
            cube_id,
            -1,
            lateralFriction=0.8,
            rollingFriction=0.001,
            spinningFriction=0.001,
            physicsClientId=self.client_id,
        )
        return cube_id

    def _create_target_marker(self, goal: np.ndarray, target_radius: float) -> int:
        marker_height = 0.002
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=target_radius,
            length=marker_height,
            rgbaColor=[0.15, 0.75, 0.25, 0.55],
            physicsClientId=self.client_id,
        )
        marker_position = [float(goal[0]), float(goal[1]), marker_height / 2.0]
        return p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual,
            basePosition=marker_position,
            physicsClientId=self.client_id,
        )

    def step(self, action: Sequence[float] | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply one clipped Cartesian/gripper command and advance physics."""
        self._ensure_open()
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},)")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain only finite values")

        max_delta = float(self.config["control"]["max_cartesian_delta"])
        clipped_delta = np.clip(action_array[:3], -max_delta, max_delta)
        gripper_command = float(np.clip(action_array[3], -1.0, 1.0))
        bounds = np.asarray(self.config["control"]["workspace_bounds"], dtype=float)
        desired_position = np.clip(
            self._ee_target_position + clipped_delta, bounds[:, 0], bounds[:, 1]
        )

        joint_targets = self._calculate_ik(desired_position)
        self._command_arm(joint_targets)
        gripper_position = float(
            self.config["control"][
                "gripper_closed_position" if gripper_command > 0.0 else "gripper_open_position"
            ]
        )
        self._command_gripper(gripper_position)

        for _ in range(int(self.config["physics"]["sim_steps_per_action"])):
            p.stepSimulation(physicsClientId=self.client_id)

        self._ee_target_position = desired_position
        self._last_applied_action = np.concatenate(
            [clipped_delta, np.asarray([gripper_command])]
        )
        observation = self.get_observation()
        info = {
            "is_success": self.is_success(),
            "ee_target_position": desired_position.copy(),
            "applied_action": self._last_applied_action.copy(),
        }
        return observation, info

    def _calculate_ik(self, desired_position: np.ndarray) -> tuple[float, ...]:
        assert self.robot_id is not None and self.ee_link_index is not None
        result = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            targetPosition=desired_position.tolist(),
            targetOrientation=self.config["robot"]["fixed_ee_orientation"],
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client_id,
        )
        return tuple(float(result[index]) for index in self.arm_joint_indices)

    def _command_arm(self, joint_targets: Sequence[float]) -> None:
        assert self.robot_id is not None
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=list(joint_targets),
            forces=[200.0] * len(self.arm_joint_indices),
            positionGains=[0.2] * len(self.arm_joint_indices),
            physicsClientId=self.client_id,
        )

    def _command_gripper(self, target_position: float) -> None:
        assert self.robot_id is not None
        p.setJointMotorControlArray(
            self.robot_id,
            self.finger_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[target_position] * len(self.finger_joint_indices),
            forces=[40.0] * len(self.finger_joint_indices),
            physicsClientId=self.client_id,
        )

    def get_observation(self) -> np.ndarray:
        """Return the 10-D state observation as float32."""
        self._ensure_open()
        finger_positions = self._get_finger_positions()
        observation = np.concatenate(
            [
                self.get_ee_position(),
                self.get_object_position(),
                self.get_goal_position(),
                np.asarray([np.mean(finger_positions)], dtype=np.float64),
            ]
        )
        return observation.astype(np.float32)

    def get_ee_position(self) -> np.ndarray:
        self._ensure_open()
        assert self.robot_id is not None and self.ee_link_index is not None
        state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return np.asarray(state[4], dtype=np.float64)

    def get_object_position(self) -> np.ndarray:
        self._ensure_open()
        assert self.cube_id is not None
        position, _ = p.getBasePositionAndOrientation(
            self.cube_id, physicsClientId=self.client_id
        )
        return np.asarray(position, dtype=np.float64)

    def get_goal_position(self) -> np.ndarray:
        self._ensure_open()
        return self._goal_position.copy()

    def _get_finger_positions(self) -> np.ndarray:
        assert self.robot_id is not None
        states = p.getJointStates(
            self.robot_id,
            self.finger_joint_indices,
            physicsClientId=self.client_id,
        )
        return np.asarray([state[0] for state in states], dtype=np.float64)

    def is_success(self) -> bool:
        object_position = self.get_object_position()
        goal_position = self.get_goal_position()
        xy_distance = float(np.linalg.norm(object_position[:2] - goal_position[:2]))
        height_error = abs(float(object_position[2] - goal_position[2]))
        return bool(
            xy_distance < self._target_radius
            and height_error < float(self.config["task"]["target_height_tolerance"])
        )

    @property
    def target_radius(self) -> float:
        """Return the active task's XY success radius in metres."""
        return self._target_radius

    @property
    def task_seed(self) -> int | None:
        """Return the seed associated with the active explicit task."""
        return self._task_seed

    @property
    def ee_target_position(self) -> np.ndarray:
        """Return a copy of the controller's current Cartesian target."""
        return self._ee_target_position.copy()

    @property
    def last_applied_action(self) -> np.ndarray:
        """Return the most recent action after clipping."""
        return self._last_applied_action.copy()

    def _ensure_open(self) -> None:
        if self._closed or not p.isConnected(self.client_id):
            raise RuntimeError("Environment is closed")

    def close(self) -> None:
        """Disconnect this environment's PyBullet client; safe to call repeatedly."""
        if not self._closed and p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
        self._closed = True

    def __enter__(self) -> "PandaPickPlaceEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
