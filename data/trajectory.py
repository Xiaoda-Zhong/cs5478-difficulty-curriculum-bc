"""Trajectory rollout, serialization, validation, and fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.task_sampler import TaskConfig
from envs.panda_pick_place import PandaPickPlaceEnv
from expert.scripted_expert import ScriptedExpert


REQUIRED_TRAJECTORY_KEYS = {
    "trajectory_id",
    "task_id",
    "task_seed",
    "success",
    "observations",
    "actions",
    "initial_ee_position",
    "initial_object_position",
    "task_object_position",
    "goal_position",
    "target_radius",
    "ee_positions",
    "object_positions",
    "final_object_position",
    "trajectory_length",
    "ee_path_length",
    "cumulative_action_magnitude",
}


def rollout_expert(
    env: PandaPickPlaceEnv,
    expert: ScriptedExpert,
    task: TaskConfig,
    trajectory_id: str,
) -> dict[str, Any]:
    """Execute and record one deterministic expert rollout."""
    observation = env.reset(
        object_position=task.object_position,
        goal_position=task.goal_position,
        target_radius=task.target_radius,
        seed=task.seed,
    )
    expert.reset(env)
    initial_observation = observation.copy()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[str] = []
    ee_positions = [env.get_ee_position()]
    object_positions = [env.get_object_position()]

    while not expert.is_done:
        observations.append(observation.copy())
        phases.append(expert.current_phase.value)
        action = expert.act(observation)
        actions.append(action.copy())
        observation, _ = env.step(action)
        ee_positions.append(env.get_ee_position())
        object_positions.append(env.get_object_position())

    result = expert.result()
    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    ee_array = np.asarray(ee_positions, dtype=np.float64)
    object_array = np.asarray(object_positions, dtype=np.float64)
    ee_path_length = float(np.linalg.norm(np.diff(ee_array, axis=0), axis=1).sum())
    cumulative_action_magnitude = float(
        np.linalg.norm(action_array[:, :3], axis=1).sum()
    )

    return {
        "trajectory_id": trajectory_id,
        "task_id": task.task_id,
        "task_seed": task.seed,
        "success": bool(result.success and env.is_success()),
        "failure_reason": result.failure_reason or "",
        "observations": observation_array,
        "actions": action_array,
        "initial_ee_position": initial_observation[:3].astype(np.float64),
        "initial_object_position": initial_observation[3:6].astype(np.float64),
        "task_object_position": task.object_position.astype(np.float64),
        "goal_position": task.goal_position.astype(np.float64),
        "target_radius": float(task.target_radius),
        "ee_positions": ee_array,
        "object_positions": object_array,
        "final_object_position": env.get_object_position(),
        "trajectory_length": int(action_array.shape[0]),
        "ee_path_length": ee_path_length,
        "cumulative_action_magnitude": cumulative_action_magnitude,
        "phases": np.asarray(phases, dtype="U12"),
    }


def save_trajectory(path: str | Path, trajectory: Mapping[str, Any]) -> None:
    """Validate and save one trajectory as a compressed NumPy archive."""
    validate_trajectory(trajectory)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **trajectory)


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Load one trajectory without permitting pickled object arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_trajectory(
    trajectory: Mapping[str, Any], *, require_success: bool = False
) -> None:
    """Raise ``ValueError`` when a trajectory violates the dataset schema."""
    missing = REQUIRED_TRAJECTORY_KEYS.difference(trajectory)
    if missing:
        raise ValueError(f"Missing trajectory fields: {sorted(missing)}")

    observations = np.asarray(trajectory["observations"])
    actions = np.asarray(trajectory["actions"])
    length = int(np.asarray(trajectory["trajectory_length"]).item())
    ee_positions = np.asarray(trajectory["ee_positions"])
    object_positions = np.asarray(trajectory["object_positions"])
    if observations.shape != (length, 10):
        raise ValueError("observations must have shape (trajectory_length, 10)")
    if actions.shape != (length, 4):
        raise ValueError("actions must have shape (trajectory_length, 4)")
    if ee_positions.shape != (length + 1, 3):
        raise ValueError("ee_positions must have shape (trajectory_length + 1, 3)")
    if object_positions.shape != (length + 1, 3):
        raise ValueError("object_positions must have shape (trajectory_length + 1, 3)")
    if length <= 0:
        raise ValueError("trajectory_length must be positive")

    numeric_values = (
        observations,
        actions,
        ee_positions,
        object_positions,
        np.asarray(trajectory["initial_ee_position"]),
        np.asarray(trajectory["initial_object_position"]),
        np.asarray(trajectory["task_object_position"]),
        np.asarray(trajectory["goal_position"]),
        np.asarray(trajectory["final_object_position"]),
        np.asarray([trajectory["target_radius"]], dtype=float),
        np.asarray([trajectory["ee_path_length"]], dtype=float),
        np.asarray([trajectory["cumulative_action_magnitude"]], dtype=float),
    )
    if not all(np.all(np.isfinite(value)) for value in numeric_values):
        raise ValueError("trajectory contains NaN or Inf")
    if np.asarray(trajectory["initial_ee_position"]).shape != (3,):
        raise ValueError("initial_ee_position must have shape (3,)")
    if np.asarray(trajectory["initial_object_position"]).shape != (3,):
        raise ValueError("initial_object_position must have shape (3,)")
    if np.asarray(trajectory["goal_position"]).shape != (3,):
        raise ValueError("goal_position must have shape (3,)")
    if float(np.asarray(trajectory["target_radius"]).item()) <= 0.0:
        raise ValueError("target_radius must be positive")
    success = bool(np.asarray(trajectory["success"]).item())
    if require_success and not success:
        raise ValueError("dataset trajectory is not successful")


def task_fingerprint(task_entries: list[Mapping[str, Any]]) -> str:
    """Hash ordered task configurations using canonical JSON encoding."""
    canonical = json.dumps(
        task_entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fingerprint_entry(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the stable task fields used by the dataset fingerprint."""
    return {
        "trajectory_id": _text_scalar(trajectory["trajectory_id"]),
        "task_id": _text_scalar(trajectory["task_id"]),
        "task_seed": int(np.asarray(trajectory["task_seed"]).item()),
        "object_position": np.asarray(
            trajectory["task_object_position"], dtype=float
        ).tolist(),
        "goal_position": np.asarray(trajectory["goal_position"], dtype=float).tolist(),
        "target_radius": float(np.asarray(trajectory["target_radius"]).item()),
    }


def _text_scalar(value: Any) -> str:
    return str(np.asarray(value).item())
