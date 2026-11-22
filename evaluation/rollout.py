"""Closed-loop rollout utilities for learned behavior-cloning policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from data.task_sampler import TaskConfig, TaskSampler
from envs.panda_pick_place import PandaPickPlaceEnv
from models.bc_policy import BCPolicyAgent


@dataclass(frozen=True)
class BCRolloutResult:
    task_id: str
    success: bool
    failure_reason: str | None
    steps: int
    final_object_position: list[float]
    final_object_goal_xy_error: float
    physical_events: dict[str, bool]
    object_goal_xy_distance_trace: list[float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rollout_bc_policy(
    env: PandaPickPlaceEnv,
    policy: BCPolicyAgent,
    task: TaskConfig,
    max_steps: int,
    step_callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
) -> BCRolloutResult:
    """Run an autonomous policy without scripted-expert intervention."""
    observation = env.reset(
        object_position=task.object_position,
        goal_position=task.goal_position,
        target_radius=task.target_radius,
        seed=task.seed,
    )
    failure_reason: str | None = None
    success = False
    steps = 0
    initial_object = env.get_object_position()
    initial_goal_distance = float(
        np.linalg.norm(initial_object[:2] - task.goal_position[:2])
    )
    events = {
        "reaches_object_vicinity": False,
        "gripper_closes": False,
        "object_lifted": False,
        "transported_toward_goal": False,
        "reaches_goal_vicinity": False,
        "releases_after_close": False,
    }
    distance_trace = [initial_goal_distance]
    has_closed = False
    for steps in range(1, max_steps + 1):
        if not np.all(np.isfinite(observation)):
            failure_reason = "invalid_observation"
            break
        try:
            action = policy.act(observation)
        except (ValueError, RuntimeError):
            failure_reason = "invalid_policy_action"
            break
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            failure_reason = "invalid_policy_action"
            break
        observation, _ = env.step(action)
        if step_callback is not None:
            step_callback(steps, observation, action)
        if env.is_success():
            success = True
        object_position = env.get_object_position()
        ee_position = env.get_ee_position()
        goal_distance = float(
            np.linalg.norm(object_position[:2] - task.goal_position[:2])
        )
        distance_trace.append(goal_distance)
        events["reaches_object_vicinity"] |= bool(
            np.linalg.norm(ee_position - object_position) < 0.08
        )
        if action[3] > 0.0:
            has_closed = True
            events["gripper_closes"] = True
        if has_closed and action[3] <= 0.0:
            events["releases_after_close"] = True
        events["object_lifted"] |= bool(
            object_position[2] > initial_object[2] + 0.05
        )
        events["transported_toward_goal"] |= bool(
            goal_distance < initial_goal_distance - 0.05
        )
        events["reaches_goal_vicinity"] |= bool(
            goal_distance < max(task.target_radius * 2.0, 0.10)
        )
        if success:
            break
        if object_position[2] < -0.05:
            failure_reason = "object_left_workspace"
            break
    if not success and failure_reason is None:
        failure_reason = "timeout"
    final_object = env.get_object_position()
    xy_error = float(np.linalg.norm(final_object[:2] - task.goal_position[:2]))
    return BCRolloutResult(
        task_id=task.task_id,
        success=success,
        failure_reason=failure_reason,
        steps=steps,
        final_object_position=final_object.tolist(),
        final_object_goal_xy_error=xy_error,
        physical_events=events,
        object_goal_xy_distance_trace=distance_trace,
    )


def load_or_create_evaluation_tasks(
    path: str | Path,
    *,
    seed: int,
    count: int,
    dataset_fingerprint: str,
    demonstration_tasks: Sequence[dict[str, Any]],
) -> list[TaskConfig]:
    """Freeze or reload a separate reproducible set of task configurations."""
    destination = Path(path)
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload["dataset_fingerprint"] != dataset_fingerprint:
            raise ValueError("Evaluation-task dataset fingerprint mismatch")
        if int(payload["seed"]) != seed or int(payload["count"]) != count:
            raise ValueError("Existing evaluation task set has different settings")
        tasks = payload["tasks"]
    else:
        sampler = TaskSampler(seed)
        tasks = [sampler.sample(task_id=f"id_eval_{index:04d}").as_dict() for index in range(count)]
        demonstration_keys = {
            _task_key(
                summary["object_position"],
                summary["goal_position"],
                summary["target_radius"],
            )
            for summary in demonstration_tasks
        }
        if any(
            _task_key(task["object_position"], task["goal_position"], task["target_radius"])
            in demonstration_keys
            for task in tasks
        ):
            raise RuntimeError("Evaluation tasks overlap demonstration configurations")
        payload = {
            "version": 1,
            "dataset_fingerprint": dataset_fingerprint,
            "seed": seed,
            "count": count,
            "distribution": "M3 in-distribution task sampler",
            "tasks": tasks,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return [
        TaskConfig(
            task_id=task["task_id"],
            seed=int(task["seed"]),
            object_position=np.asarray(task["object_position"], dtype=np.float64),
            goal_position=np.asarray(task["goal_position"], dtype=np.float64),
            target_radius=float(task["target_radius"]),
        )
        for task in tasks
    ]


def _task_key(
    object_position: Sequence[float],
    goal_position: Sequence[float],
    target_radius: float,
) -> str:
    return json.dumps(
        [list(object_position), list(goal_position), float(target_radius)],
        separators=(",", ":"),
    )
