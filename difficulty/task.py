"""Task-geometry difficulty components."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


TASK_COMPONENT_NAMES = (
    "task_ee_object_distance",
    "task_object_goal_distance",
    "task_inverse_target_radius",
)


def task_components(trajectory: Mapping[str, Any]) -> dict[str, float]:
    initial_ee = np.asarray(trajectory["initial_ee_position"], dtype=np.float64)
    initial_object = np.asarray(
        trajectory["initial_object_position"], dtype=np.float64
    )
    goal = np.asarray(trajectory["goal_position"], dtype=np.float64)
    radius = float(np.asarray(trajectory["target_radius"]).item())
    if radius <= 0.0:
        raise ValueError("target radius must be positive")
    components = {
        "task_ee_object_distance": float(np.linalg.norm(initial_ee - initial_object)),
        "task_object_goal_distance": float(np.linalg.norm(initial_object - goal)),
        "task_inverse_target_radius": 1.0 / radius,
    }
    if not all(np.isfinite(value) for value in components.values()):
        raise ValueError("task difficulty components must be finite")
    return components
