"""Geometry-based Cartesian waypoints for the scripted expert."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpertWaypoints:
    """Named end-effector targets, expressed in world-frame metres."""

    safe_current: np.ndarray
    above_object: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    above_goal: np.ndarray
    place: np.ndarray
    retreat: np.ndarray


def build_waypoints(
    current_ee_position: np.ndarray,
    object_position: np.ndarray,
    goal_position: np.ndarray,
    *,
    safe_height: float,
    grasp_height_offset: float,
    place_height_offset: float,
) -> ExpertWaypoints:
    """Build deterministic waypoints from the observed task geometry."""
    ee = _position(current_ee_position, "current_ee_position")
    obj = _position(object_position, "object_position")
    goal = _position(goal_position, "goal_position")
    if safe_height <= max(obj[2], goal[2]):
        raise ValueError("safe_height must be above the object and goal")

    safe_current = np.array([ee[0], ee[1], safe_height], dtype=np.float64)
    above_object = np.array([obj[0], obj[1], safe_height], dtype=np.float64)
    grasp = np.array(
        [obj[0], obj[1], obj[2] + grasp_height_offset], dtype=np.float64
    )
    lift = above_object.copy()
    above_goal = np.array([goal[0], goal[1], safe_height], dtype=np.float64)
    place = np.array(
        [goal[0], goal[1], goal[2] + place_height_offset], dtype=np.float64
    )
    retreat = above_goal.copy()
    return ExpertWaypoints(
        safe_current=safe_current,
        above_object=above_object,
        grasp=grasp,
        lift=lift,
        above_goal=above_goal,
        place=place,
        retreat=retreat,
    )


def _position(value: np.ndarray, name: str) -> np.ndarray:
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{name} must be a finite 3-D position")
    return position.copy()
