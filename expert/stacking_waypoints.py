"""Deterministic geometry-derived waypoints for physical cube stacking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StackingWaypoints:
    safe_current: np.ndarray
    above_source: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray
    above_support: np.ndarray
    aligned: np.ndarray
    precontact: np.ndarray
    contact: np.ndarray
    retreat: np.ndarray


def build_stacking_waypoints(
    current_ee_position: np.ndarray,
    source_position: np.ndarray,
    support_position: np.ndarray,
    *,
    cube_edge_length: float,
    safe_height: float,
    grasp_height_offset: float,
    transport_clearance: float,
    precontact_clearance: float,
    contact_descent_margin: float,
) -> StackingWaypoints:
    """Build targets in world metres from current observed cube geometry."""
    ee = _position(current_ee_position, "current_ee_position")
    source = _position(source_position, "source_position")
    support = _position(support_position, "support_position")
    if cube_edge_length <= 0 or safe_height <= max(source[2], support[2]):
        raise ValueError("invalid cube edge length or safe height")
    grasp_offset = float(grasp_height_offset)
    source_stack_center_z = support[2] + cube_edge_length
    contact_ee_z = source_stack_center_z + grasp_offset
    transport_z = max(safe_height, contact_ee_z + transport_clearance)
    return StackingWaypoints(
        safe_current=np.array([ee[0], ee[1], safe_height]),
        above_source=np.array([source[0], source[1], safe_height]),
        grasp=np.array([source[0], source[1], source[2] + grasp_offset]),
        lift=np.array([source[0], source[1], transport_z]),
        above_support=np.array([support[0], support[1], transport_z]),
        aligned=np.array([support[0], support[1], transport_z]),
        precontact=np.array([support[0], support[1], contact_ee_z + precontact_clearance]),
        contact=np.array([support[0], support[1], contact_ee_z - contact_descent_margin]),
        retreat=np.array([support[0], support[1], safe_height]),
    )


def _position(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-D position")
    return result.copy()
