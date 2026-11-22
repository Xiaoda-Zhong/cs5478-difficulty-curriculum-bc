"""Scripted-expert effort difficulty components."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


EXPERT_COMPONENT_NAMES = (
    "expert_duration",
    "expert_ee_path_length",
    "expert_action_magnitude",
)


def expert_components(trajectory: Mapping[str, Any]) -> dict[str, float]:
    actions = np.asarray(trajectory["actions"], dtype=np.float64)
    ee_positions = np.asarray(trajectory["ee_positions"], dtype=np.float64)
    duration = int(np.asarray(trajectory["trajectory_length"]).item())
    if actions.shape != (duration, 4) or ee_positions.shape != (duration + 1, 3):
        raise ValueError("trajectory arrays do not match trajectory_length")
    components = {
        "expert_duration": float(duration),
        "expert_ee_path_length": float(
            np.linalg.norm(np.diff(ee_positions, axis=0), axis=1).sum()
        ),
        "expert_action_magnitude": float(
            np.linalg.norm(actions[:, :3], axis=1).sum()
        ),
    }
    if not all(np.isfinite(value) for value in components.values()):
        raise ValueError("expert difficulty components must be finite")
    return components
