"""Reproducible sampler for controlled pick-and-place task geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@dataclass(frozen=True)
class TaskConfig:
    """One fully specified deterministic pick-and-place task."""

    task_id: str
    seed: int
    object_position: np.ndarray
    goal_position: np.ndarray
    target_radius: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "object_position": self.object_position.tolist(),
            "goal_position": self.goal_position.tolist(),
            "target_radius": self.target_radius,
        }


class TaskSampler:
    """Sample tasks deterministically from explicit rectangular bounds."""

    def __init__(
        self,
        seed: int,
        config: Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise ValueError("Provide config or config_path, not both")
        if config is None:
            path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
            with path.open("r", encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file)
            config = loaded["task_randomization"]
            self._object_height = float(loaded["task"]["cube_position"][2])
            self._goal_height = float(loaded["task"]["goal_position"][2])
        else:
            config = dict(config)
            self._object_height = float(config.get("object_height", 0.025))
            self._goal_height = float(config.get("goal_height", 0.025))

        self.config = dict(config)
        self.global_seed = int(seed)
        self._rng = np.random.default_rng(self.global_seed)
        self._next_index = 0
        self._validate_config()

    def sample(self, task_id: str | None = None) -> TaskConfig:
        """Return the next task; sequence is fixed by ``global_seed``."""
        index = self._next_index
        self._next_index += 1
        sampled_seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        task_rng = np.random.default_rng(sampled_seed)

        for _ in range(10_000):
            object_position = np.array(
                [
                    task_rng.uniform(*self.config["object_x_range"]),
                    task_rng.uniform(*self.config["object_y_range"]),
                    self._object_height,
                ],
                dtype=np.float64,
            )
            goal_position = np.array(
                [
                    task_rng.uniform(*self.config["goal_x_range"]),
                    task_rng.uniform(*self.config["goal_y_range"]),
                    self._goal_height,
                ],
                dtype=np.float64,
            )
            if (
                np.linalg.norm(object_position[:2] - goal_position[:2])
                >= float(self.config["min_object_goal_distance"])
            ):
                break
        else:
            raise RuntimeError("Could not sample a task satisfying minimum distance")

        radius = float(task_rng.uniform(*self.config["target_radius_range"]))
        return TaskConfig(
            task_id=task_id or f"task_{index:06d}",
            seed=sampled_seed,
            object_position=object_position,
            goal_position=goal_position,
            target_radius=radius,
        )

    def _validate_config(self) -> None:
        for key in (
            "object_x_range",
            "object_y_range",
            "goal_x_range",
            "goal_y_range",
            "target_radius_range",
        ):
            bounds = np.asarray(self.config[key], dtype=float)
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
                raise ValueError(f"{key} must contain two finite values")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"{key} lower bound must be below upper bound")
        if float(self.config["min_object_goal_distance"]) <= 0.0:
            raise ValueError("min_object_goal_distance must be positive")
