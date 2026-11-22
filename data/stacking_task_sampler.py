"""Deterministic sampler for the frozen Stacking V1 task distribution."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/stacking_v1.yaml"

@dataclass(frozen=True)
class StackingTaskConfig:
    task_id: str
    seed: int
    source_position: np.ndarray
    support_position: np.ndarray
    placement_tolerance: float
    def as_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "seed": self.seed,
                "source_position": self.source_position.tolist(),
                "support_position": self.support_position.tolist(),
                "placement_tolerance": self.placement_tolerance}

class StackingTaskSampler:
    def __init__(self, seed: int, config_path: str | Path = DEFAULT_CONFIG) -> None:
        config = yaml.safe_load(Path(config_path).read_text())
        self.config = config["task_randomization"]
        self.global_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.index = 0

    def sample(self, task_id: str | None = None) -> StackingTaskConfig:
        index = self.index; self.index += 1
        seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(seed); c = self.config
        source = np.array([rng.uniform(*c["source_x_range"]), rng.uniform(*c["source_y_range"]), c["cube_center_z"]])
        support = np.array([rng.uniform(*c["support_x_range"]), rng.uniform(*c["support_y_range"]), c["cube_center_z"]])
        tolerance = float(rng.uniform(*c["placement_tolerance_range"]))
        return StackingTaskConfig(task_id or f"stack_task_{index:06d}", seed, source, support, tolerance)
