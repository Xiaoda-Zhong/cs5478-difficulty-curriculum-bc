"""Deterministic transition-level sampling for controlled curricula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


METHODS = ("uniform", "task", "expert", "learner", "reverse_learner")


def stage_for_update(
    update: int, total_updates: int, stage_boundaries: Sequence[float]
) -> int:
    """Return stage 1, 2, or 3 for a one-indexed optimizer update."""
    if total_updates <= 0 or not 1 <= update <= total_updates:
        raise ValueError("update must be in [1, total_updates]")
    if len(stage_boundaries) != 2:
        raise ValueError("exactly two stage boundaries are required")
    first, second = (float(value) for value in stage_boundaries)
    if not 0.0 < first < second < 1.0:
        raise ValueError("stage boundaries must be ordered inside (0, 1)")
    first_last = int(np.floor(total_updates * first))
    second_last = int(np.floor(total_updates * second))
    return 1 if update <= first_last else 2 if update <= second_last else 3


def eligible_trajectory_ids(
    method: str,
    stage: int,
    all_training_ids: Sequence[str],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[str, ...]:
    """Return the predefined trajectory set for one method and stage."""
    if method not in METHODS:
        raise ValueError(f"Unknown curriculum method: {method}")
    if stage not in {1, 2, 3}:
        raise ValueError("stage must be 1, 2, or 3")
    all_ids = set(str(value) for value in all_training_ids)
    if method == "uniform" or stage == 3:
        return tuple(sorted(all_ids))
    ranking_name = "learner" if method == "reverse_learner" else method
    groups = rankings[ranking_name]
    first_group = "hard" if method == "reverse_learner" else "easy"
    names = [first_group] if stage == 1 else [first_group, "medium"]
    selected = {trajectory_id for name in names for trajectory_id in groups[name]}
    if not selected <= all_ids:
        raise ValueError("Ranking contains IDs outside the training split")
    return tuple(sorted(selected))


@dataclass(frozen=True)
class SampledBatch:
    indices: np.ndarray
    trajectory_ids: np.ndarray
    stage: int
    eligible_trajectory_count: int
    eligible_transition_count: int


class CurriculumTransitionSampler:
    """Uniformly sample transitions from the current eligible trajectory set."""

    def __init__(
        self,
        transition_trajectory_ids: Sequence[str],
        all_training_ids: Sequence[str],
        rankings: Mapping[str, Mapping[str, Sequence[str]]],
        method: str,
        seed: int,
    ) -> None:
        self.transition_ids = np.asarray(transition_trajectory_ids, dtype=str)
        self.all_training_ids = tuple(sorted(str(value) for value in all_training_ids))
        self.rankings = rankings
        self.method = method
        self.rng = np.random.default_rng(seed)
        if set(self.transition_ids.tolist()) != set(self.all_training_ids):
            raise ValueError("Transition IDs do not exactly cover the training split")
        self._indices_by_stage: dict[int, np.ndarray] = {}
        self._eligible_by_stage: dict[int, tuple[str, ...]] = {}
        for stage in (1, 2, 3):
            eligible = eligible_trajectory_ids(
                method, stage, self.all_training_ids, rankings
            )
            indices = np.flatnonzero(np.isin(self.transition_ids, eligible))
            if len(indices) == 0:
                raise ValueError("Curriculum stage has no eligible transitions")
            self._eligible_by_stage[stage] = eligible
            self._indices_by_stage[stage] = indices

    def sample(self, batch_size: int, stage: int) -> SampledBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pool = self._indices_by_stage[stage]
        selected = pool[self.rng.integers(0, len(pool), size=batch_size)]
        return SampledBatch(
            indices=selected,
            trajectory_ids=self.transition_ids[selected],
            stage=stage,
            eligible_trajectory_count=len(self._eligible_by_stage[stage]),
            eligible_transition_count=len(pool),
        )

    def get_rng_state(self) -> dict:
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state

    def eligible_ids(self, stage: int) -> tuple[str, ...]:
        return self._eligible_by_stage[stage]
