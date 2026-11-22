"""Deterministic learner-ranking corruption utilities for Milestone 9."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DATASET_FINGERPRINT = "1b6cec9dda067778630268c307241c62cccbd49cfe849a031877ee00ea938bc0"
RANKING_SEED = 85478
CORRUPTION_LEVELS = (25, 50, 100)
TRAINING_SEEDS = (5478, 5479, 5480)
GROUPS = ("easy", "medium", "hard")


def ordered_learner_ids(ranking_payload: Mapping[str, Any]) -> list[str]:
    groups = ranking_payload["rankings"]["learner"]["groups"]
    return [str(value) for group in GROUPS for value in groups[group]]


def corrupt_ranking(
    ordered_ids: Sequence[str], *, percentage: int, seed: int = RANKING_SEED,
) -> dict[str, Any]:
    """Permute values at a fixed percentage of rank positions."""
    if percentage not in CORRUPTION_LEVELS:
        raise ValueError(f"Unsupported corruption percentage: {percentage}")
    original = np.asarray([str(value) for value in ordered_ids])
    if len(original) != 240 or len(set(original.tolist())) != 240:
        raise ValueError("Learner ranking must contain 240 unique trajectory IDs")
    count = int(len(original) * percentage / 100)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(original), size=count, replace=False))
    replacement = original[selected][rng.permutation(count)]
    corrupted = original.copy(); corrupted[selected] = replacement
    groups = {
        "easy": corrupted[:80].tolist(),
        "medium": corrupted[80:160].tolist(),
        "hard": corrupted[160:].tolist(),
    }
    original_group = {trajectory_id: index // 80 for index, trajectory_id in enumerate(original)}
    corrupted_group = {trajectory_id: GROUPS.index(group) for group, ids in groups.items() for trajectory_id in ids}
    rank_changed = sum(a != b for a, b in zip(original.tolist(), corrupted.tolist()))
    group_changed = sum(original_group[value] != corrupted_group[value] for value in original)
    return {
        "ordered_trajectory_ids": corrupted.tolist(), "groups": groups,
        "selected_rank_positions_zero_based": selected.tolist(),
        "selected_trajectory_count": count,
        "rank_position_changes": rank_changed,
        "rank_position_change_fraction": rank_changed / len(original),
        "group_membership_changes": group_changed,
        "group_membership_change_fraction": group_changed / len(original),
    }


def validate_corrupted_payload(
    payload: Mapping[str, Any], original_ids: Sequence[str], *, percentage: int,
) -> None:
    if payload["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise ValueError("Corrupted ranking dataset fingerprint mismatch")
    if int(payload["ranking_seed"]) != RANKING_SEED or int(payload["corruption_percentage"]) != percentage:
        raise ValueError("Corrupted ranking identity mismatch")
    groups = payload["groups"]
    if set(groups) != set(GROUPS) or any(len(groups[group]) != 80 for group in GROUPS):
        raise ValueError("Corrupted ranking must contain three ordered groups of 80")
    flattened = [value for group in GROUPS for value in groups[group]]
    if len(flattened) != 240 or len(set(flattened)) != 240 or set(flattened) != set(original_ids):
        raise ValueError("Corrupted ranking has duplicate, missing, or unexpected IDs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
