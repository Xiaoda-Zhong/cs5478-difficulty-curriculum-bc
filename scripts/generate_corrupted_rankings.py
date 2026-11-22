#!/usr/bin/env python3
"""Generate and freeze the three Milestone 9 learner-ranking corruptions."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ranking_robustness import (  # noqa: E402
    CORRUPTION_LEVELS, DATASET_FINGERPRINT, RANKING_SEED, corrupt_ranking,
    ordered_learner_ids, sha256_file, validate_corrupted_payload, write_json,
)


def main() -> None:
    original_path = ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json"
    original = json.loads(original_path.read_text())
    if original["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise SystemExit("Original learner ranking fingerprint mismatch")
    ordered = ordered_learner_ids(original)
    output_dir = ROOT / "data/metadata/ranking_robustness_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for percentage in CORRUPTION_LEVELS:
        destination = output_dir / f"learner_ranking_corruption_{percentage:03d}.json"
        corruption = corrupt_ranking(ordered, percentage=percentage)
        payload = {
            "version": 1, "dataset_fingerprint": DATASET_FINGERPRINT,
            "source_ranking": str(original_path.relative_to(ROOT)),
            "source_ranking_sha256": sha256_file(original_path),
            "ranking_seed": RANKING_SEED, "corruption_percentage": percentage,
            "corruption_definition": "uniformly select the stated fraction of rank positions and randomly permute the trajectory IDs among those positions",
            "numeric_difficulty_scores_modified": False, **corruption,
        }
        validate_corrupted_payload(payload, ordered, percentage=percentage)
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing != payload:
                raise SystemExit(f"Refusing to change frozen corrupted ranking: {destination}")
        else:
            write_json(destination, payload)
        records.append({
            "corruption_percentage": percentage,
            "path": str(destination.relative_to(ROOT)), "sha256": sha256_file(destination),
            "selected_trajectory_count": payload["selected_trajectory_count"],
            "rank_position_changes": payload["rank_position_changes"],
            "group_membership_changes": payload["group_membership_changes"],
            "group_membership_change_fraction": payload["group_membership_change_fraction"],
        })
    manifest_path = output_dir / "ranking_corruption_manifest_v1.json"
    manifest = {
        "version": 1, "date_frozen_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": DATASET_FINGERPRINT, "ranking_seed": RANKING_SEED,
        "source_ranking_sha256": sha256_file(original_path),
        "policy_results_observed_before_freeze": False, "rankings": records,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text()); manifest["date_frozen_utc"] = old["date_frozen_utc"]
        if old != manifest:
            raise SystemExit("Refusing to change frozen corruption manifest")
    else:
        write_json(manifest_path, manifest)
    for row in records:
        print(f"{row['corruption_percentage']}%: selected={row['selected_trajectory_count']}, "
              f"rank-changed={row['rank_position_changes']}, group-changed={row['group_membership_changes']} "
              f"({row['group_membership_change_fraction']:.1%})")


if __name__ == "__main__":
    main()
