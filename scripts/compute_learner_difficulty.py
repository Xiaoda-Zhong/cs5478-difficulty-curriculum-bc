#!/usr/bin/env python3
"""Train five fold-specific BC models and compute OOF learner difficulty."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.trajectory import load_trajectory, validate_trajectory  # noqa: E402
from difficulty.learner import (  # noqa: E402
    generate_oof_folds,
    learner_errors,
    split_fingerprint,
    validate_oof_folds,
)
from difficulty.normalization import (  # noqa: E402
    equal_weight_composite,
    fit_standardizer,
    standardize,
)
from models.bc_policy import load_policy_checkpoint  # noqa: E402
from training.bc_dataset import (  # noqa: E402
    compute_observation_statistics,
    load_transitions,
)
from training.train_bc import (  # noqa: E402
    select_device,
    train_fixed_epoch_behavior_cloning,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    bc = config["bc"]
    difficulty = config["difficulty"]
    device = select_device(args.device or difficulty["oof_device"])
    split = json.loads(
        (REPOSITORY_ROOT / "data/metadata/bc_split_v1.json").read_text(
            encoding="utf-8"
        )
    )
    trajectory_ids = split["train_trajectory_ids"]
    folds_path = REPOSITORY_ROOT / "data/metadata/learner_oof_folds_v1.json"
    folds_payload = _load_or_create_folds(
        folds_path,
        trajectory_ids,
        dataset_fingerprint=split["dataset_fingerprint"],
        fold_seed=int(difficulty["oof_fold_seed"]),
        number_of_folds=int(difficulty["oof_folds"]),
    )
    folds = folds_payload["folds"]
    validate_oof_folds(folds, trajectory_ids)

    demo_dir = REPOSITORY_ROOT / "data/demonstrations"
    checkpoint_dir = REPOSITORY_ROOT / "outputs/checkpoints/learner_oof"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, float | str | int]] = {}
    print(f"OOF device: {device}")

    for fold in folds:
        fold_index = int(fold["fold_index"])
        checkpoint_path = checkpoint_dir / f"fold_{fold_index}.pt"
        training_ids = fold["training_trajectory_ids"]
        held_out_ids = fold["held_out_trajectory_ids"]
        print(
            f"Fold {fold_index}: train={len(training_ids)}, "
            f"held_out={len(held_out_ids)}"
        )
        if args.overwrite or not checkpoint_path.exists():
            train_observations, train_actions = load_transitions(
                demo_dir, training_ids
            )
            observation_mean, observation_std = compute_observation_statistics(
                train_observations
            )
            fold_seed = int(difficulty["oof_fold_seed"]) + fold_index
            train_fixed_epoch_behavior_cloning(
                train_observations=train_observations,
                train_actions=train_actions,
                observation_mean=observation_mean,
                observation_std=observation_std,
                hidden_dims=bc["hidden_dims"],
                batch_size=int(bc["batch_size"]),
                learning_rate=float(bc["learning_rate"]),
                epochs=int(difficulty["oof_training_epochs"]),
                weight_decay=float(bc["weight_decay"]),
                lambda_gripper=float(bc["lambda_gripper"]),
                normalization_epsilon=float(bc["normalization_epsilon"]),
                seed=fold_seed,
                device=device,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata={
                    "dataset_fingerprint": split["dataset_fingerprint"],
                    "fold_index": fold_index,
                    "training_trajectory_ids": training_ids,
                    "held_out_trajectory_ids": held_out_ids,
                    "fixed_training_epochs": int(difficulty["oof_training_epochs"]),
                    "training_device": str(device),
                },
            )
        policy, checkpoint = load_policy_checkpoint(checkpoint_path, device)
        _validate_checkpoint(checkpoint, split["dataset_fingerprint"], fold)
        motion_scale = float(checkpoint["motion_scale"])
        for trajectory_id in held_out_ids:
            trajectory = load_trajectory(demo_dir / f"{trajectory_id}.npz")
            validate_trajectory(trajectory, require_success=True)
            observations = np.asarray(trajectory["observations"], dtype=np.float32)
            actions = np.asarray(trajectory["actions"], dtype=np.float32)
            predicted_motion, gripper_logits = policy.predict_batch_with_logits(
                observations
            )
            records[trajectory_id] = {
                "trajectory_id": trajectory_id,
                "oof_fold": fold_index,
                **learner_errors(
                    predicted_motion, gripper_logits, actions, motion_scale
                ),
            }

    if set(records) != set(trajectory_ids) or len(records) != len(trajectory_ids):
        raise RuntimeError("OOF scoring did not produce exactly one score per trajectory")
    ordered_records = [records[trajectory_id] for trajectory_id in trajectory_ids]
    components = np.asarray(
        [
            [record["learner_motion_error"], record["learner_gripper_error"]]
            for record in ordered_records
        ],
        dtype=np.float64,
    )
    combined_normalization = fit_standardizer(
        components,
        ["learner_motion_error", "learner_gripper_error"],
        float(difficulty["normalization_epsilon"]),
    )
    combined = equal_weight_composite(
        standardize(components, combined_normalization)
    )
    for record, combined_score in zip(ordered_records, combined, strict=True):
        record["learner_difficulty"] = record["learner_motion_error"]
        record["learner_combined_difficulty"] = float(combined_score)

    output = REPOSITORY_ROOT / "outputs/results/difficulty/learner_difficulty_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset_fingerprint": split["dataset_fingerprint"],
                "trajectory_ids": trajectory_ids,
                "definition": "OOF mean per-step squared error of motion normalized by fold training-only motion scale",
                "gripper_definition": "OOF mean BCE-with-logits per trajectory",
                "primary_score": "learner_motion_error",
                "combined_normalization": combined_normalization,
                "fold_assignment_file": str(folds_path.relative_to(REPOSITORY_ROOT)),
                "training_epochs": int(difficulty["oof_training_epochs"]),
                "training_device": str(device),
                "fold_seed": int(difficulty["oof_fold_seed"]),
                "number_of_folds": int(difficulty["oof_folds"]),
                "records": ordered_records,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Learner difficulty: scored {len(ordered_records)} trajectories OOF")
    print(f"Saved: {output}")


def _load_or_create_folds(
    path: Path,
    trajectory_ids: list[str],
    *,
    dataset_fingerprint: str,
    fold_seed: int,
    number_of_folds: int,
) -> dict:
    expected_split_fingerprint = split_fingerprint(trajectory_ids)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["dataset_fingerprint"] != dataset_fingerprint:
            raise ValueError("OOF fold dataset fingerprint mismatch")
        if payload["training_split_fingerprint"] != expected_split_fingerprint:
            raise ValueError("OOF fold training split fingerprint mismatch")
        if int(payload["fold_seed"]) != fold_seed:
            raise ValueError("OOF fold seed mismatch")
    else:
        payload = {
            "version": 1,
            "dataset_fingerprint": dataset_fingerprint,
            "bc_split_reference": "data/metadata/bc_split_v1.json",
            "training_split_fingerprint": expected_split_fingerprint,
            "fold_seed": fold_seed,
            "number_of_folds": number_of_folds,
            "folds": generate_oof_folds(
                trajectory_ids, number_of_folds, fold_seed
            ),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return payload


def _validate_checkpoint(checkpoint: dict, dataset_fingerprint: str, fold: dict) -> None:
    if checkpoint["dataset_fingerprint"] != dataset_fingerprint:
        raise ValueError("OOF checkpoint dataset fingerprint mismatch")
    if checkpoint["training_trajectory_ids"] != fold["training_trajectory_ids"]:
        raise ValueError("OOF checkpoint training IDs mismatch")
    if checkpoint["held_out_trajectory_ids"] != fold["held_out_trajectory_ids"]:
        raise ValueError("OOF checkpoint held-out IDs mismatch")
    if set(checkpoint["training_trajectory_ids"]) & set(
        checkpoint["held_out_trajectory_ids"]
    ):
        raise ValueError("OOF checkpoint contains train/held-out overlap")


if __name__ == "__main__":
    main()
