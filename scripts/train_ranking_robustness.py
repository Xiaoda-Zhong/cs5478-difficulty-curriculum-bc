#!/usr/bin/env python3
"""Train the nine new Reverse Learner ranking-corruption runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.trajectory import load_trajectory  # noqa: E402
from evaluation.ranking_robustness import (  # noqa: E402
    CORRUPTION_LEVELS, DATASET_FINGERPRINT, RANKING_SEED, TRAINING_SEEDS,
    ordered_learner_ids, sha256_file, validate_corrupted_payload,
)
from training.bc_dataset import compute_observation_statistics, load_transitions  # noqa: E402
from training.train_bc import select_device  # noqa: E402
from training.train_curriculum import (  # noqa: E402
    checkpoint_steps, create_initial_state, state_dict_fingerprint,
    train_curriculum_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, choices=CORRUPTION_LEVELS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    levels = tuple(args.levels or CORRUPTION_LEVELS)
    seeds = tuple(args.seeds or TRAINING_SEEDS)
    if args.smoke:
        levels, seeds = (levels[0],), (seeds[0],)
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    bc, curriculum = config["bc"], config["curriculum"]
    split = json.loads((ROOT / "data/metadata/bc_split_v1.json").read_text())
    original_path = ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json"
    original = json.loads(original_path.read_text())
    if {split["dataset_fingerprint"], original["dataset_fingerprint"]} != {DATASET_FINGERPRINT}:
        raise SystemExit("Dataset, split, or original ranking fingerprint mismatch")
    original_ids = ordered_learner_ids(original)
    train_ids = split["train_trajectory_ids"]; validation_ids = split["validation_trajectory_ids"]
    if set(original_ids) != set(train_ids) or set(train_ids) & set(validation_ids):
        raise ValueError("Learner ranking does not exactly match the leakage-free training split")
    demo_dir = ROOT / "data/demonstrations"
    train_obs, train_actions = load_transitions(demo_dir, train_ids)
    validation_obs, validation_actions = load_transitions(demo_dir, validation_ids)
    transition_ids = np.concatenate([
        np.repeat(trajectory_id, int(load_trajectory(demo_dir / f"{trajectory_id}.npz")["trajectory_length"]))
        for trajectory_id in train_ids
    ])
    observation_mean, observation_std = compute_observation_statistics(train_obs)
    motion_scale = float(np.linalg.norm(train_actions[:, :3], axis=1).max())
    reference = torch.load(ROOT / "outputs/checkpoints/uniform_bc_scaled/best.pt", map_location="cpu", weights_only=False)
    np.testing.assert_array_equal(observation_mean, reference["observation_mean"])
    np.testing.assert_array_equal(observation_std, reference["observation_std"])
    if motion_scale != float(reference["motion_scale"]):
        raise ValueError("Motion scale differs from M7 preprocessing")
    calculated = int(curriculum["full_dataset_equivalent_epochs"]) * math.ceil(len(train_obs) / int(curriculum["batch_size"]))
    if calculated != int(curriculum["total_optimizer_updates"]):
        raise ValueError("Optimizer-update budget no longer matches M7")
    total_updates = 60 if args.smoke else int(curriculum["total_optimizer_updates"])
    evaluation_steps = [20, 40, 60] if args.smoke else checkpoint_steps(total_updates, curriculum["evaluation_checkpoint_fractions"])
    checkpoint_root = ROOT / "outputs/checkpoints" / ("ranking_robustness_v1_smoke" if args.smoke else "ranking_robustness_v1")
    result_root = ROOT / "outputs/results" / ("ranking_robustness_v1_smoke" if args.smoke else "ranking_robustness_v1")
    device = select_device(args.device)
    print(f"Device: {device}; transitions={len(train_obs)}; updates/run={total_updates}; runs={len(levels) * len(seeds)}")
    for level in levels:
        ranking_path = ROOT / f"data/metadata/ranking_robustness_v1/learner_ranking_corruption_{level:03d}.json"
        ranking = json.loads(ranking_path.read_text())
        validate_corrupted_payload(ranking, original_ids, percentage=level)
        ranking_hash = sha256_file(ranking_path)
        rankings = {"learner": ranking["groups"]}
        for seed in seeds:
            initial_path = ROOT / f"outputs/checkpoints/curriculum_v1/initial_seed_{seed}.pt"
            initial = torch.load(initial_path, map_location="cpu", weights_only=False)
            generated_fingerprint = state_dict_fingerprint(create_initial_state(seed, bc["hidden_dims"]))
            loaded_fingerprint = state_dict_fingerprint(initial["model_state_dict"])
            if initial["fingerprint"] != generated_fingerprint or loaded_fingerprint != generated_fingerprint:
                raise ValueError(f"M7 initial-state mismatch for seed {seed}")
            run_dir = checkpoint_root / f"corruption_{level:03d}" / f"seed_{seed}"
            print(f"Starting corruption={level}% seed={seed}; init={loaded_fingerprint[:12]}", flush=True)
            history = train_curriculum_run(
                method="reverse_learner", seed=seed,
                observations=train_obs, actions=train_actions,
                transition_trajectory_ids=transition_ids,
                validation_observations=validation_obs, validation_actions=validation_actions,
                training_ids=train_ids, validation_ids=validation_ids,
                rankings=rankings, initial_state=initial["model_state_dict"],
                observation_mean=observation_mean, observation_std=observation_std,
                motion_scale=motion_scale, hidden_dims=bc["hidden_dims"],
                total_updates=total_updates, stage_boundaries=curriculum["stage_boundaries"],
                evaluation_steps=evaluation_steps, batch_size=int(curriculum["batch_size"]),
                learning_rate=float(curriculum["learning_rate"]),
                weight_decay=float(curriculum["weight_decay"]),
                lambda_gripper=float(curriculum["lambda_gripper"]),
                normalization_epsilon=float(bc["normalization_epsilon"]),
                resume_interval=min(int(curriculum["resume_interval"]), total_updates),
                device=device, output_dir=run_dir,
                dataset_fingerprint=DATASET_FINGERPRINT,
                ranking_reference=str(ranking_path.relative_to(ROOT)),
                ranking_fingerprint=ranking_hash, overwrite=args.overwrite,
            )
            annotated = dict(history)
            annotated.update({
                "experiment": "ranking_robustness_v1", "corruption_percentage": level,
                "ranking_seed": RANKING_SEED, "ranking_reference": str(ranking_path.relative_to(ROOT)),
                "ranking_fingerprint": ranking_hash, "source_initial_checkpoint": str(initial_path.relative_to(ROOT)),
            })
            destination = result_root / f"corruption_{level:03d}" / f"seed_{seed}" / "training_history.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(annotated, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
