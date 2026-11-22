#!/usr/bin/env python3
"""Train the five matched fixed-budget curriculum BC strategies."""

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.trajectory import load_trajectory  # noqa: E402
from training.bc_dataset import compute_observation_statistics, load_transitions  # noqa: E402
from training.curriculum_sampler import METHODS  # noqa: E402
from training.train_bc import select_device  # noqa: E402
from training.train_curriculum import (  # noqa: E402
    checkpoint_steps, create_initial_state, state_dict_fingerprint,
    train_curriculum_run,
)

DATASET_FINGERPRINT = "1b6cec9dda067778630268c307241c62cccbd49cfe849a031877ee00ea938bc0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    bc, curriculum = config["bc"], config["curriculum"]
    methods = args.methods or list(curriculum["methods"])
    seeds = args.seeds or list(curriculum["seeds"])
    if args.smoke:
        seeds = [seeds[0]]
    device = select_device(args.device)
    split = _read_json(REPOSITORY_ROOT / "data/metadata/bc_split_v1.json")
    rankings_payload = _read_json(REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json")
    if {split["dataset_fingerprint"], rankings_payload["dataset_fingerprint"]} != {DATASET_FINGERPRINT}:
        raise SystemExit("Frozen dataset/ranking fingerprint mismatch")
    rankings = {name: value["groups"] for name, value in rankings_payload["rankings"].items()}
    demo_dir = REPOSITORY_ROOT / "data/demonstrations"
    train_obs, train_actions = load_transitions(demo_dir, split["train_trajectory_ids"])
    validation_obs, validation_actions = load_transitions(demo_dir, split["validation_trajectory_ids"])
    transition_ids = np.concatenate([
        np.repeat(trajectory_id, int(load_trajectory(demo_dir / f"{trajectory_id}.npz")["trajectory_length"]))
        for trajectory_id in split["train_trajectory_ids"]
    ])
    observation_mean, observation_std = compute_observation_statistics(train_obs)
    motion_scale = float(np.linalg.norm(train_actions[:, :3], axis=1).max())
    _validate_reference_preprocessing(observation_mean, observation_std, motion_scale)
    batch_size = int(curriculum["batch_size"])
    calculated_updates = int(curriculum["full_dataset_equivalent_epochs"]) * math.ceil(len(train_obs) / batch_size)
    configured_updates = int(curriculum["total_optimizer_updates"])
    if calculated_updates != configured_updates:
        raise ValueError(f"Configured update budget {configured_updates} != calculated {calculated_updates}")
    total_updates = 60 if args.smoke else configured_updates
    eval_steps = [20, 40, 60] if args.smoke else checkpoint_steps(
        total_updates, curriculum["evaluation_checkpoint_fractions"]
    )
    root_name = "curriculum_v1_smoke" if args.smoke else "curriculum_v1"
    checkpoint_root = REPOSITORY_ROOT / "outputs/checkpoints" / root_name
    result_root = REPOSITORY_ROOT / "outputs/results" / root_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    ranking_path = REPOSITORY_ROOT / "outputs/results/difficulty/difficulty_rankings_v1.json"
    ranking_fingerprint = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
    print(f"Device: {device}; transitions={len(train_obs)}; updates/run={total_updates}")
    for seed in seeds:
        initial_path = checkpoint_root / f"initial_seed_{seed}.pt"
        generated = create_initial_state(seed, bc["hidden_dims"])
        generated_fingerprint = state_dict_fingerprint(generated)
        if initial_path.exists():
            payload = torch.load(initial_path, map_location="cpu", weights_only=False)
            if payload["fingerprint"] != generated_fingerprint:
                raise ValueError(f"Initial state mismatch for seed {seed}")
            initial_state = payload["model_state_dict"]
        else:
            initial_state = generated
            torch.save({"training_seed": seed, "model_state_dict": initial_state, "fingerprint": generated_fingerprint}, initial_path)
        initial_eval_path = checkpoint_root / f"initial_eval_seed_{seed}.pt"
        if not initial_eval_path.exists():
            torch.save({
                "model_state_dict": initial_state,
                "architecture": {"observation_dim": 10, "hidden_dims": list(bc["hidden_dims"])},
                "observation_mean": observation_mean.tolist(),
                "observation_std": observation_std.tolist(),
                "normalization_epsilon": float(bc["normalization_epsilon"]),
                "max_motion_norm": motion_scale, "motion_scale": motion_scale,
                "motion_targets_normalized": True, "training_seed": seed,
                "epoch": 0, "optimizer_step": 0, "method_name": "initial",
                "active_curriculum_stage": 0, "dataset_fingerprint": DATASET_FINGERPRINT,
                "initial_state_fingerprint": generated_fingerprint,
            }, initial_eval_path)
        for method in methods:
            run_dir = checkpoint_root / method / f"seed_{seed}"
            print(f"Starting {method}, seed {seed}; init={generated_fingerprint[:12]}", flush=True)
            history = train_curriculum_run(
                method=method, seed=seed, observations=train_obs, actions=train_actions,
                transition_trajectory_ids=transition_ids,
                validation_observations=validation_obs, validation_actions=validation_actions,
                training_ids=split["train_trajectory_ids"], validation_ids=split["validation_trajectory_ids"],
                rankings=rankings, initial_state=initial_state,
                observation_mean=observation_mean, observation_std=observation_std,
                motion_scale=motion_scale, hidden_dims=bc["hidden_dims"],
                total_updates=total_updates, stage_boundaries=curriculum["stage_boundaries"],
                evaluation_steps=eval_steps, batch_size=batch_size,
                learning_rate=float(curriculum["learning_rate"]),
                weight_decay=float(curriculum["weight_decay"]),
                lambda_gripper=float(curriculum["lambda_gripper"]),
                normalization_epsilon=float(bc["normalization_epsilon"]),
                resume_interval=min(int(curriculum["resume_interval"]), total_updates),
                device=device, output_dir=run_dir,
                dataset_fingerprint=DATASET_FINGERPRINT,
                ranking_reference=str(ranking_path.relative_to(REPOSITORY_ROOT)),
                ranking_fingerprint=ranking_fingerprint, overwrite=args.overwrite,
            )
            result_dir = result_root / method / f"seed_{seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "training_history.json").write_text(
                json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


def _validate_reference_preprocessing(mean: np.ndarray, std: np.ndarray, motion_scale: float) -> None:
    checkpoint = torch.load(
        REPOSITORY_ROOT / "outputs/checkpoints/uniform_bc_scaled/best.pt",
        map_location="cpu", weights_only=False,
    )
    np.testing.assert_allclose(mean, checkpoint["observation_mean"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(std, checkpoint["observation_std"], atol=0.0, rtol=0.0)
    if motion_scale != float(checkpoint["motion_scale"]):
        raise ValueError("Curriculum motion scale differs from scaled BC reference")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
