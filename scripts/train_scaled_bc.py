#!/usr/bin/env python3
"""Train Uniform BC with training-only normalized Cartesian targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import yaml


matplotlib.use("Agg")


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_uniform_bc import _plot_history  # noqa: E402
from training.bc_dataset import (  # noqa: E402
    compute_observation_statistics,
    load_or_create_split,
    load_transitions,
)
from training.train_bc import save_history, select_device, train_behavior_cloning  # noqa: E402


DATASET_FINGERPRINT = "1b6cec9dda067778630268c307241c62cccbd49cfe849a031877ee00ea938bc0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bc = config["bc"]
    seed = int(args.seed if args.seed is not None else bc["seed"])
    device = select_device(args.device or bc["device"])

    metadata = json.loads(
        (REPOSITORY_ROOT / "data/metadata/dataset_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise SystemExit("Frozen Dataset V1 fingerprint mismatch; refusing to train")
    demo_dir = REPOSITORY_ROOT / "data/demonstrations"
    trajectory_ids = sorted(path.stem for path in demo_dir.glob("traj_*.npz"))
    split = load_or_create_split(
        REPOSITORY_ROOT / "data/metadata/bc_split_v1.json",
        trajectory_ids,
        dataset_fingerprint=DATASET_FINGERPRINT,
        split_seed=int(bc["split_seed"]),
        train_fraction=float(bc["train_fraction"]),
    )
    train_observations, train_actions = load_transitions(
        demo_dir, split["train_trajectory_ids"]
    )
    validation_observations, validation_actions = load_transitions(
        demo_dir, split["validation_trajectory_ids"]
    )
    observation_mean, observation_std = compute_observation_statistics(
        train_observations
    )
    run_name = "uniform_bc_scaled"
    checkpoint_dir = REPOSITORY_ROOT / "outputs/checkpoints" / run_name
    results_dir = REPOSITORY_ROOT / "outputs/results" / run_name
    figures_dir = REPOSITORY_ROOT / "outputs/figures" / run_name
    if (checkpoint_dir / "best.pt").exists() and not args.overwrite:
        raise SystemExit(
            f"Checkpoint already exists at {checkpoint_dir}; pass --overwrite explicitly"
        )
    for directory in (checkpoint_dir, results_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Trajectories: train={len(split['train_trajectory_ids'])}, "
        f"validation={len(split['validation_trajectory_ids'])}"
    )
    print(
        f"Transitions: train={len(train_observations)}, "
        f"validation={len(validation_observations)}"
    )
    history, best_metrics = train_behavior_cloning(
        train_observations=train_observations,
        train_actions=train_actions,
        validation_observations=validation_observations,
        validation_actions=validation_actions,
        observation_mean=observation_mean,
        observation_std=observation_std,
        hidden_dims=bc["hidden_dims"],
        batch_size=int(bc["batch_size"]),
        learning_rate=float(bc["learning_rate"]),
        epochs=int(bc["scaled_max_epochs"]),
        weight_decay=float(bc["weight_decay"]),
        lambda_gripper=float(bc["lambda_gripper"]),
        normalization_epsilon=float(bc["normalization_epsilon"]),
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_metadata={
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "split_metadata": split,
            "training_configuration": bc,
            "train_transition_count": len(train_observations),
            "validation_transition_count": len(validation_observations),
            "training_device": str(device),
        },
        normalize_motion_targets=True,
        early_stopping_patience=int(bc["early_stopping_patience"]),
    )
    save_history(results_dir / "training_history.json", history)
    (results_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_history(history, figures_dir)
    print(f"Best epoch: {best_metrics['epoch']}")
    print(f"Best normalized validation loss: {best_metrics['total_loss']:.8f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
