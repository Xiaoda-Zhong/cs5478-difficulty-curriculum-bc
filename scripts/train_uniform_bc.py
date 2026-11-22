#!/usr/bin/env python3
"""Train the Uniform Behavior Cloning baseline on Dataset V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.bc_dataset import (  # noqa: E402
    compute_observation_statistics,
    load_or_create_split,
    load_transitions,
)
from training.train_bc import (  # noqa: E402
    save_history,
    select_device,
    train_behavior_cloning,
)


DATASET_FINGERPRINT = "1b6cec9dda067778630268c307241c62cccbd49cfe849a031877ee00ea938bc0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overfit-small",
        action="store_true",
        help="debug on the first 512 training transitions instead of full data",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bc_config = config["bc"]
    seed = int(args.seed if args.seed is not None else bc_config["seed"])
    device = select_device(args.device or bc_config["device"])
    epochs = int(args.epochs if args.epochs is not None else bc_config["epochs"])

    metadata_path = REPOSITORY_ROOT / "data/metadata/dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["dataset_fingerprint"] != DATASET_FINGERPRINT:
        raise SystemExit("Frozen Dataset V1 fingerprint mismatch; refusing to train")
    demonstration_dir = REPOSITORY_ROOT / "data/demonstrations"
    trajectory_ids = sorted(path.stem for path in demonstration_dir.glob("traj_*.npz"))
    split_path = REPOSITORY_ROOT / "data/metadata/bc_split_v1.json"
    split = load_or_create_split(
        split_path,
        trajectory_ids,
        dataset_fingerprint=DATASET_FINGERPRINT,
        split_seed=int(bc_config["split_seed"]),
        train_fraction=float(bc_config["train_fraction"]),
    )

    train_observations, train_actions = load_transitions(
        demonstration_dir, split["train_trajectory_ids"]
    )
    validation_observations, validation_actions = load_transitions(
        demonstration_dir, split["validation_trajectory_ids"]
    )
    if args.overfit_small:
        train_observations = train_observations[:512]
        train_actions = train_actions[:512]
        validation_observations = train_observations.copy()
        validation_actions = train_actions.copy()

    observation_mean, observation_std = compute_observation_statistics(
        train_observations
    )
    run_name = "uniform_bc_overfit" if args.overfit_small else "uniform_bc"
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
        hidden_dims=bc_config["hidden_dims"],
        batch_size=int(bc_config["batch_size"]),
        learning_rate=float(bc_config["learning_rate"]),
        epochs=epochs,
        weight_decay=float(bc_config["weight_decay"]),
        lambda_gripper=float(bc_config["lambda_gripper"]),
        normalization_epsilon=float(bc_config["normalization_epsilon"]),
        seed=seed,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_metadata={
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "split_metadata": split,
            "training_configuration": bc_config,
            "train_transition_count": len(train_observations),
            "validation_transition_count": len(validation_observations),
        },
    )
    save_history(results_dir / "training_history.json", history)
    (results_dir / "best_metrics.json").write_text(
        json.dumps(best_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_history(history, figures_dir)
    print(f"Best epoch: {best_metrics['epoch']}")
    print(f"Best validation loss: {best_metrics['total_loss']:.8f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")


def _plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_total_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, history["train_total_loss"], label="train")
    axis.plot(epochs, history["validation_total_loss"], label="validation")
    axis.set(xlabel="Epoch", ylabel="Total loss", title="Uniform BC loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_motion_mse"], label="train")
    axes[0].plot(epochs, history["validation_motion_mse"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="MSE", title="Motion")
    axes[1].plot(epochs, history["train_gripper_accuracy"], label="train")
    axes[1].plot(epochs, history["validation_gripper_accuracy"], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Gripper")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "component_metrics.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
