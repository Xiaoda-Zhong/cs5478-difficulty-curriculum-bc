#!/usr/bin/env python3
"""Compute scale-aware, per-phase, and gripper BC validation diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.bc_diagnostics import (  # noqa: E402
    PHASE_ORDER,
    compute_bc_diagnostics,
    load_validation_with_phases,
)
from models.bc_policy import load_policy_checkpoint  # noqa: E402
from training.train_bc import select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/checkpoints/uniform_bc/best.pt",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = select_device(args.device)
    policy, checkpoint = load_policy_checkpoint(args.checkpoint, device)
    split = checkpoint["split_metadata"]
    observations, actions, phases = load_validation_with_phases(
        REPOSITORY_ROOT / "data/demonstrations",
        split["validation_trajectory_ids"],
    )
    diagnostics = compute_bc_diagnostics(
        policy,
        observations,
        actions,
        phases,
        float(checkpoint["max_motion_norm"]),
    )
    output = args.output or (
        REPOSITORY_ROOT
        / "outputs/results"
        / args.checkpoint.parent.name
        / "validation_diagnostics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    motion = diagnostics["motion"]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Motion MSE: {motion['motion_mse']:.10f}")
    print(f"RMSE per component: {motion['rmse_per_component']}")
    print(f"Mean / median L2 error: {motion['mean_l2_error']:.6f} / {motion['median_l2_error']:.6f} m")
    print(f"Mean expert / predicted magnitude: {motion['mean_expert_action_magnitude']:.6f} / {motion['mean_predicted_action_magnitude']:.6f} m")
    print(f"Mean cosine similarity: {motion['mean_cosine_similarity']:.6f}")
    print(f"Relative motion error: {motion['relative_motion_error']:.3f}")
    print("\nPer-phase validation motion diagnostics")
    for phase in PHASE_ORDER:
        values = diagnostics["per_phase"][phase]
        cosine = values["mean_cosine_similarity"]
        cosine_text = "n/a" if cosine is None else f"{cosine:.4f}"
        print(
            f"{phase:9s} n={values['transitions']:5d} "
            f"mse={values['motion_mse']:.10f} "
            f"l2={values['mean_l2_error']:.6f} cos={cosine_text}"
        )
    gripper = diagnostics["gripper"]
    print("\nGripper diagnostics")
    print(f"Open / close targets: {gripper['open_targets']} / {gripper['close_targets']}")
    print(
        f"Accuracy={gripper['accuracy']:.6f} precision={gripper['precision']:.6f} "
        f"recall={gripper['recall']:.6f} F1={gripper['f1']:.6f}"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
