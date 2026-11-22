"""Fixed-update, resumable trainer shared by all curriculum strategies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from models.bc_policy import BCPolicy
from training.curriculum_sampler import CurriculumTransitionSampler, stage_for_update
from training.train_bc import seed_everything


def state_dict_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def create_initial_state(seed: int, hidden_dims: Sequence[int], observation_dim: int = 10) -> dict[str, torch.Tensor]:
    seed_everything(seed)
    model = BCPolicy(observation_dim=observation_dim, hidden_dims=hidden_dims)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def checkpoint_steps(total_updates: int, fractions: Sequence[float]) -> list[int]:
    steps = [int(round(total_updates * float(fraction))) for fraction in fractions]
    if any(not 1 <= step <= total_updates for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("Evaluation checkpoint fractions produce invalid steps")
    if steps[-1] != total_updates:
        raise ValueError("Evaluation checkpoints must include the final update")
    return steps


def validate_training_history(history: Mapping[str, Any]) -> None:
    """Validate required fields and monotonic checkpoint records."""
    required = {
        "method", "training_seed", "total_optimizer_updates", "batch_size",
        "learning_rate", "stage_boundaries", "evaluation_steps",
        "initial_state_fingerprint", "records",
    }
    if not required <= set(history):
        raise ValueError("Training history is missing required fields")
    records = list(history["records"])
    if [int(record["optimizer_step"]) for record in records] != list(history["evaluation_steps"]):
        raise ValueError("Training history steps do not match evaluation steps")
    required_record = {
        "optimizer_step", "curriculum_stage", "eligible_trajectory_count",
        "eligible_transition_count", "training_total_loss", "training_motion_mse",
        "training_gripper_loss", "validation_total_loss", "validation_motion_mse",
        "validation_gripper_accuracy", "sampled_group_composition",
    }
    for record in records:
        if not required_record <= set(record):
            raise ValueError("Training record is missing required fields")
        numeric = [
            record[key] for key in (
                "training_total_loss", "training_motion_mse", "training_gripper_loss",
                "validation_total_loss", "validation_motion_mse",
                "validation_gripper_accuracy",
            )
        ]
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Training history contains non-finite metrics")


def train_curriculum_run(
    *,
    method: str,
    seed: int,
    observations: np.ndarray,
    actions: np.ndarray,
    transition_trajectory_ids: np.ndarray,
    validation_observations: np.ndarray,
    validation_actions: np.ndarray,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    initial_state: Mapping[str, torch.Tensor],
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    motion_scale: float,
    hidden_dims: Sequence[int],
    total_updates: int,
    stage_boundaries: Sequence[float],
    evaluation_steps: Sequence[int],
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    lambda_gripper: float,
    normalization_epsilon: float,
    resume_interval: int,
    device: torch.device,
    output_dir: str | Path,
    dataset_fingerprint: str,
    ranking_reference: str,
    ranking_fingerprint: str,
    overwrite: bool = False,
    feature_transform: str | None = None,
    phase_critical_oversampling_factor: int = 1,
) -> dict[str, Any]:
    """Train one method/seed with an identical fixed optimizer-step budget."""
    destination = Path(output_dir)
    final_path = destination / "final.pt"
    history_path = destination / "training_history.json"
    latest_path = destination / "latest_state.pt"
    if final_path.exists() and history_path.exists() and not overwrite:
        return json.loads(history_path.read_text(encoding="utf-8"))
    if overwrite:
        for path in destination.glob("*.pt"):
            path.unlink()
        if history_path.exists():
            history_path.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    if set(training_ids) & set(validation_ids):
        raise ValueError("Training and validation trajectory IDs overlap")

    observation_dim=int(observations.shape[1])
    model = BCPolicy(observation_dim=observation_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sampler = CurriculumTransitionSampler(
        transition_trajectory_ids, training_ids, rankings, method, seed
    )
    mean = torch.as_tensor(observation_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(observation_std, dtype=torch.float32, device=device)
    obs_tensor = torch.from_numpy(np.asarray(observations, dtype=np.float32))
    action_tensor = torch.from_numpy(np.asarray(actions, dtype=np.float32))
    motion_loss_function = nn.MSELoss()
    gripper_loss_function = nn.BCEWithLogitsLoss()
    initial_fingerprint = state_dict_fingerprint(initial_state)
    history: dict[str, Any] = {
        "method": method, "training_seed": seed, "total_optimizer_updates": total_updates,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "lambda_gripper": lambda_gripper,
        "stage_boundaries": list(stage_boundaries), "evaluation_steps": list(evaluation_steps),
        "initial_state_fingerprint": initial_fingerprint, "records": [],
    }
    start_step = 1
    resume_state: dict[str, Any] | None = None
    if latest_path.exists() and not overwrite:
        resume_state = torch.load(latest_path, map_location=device, weights_only=False)
        if resume_state["initial_state_fingerprint"] != initial_fingerprint:
            raise ValueError("Resume state uses different initial weights")
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        sampler.set_rng_state(resume_state["sampler_rng_state"])
        history = resume_state["history"]
        start_step = int(resume_state["optimizer_step"]) + 1

    ranking_group_maps = {
        ranking: {trajectory_id: group for group, ids in groups.items() for trajectory_id in ids}
        for ranking, groups in rankings.items()
    }
    group_code = {"easy": 0, "medium": 1, "hard": 2}
    transition_group_codes = {
        ranking: np.asarray(
            [group_code[group_map[str(trajectory_id)]] for trajectory_id in transition_trajectory_ids],
            dtype=np.int8,
        )
        for ranking, group_map in ranking_group_maps.items()
    }
    interval_loss = float(resume_state.get("interval_loss", 0.0)) if resume_state else 0.0
    interval_motion = float(resume_state.get("interval_motion", 0.0)) if resume_state else 0.0
    interval_gripper = float(resume_state.get("interval_gripper", 0.0)) if resume_state else 0.0
    interval_examples = int(resume_state.get("interval_examples", 0)) if resume_state else 0
    sampled_composition = resume_state.get("sampled_composition") if resume_state else {
        ranking: {group: 0 for group in ("easy", "medium", "hard")}
        for ranking in rankings
    }
    best_validation = min(
        (record["validation_total_loss"] for record in history["records"]),
        default=float("inf"),
    )
    for step in range(start_step, total_updates + 1):
        stage = stage_for_update(step, total_updates, stage_boundaries)
        sampled = sampler.sample(batch_size, stage)
        batch_observations = obs_tensor[sampled.indices].to(device)
        batch_actions = action_tensor[sampled.indices].to(device)
        normalized = (batch_observations - mean) / (std + normalization_epsilon)
        predicted_motion, gripper_logits = model(normalized)
        motion_targets = batch_actions[:, :3] / motion_scale
        motion_loss = motion_loss_function(predicted_motion, motion_targets)
        gripper_targets = (batch_actions[:, 3] > 0.0).float().unsqueeze(1)
        gripper_loss = gripper_loss_function(gripper_logits, gripper_targets)
        total_loss = motion_loss + lambda_gripper * gripper_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        interval_examples += batch_size
        interval_loss += float(total_loss.item()) * batch_size
        interval_motion += float(motion_loss.item()) * batch_size
        interval_gripper += float(gripper_loss.item()) * batch_size
        for ranking, codes in transition_group_codes.items():
            counts = np.bincount(codes[sampled.indices], minlength=3)
            for group, code in group_code.items():
                sampled_composition[ranking][group] += int(counts[code])

        if step in evaluation_steps:
            validation = evaluate_validation(
                model, validation_observations, validation_actions, observation_mean,
                observation_std, normalization_epsilon, motion_scale, lambda_gripper,
                batch_size, device,
            )
            record = {
                "optimizer_step": step, "normalized_progress": step / total_updates,
                "curriculum_stage": stage,
                "eligible_trajectory_count": sampled.eligible_trajectory_count,
                "eligible_transition_count": sampled.eligible_transition_count,
                "training_total_loss": interval_loss / interval_examples,
                "training_motion_mse": interval_motion / interval_examples,
                "training_gripper_loss": interval_gripper / interval_examples,
                **validation, "sampled_group_composition": sampled_composition,
            }
            history["records"].append(record)
            checkpoint = _inference_checkpoint(
                model, method, seed, step, stage, validation, hidden_dims,
                observation_mean, observation_std, normalization_epsilon, motion_scale,
                dataset_fingerprint, ranking_reference, ranking_fingerprint,
                initial_fingerprint, total_updates, batch_size, learning_rate, weight_decay,
                lambda_gripper, sampled.eligible_trajectory_count,
                sampled.eligible_transition_count,
                observation_dim, feature_transform, phase_critical_oversampling_factor,
            )
            step_path = destination / f"step_{step:06d}.pt"
            torch.save(checkpoint, step_path)
            if validation["validation_total_loss"] < best_validation:
                best_validation = validation["validation_total_loss"]
                torch.save(checkpoint, destination / "best.pt")
            if step == total_updates:
                torch.save(checkpoint, final_path)
            history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"{method} seed={seed} step={step}/{total_updates} stage={stage} "
                f"train={record['training_total_loss']:.6f} val={validation['validation_total_loss']:.6f}",
                flush=True,
            )
            interval_loss = interval_motion = interval_gripper = 0.0
            interval_examples = 0
            sampled_composition = {
                ranking: {group: 0 for group in ("easy", "medium", "hard")}
                for ranking in rankings
            }
        if step % resume_interval == 0 or step in evaluation_steps or step == total_updates:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sampler_rng_state": sampler.get_rng_state(),
                "optimizer_step": step, "history": history,
                "initial_state_fingerprint": initial_fingerprint,
                "interval_loss": interval_loss, "interval_motion": interval_motion,
                "interval_gripper": interval_gripper,
                "interval_examples": interval_examples,
                "sampled_composition": sampled_composition,
            }, latest_path)
    if not final_path.exists():
        raise RuntimeError("Curriculum run did not produce a final checkpoint")
    latest_path.unlink(missing_ok=True)
    validate_training_history(history)
    return history


def evaluate_validation(
    model: BCPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    epsilon: float,
    motion_scale: float,
    lambda_gripper: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "motion": 0.0, "correct": 0.0}
    count = len(observations)
    mean = torch.as_tensor(observation_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(observation_std, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, count, batch_size):
            obs = torch.from_numpy(np.asarray(observations[start:start + batch_size], dtype=np.float32)).to(device)
            action = torch.from_numpy(np.asarray(actions[start:start + batch_size], dtype=np.float32)).to(device)
            motion, logits = model((obs - mean) / (std + epsilon))
            targets = action[:, :3] / motion_scale
            motion_loss = torch.mean((motion - targets) ** 2)
            grip_targets = (action[:, 3] > 0.0).float().unsqueeze(1)
            grip_loss = nn.functional.binary_cross_entropy_with_logits(logits, grip_targets)
            size = len(obs)
            totals["loss"] += float((motion_loss + lambda_gripper * grip_loss).item()) * size
            totals["motion"] += float(motion_loss.item()) * size
            totals["correct"] += int(((torch.sigmoid(logits) >= 0.5) == grip_targets.bool()).sum().item())
    model.train()
    return {
        "validation_total_loss": totals["loss"] / count,
        "validation_motion_mse": totals["motion"] / count,
        "validation_gripper_accuracy": totals["correct"] / count,
    }


def _inference_checkpoint(
    model: BCPolicy, method: str, seed: int, step: int, stage: int,
    validation: dict[str, float], hidden_dims: Sequence[int],
    observation_mean: np.ndarray, observation_std: np.ndarray, epsilon: float,
    motion_scale: float, dataset_fingerprint: str, ranking_reference: str,
    ranking_fingerprint: str, initial_fingerprint: str, total_updates: int,
    batch_size: int, learning_rate: float, weight_decay: float,
    lambda_gripper: float, eligible_trajectories: int, eligible_transitions: int,
    observation_dim: int, feature_transform: str | None,
    phase_critical_oversampling_factor: int,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "architecture": {"observation_dim": observation_dim, "hidden_dims": list(hidden_dims)},
        "observation_mean": observation_mean.tolist(), "observation_std": observation_std.tolist(),
        "normalization_epsilon": epsilon, "max_motion_norm": motion_scale,
        "motion_scale": motion_scale, "motion_targets_normalized": True,
        "training_seed": seed, "epoch": step, "optimizer_step": step,
        "method_name": method, "active_curriculum_stage": stage,
        "validation_metrics": validation, "dataset_fingerprint": dataset_fingerprint,
        "ranking_reference": ranking_reference, "ranking_fingerprint": ranking_fingerprint,
        "initial_state_fingerprint": initial_fingerprint,
        "training_configuration": {
            "total_optimizer_updates": total_updates, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "lambda_gripper": lambda_gripper, "optimizer": "AdamW",
            "sampling": "uniform transitions with replacement",
        },
        "eligible_trajectory_count": eligible_trajectories,
        "eligible_transition_count": eligible_transitions,
        "feature_transform": feature_transform,
        "phase_critical_oversampling_factor": phase_critical_oversampling_factor,
    }
