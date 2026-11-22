"""Fast unit tests for Uniform Behavior Cloning components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.bc_policy import BCPolicy, BCPolicyAgent, load_policy_checkpoint
from evaluation.bc_diagnostics import compute_bc_diagnostics
from training.bc_dataset import (
    compute_observation_statistics,
    generate_trajectory_split,
)


def test_bc_model_forward_shapes_and_finite_outputs() -> None:
    model = BCPolicy(observation_dim=10, hidden_dims=(32, 16))
    motion, gripper = model(torch.zeros(7, 10))
    assert motion.shape == (7, 3)
    assert gripper.shape == (7, 1)
    assert torch.isfinite(motion).all()
    assert torch.isfinite(gripper).all()


def test_observation_normalization_statistics() -> None:
    observations = np.arange(50, dtype=np.float32).reshape(5, 10)
    mean, std = compute_observation_statistics(observations)
    normalized = (observations - mean) / (std + 1e-6)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-6)


def test_trajectory_split_is_deterministic_and_disjoint() -> None:
    trajectory_ids = [f"traj_{index:04d}" for index in range(300)]
    first = generate_trajectory_split(trajectory_ids, 0.8, 5478)
    second = generate_trajectory_split(trajectory_ids, 0.8, 5478)
    assert first == second
    train_ids, validation_ids = first
    assert len(train_ids) == 240
    assert len(validation_ids) == 60
    assert not set(train_ids) & set(validation_ids)
    assert set(train_ids) | set(validation_ids) == set(trajectory_ids)


def test_inference_clips_motion_and_thresholds_gripper() -> None:
    model = BCPolicy(observation_dim=10, hidden_dims=())
    with torch.no_grad():
        model.motion_head.weight.zero_()
        model.motion_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        model.gripper_head.weight.zero_()
        model.gripper_head.bias.fill_(0.0)
    agent = BCPolicyAgent(
        model,
        observation_mean=np.zeros(10),
        observation_std=np.ones(10),
        normalization_epsilon=1e-6,
        max_motion_norm=0.008,
    )
    action = agent.act(np.zeros(10, dtype=np.float32))
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))
    np.testing.assert_allclose(np.linalg.norm(action[:3]), 0.008, atol=1e-8)
    assert action[3] == 1.0

    with torch.no_grad():
        model.gripper_head.bias.fill_(-0.01)
    assert agent.act(np.zeros(10, dtype=np.float32))[3] == -1.0


def test_checkpoint_save_load_round_trip(tmp_path: Path) -> None:
    model = BCPolicy(observation_dim=10, hidden_dims=(16, 16))
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": {"observation_dim": 10, "hidden_dims": [16, 16]},
        "observation_mean": np.zeros(10).tolist(),
        "observation_std": np.ones(10).tolist(),
        "normalization_epsilon": 1e-6,
        "max_motion_norm": 0.008,
        "dataset_fingerprint": "test",
        "epoch": 1,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    agent, loaded = load_policy_checkpoint(path, "cpu")
    assert loaded["dataset_fingerprint"] == "test"
    action = agent.act(np.zeros(10, dtype=np.float32))
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))


def test_normalized_motion_output_is_rescaled() -> None:
    model = BCPolicy(observation_dim=10, hidden_dims=())
    with torch.no_grad():
        model.motion_head.weight.zero_()
        model.motion_head.bias.copy_(torch.tensor([0.5, 0.0, 0.0]))
        model.gripper_head.weight.zero_()
        model.gripper_head.bias.zero_()
    agent = BCPolicyAgent(
        model,
        observation_mean=np.zeros(10),
        observation_std=np.ones(10),
        normalization_epsilon=1e-6,
        max_motion_norm=0.008,
        motion_output_scale=0.008,
    )
    np.testing.assert_allclose(
        agent.act(np.zeros(10, dtype=np.float32))[:3], [0.004, 0.0, 0.0]
    )


def test_diagnostics_report_per_phase_and_gripper_metrics() -> None:
    model = BCPolicy(observation_dim=10, hidden_dims=())
    with torch.no_grad():
        model.motion_head.weight.zero_()
        model.motion_head.bias.zero_()
        model.gripper_head.weight.zero_()
        model.gripper_head.bias.zero_()
    agent = BCPolicyAgent(
        model,
        observation_mean=np.zeros(10),
        observation_std=np.ones(10),
        normalization_epsilon=1e-6,
        max_motion_norm=0.008,
    )
    observations = np.zeros((9, 10), dtype=np.float32)
    actions = np.zeros((9, 4), dtype=np.float32)
    actions[:5, 3] = 1.0
    actions[5:, 3] = -1.0
    phases = np.asarray(
        ["RAISE", "APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "RETREAT"]
    )
    diagnostics = compute_bc_diagnostics(
        agent, observations, actions, phases, motion_scale=0.008
    )
    assert diagnostics["motion"]["mean_l2_error"] == 0.0
    assert diagnostics["per_phase"]["TRANSPORT"]["transitions"] == 1
    assert diagnostics["gripper"]["close_targets"] == 5
    assert diagnostics["gripper"]["open_targets"] == 4
