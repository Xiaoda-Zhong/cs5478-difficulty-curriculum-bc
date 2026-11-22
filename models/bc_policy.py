"""Small two-head MLP for state-based behavior cloning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


class BCPolicy(nn.Module):
    """Shared MLP encoder with separate motion and gripper heads."""

    def __init__(
        self,
        observation_dim: int = 10,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.motion_head = nn.Linear(input_dim, 3)
        self.gripper_head = nn.Linear(input_dim, 1)
        self.observation_dim = observation_dim
        self.hidden_dims = tuple(int(value) for value in hidden_dims)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        return self.motion_head(features), self.gripper_head(features)


class BCPolicyAgent:
    """Rollout wrapper applying training normalization and safe action conversion."""

    def __init__(
        self,
        model: BCPolicy,
        observation_mean: np.ndarray,
        observation_std: np.ndarray,
        normalization_epsilon: float,
        max_motion_norm: float,
        motion_output_scale: float = 1.0,
        normalization_clip: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.observation_mean = np.asarray(observation_mean, dtype=np.float32)
        self.observation_std = np.asarray(observation_std, dtype=np.float32)
        if self.observation_mean.shape != (model.observation_dim,):
            raise ValueError("observation_mean has the wrong shape")
        if self.observation_std.shape != (model.observation_dim,):
            raise ValueError("observation_std has the wrong shape")
        if not np.all(np.isfinite(self.observation_mean)) or not np.all(
            np.isfinite(self.observation_std)
        ):
            raise ValueError("normalization statistics must be finite")
        self.normalization_epsilon = float(normalization_epsilon)
        self.max_motion_norm = float(max_motion_norm)
        self.motion_output_scale = float(motion_output_scale)
        self.normalization_clip = None if normalization_clip is None else float(normalization_clip)
        if (
            self.normalization_epsilon <= 0.0
            or self.max_motion_norm <= 0.0
            or self.motion_output_scale <= 0.0
        ):
            raise ValueError("normalization epsilon and motion limit must be positive")

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (self.model.observation_dim,):
            raise ValueError(
                f"observation must have shape ({self.model.observation_dim},)"
            )
        if not np.all(np.isfinite(observation_array)):
            raise ValueError("observation must contain only finite values")
        motion_batch, gripper_probabilities = self.predict_batch(
            observation_array[None, :]
        )
        motion = motion_batch[0]
        close_probability = float(gripper_probabilities[0])
        gripper_command = 1.0 if close_probability >= 0.5 else -1.0
        return np.concatenate([motion, np.asarray([gripper_command])])

    def predict_batch(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict safely clipped metric motion and close probabilities."""
        motion, logits = self.predict_batch_with_logits(observations)
        # Numerically stable sigmoid: PyTorch avoids overflow for saturated logits.
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return motion, probabilities

    def predict_batch_with_logits(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict safely clipped metric motion and raw gripper logits."""
        observation_array = np.asarray(observations, dtype=np.float32)
        if observation_array.ndim != 2 or observation_array.shape[1] != self.model.observation_dim:
            raise ValueError(
                f"observations must have shape (N, {self.model.observation_dim})"
            )
        if not np.all(np.isfinite(observation_array)):
            raise ValueError("observations must contain only finite values")
        normalized = (observation_array - self.observation_mean) / (
            self.observation_std + self.normalization_epsilon
        )
        if self.normalization_clip is not None:
            normalized = np.clip(normalized, -self.normalization_clip, self.normalization_clip)
        tensor = torch.from_numpy(normalized).to(self.device)
        with torch.no_grad():
            motion_tensor, gripper_logit = self.model(tensor)
        motion = (
            motion_tensor.cpu().numpy().astype(np.float64) * self.motion_output_scale
        )
        norms = np.linalg.norm(motion, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_motion_norm / np.maximum(norms, 1e-12))
        motion *= scale
        logits = (
            gripper_logit.squeeze(1).cpu().numpy().astype(np.float64)
        )
        return motion, logits


def load_policy_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[BCPolicyAgent, dict[str, Any]]:
    """Load a checkpoint and construct its rollout-ready policy wrapper."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=torch.device(device), weights_only=False
    )
    architecture = checkpoint["architecture"]
    model = BCPolicy(
        observation_dim=int(architecture["observation_dim"]),
        hidden_dims=architecture["hidden_dims"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    agent = BCPolicyAgent(
        model=model,
        observation_mean=np.asarray(checkpoint["observation_mean"], dtype=np.float32),
        observation_std=np.asarray(checkpoint["observation_std"], dtype=np.float32),
        normalization_epsilon=float(checkpoint["normalization_epsilon"]),
        max_motion_norm=float(checkpoint["max_motion_norm"]),
        motion_output_scale=float(
            checkpoint.get("motion_scale", 1.0)
            if checkpoint.get("motion_targets_normalized", False)
            else 1.0
        ),
        device=device,
        normalization_clip=checkpoint.get("normalization_clip"),
    )
    feature_transform = checkpoint.get("feature_transform")
    if feature_transform:
        from models.stacking_features import transform_stacking_observations
        agent = FeatureTransformPolicyAgent(agent, str(feature_transform))
    return agent, checkpoint


class FeatureTransformPolicyAgent:
    """Apply a checkpoint-declared, observable-only transform before inference."""
    def __init__(self, base_agent: BCPolicyAgent, feature_transform: str) -> None:
        self.base_agent=base_agent; self.feature_transform=feature_transform
        for name in ("model","observation_mean","observation_std","normalization_epsilon","max_motion_norm","motion_output_scale"):
            setattr(self,name,getattr(base_agent,name))
    def _transform(self, observations: np.ndarray) -> np.ndarray:
        from models.stacking_features import transform_stacking_observations
        return transform_stacking_observations(observations,self.feature_transform)
    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.base_agent.act(self._transform(np.asarray(observation)))
    def predict_batch(self, observations: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        return self.base_agent.predict_batch(self._transform(observations))
    def predict_batch_with_logits(self, observations: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        return self.base_agent.predict_batch_with_logits(self._transform(observations))


class DebouncedGripperPolicyAgent:
    """Require persistent output evidence before changing binary gripper state."""
    def __init__(self, base_agent: Any, confirmation_steps: int = 3) -> None:
        if confirmation_steps < 1: raise ValueError("confirmation_steps must be positive")
        self.base_agent=base_agent;self.confirmation_steps=confirmation_steps;self.reset()
    def reset(self) -> None:
        self._closed=False;self._candidate=False;self._candidate_steps=0
    def act(self, observation: np.ndarray) -> np.ndarray:
        motion,prob=self.base_agent.predict_batch(np.asarray(observation,dtype=np.float32)[None]);candidate=bool(prob[0]>=.5)
        if candidate==self._candidate:self._candidate_steps+=1
        else:self._candidate=candidate;self._candidate_steps=1
        if candidate!=self._closed and self._candidate_steps>=self.confirmation_steps:self._closed=candidate
        return np.r_[motion[0],1.0 if self._closed else -1.0]
    def predict_batch(self, observations: np.ndarray): return self.base_agent.predict_batch(observations)
    def predict_batch_with_logits(self, observations: np.ndarray): return self.base_agent.predict_batch_with_logits(observations)
