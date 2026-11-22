#!/usr/bin/env python3
"""Reproducible S4.5 diagnostics for the frozen stacking Uniform-BC baseline."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.stacking_trajectory import load_stacking_trajectory, rollout_stacking_trajectory
from envs.panda_stacking import PandaStackingEnv
from expert.stacking_expert import ScriptedStackingExpert
from models.bc_policy import BCPolicyAgent, load_policy_checkpoint
from scripts.evaluate_stacking_bc import load_tasks


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out=np.empty_like(x,dtype=np.float64);positive=x>=0;out[positive]=1.0/(1.0+np.exp(-x[positive]));z=np.exp(x[~positive]);out[~positive]=z/(1.0+z);return out


def phase_metrics(policy: BCPolicyAgent, ids: list[str]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for trajectory_id in ids:
        t = load_stacking_trajectory(ROOT / "data/stacking_v1/demonstrations" / f"{trajectory_id}.npz")
        for phase in np.unique(t["phases"]):
            mask = t["phases"] == phase
            grouped[str(phase)].append((t["observations"][mask], t["actions"][mask]))
    result = {}
    for phase, chunks in grouped.items():
        obs = np.concatenate([x[0] for x in chunks]); actions = np.concatenate([x[1] for x in chunks])
        predicted, logits = policy.predict_batch_with_logits(obs)
        target = actions[:, :3].astype(np.float64)
        errors = np.linalg.norm(predicted - target, axis=1)
        target_norm = np.linalg.norm(target, axis=1); predicted_norm = np.linalg.norm(predicted, axis=1)
        moving = (target_norm > 1e-8) & (predicted_norm > 1e-8)
        cosine = np.sum(predicted[moving] * target[moving], axis=1) / (target_norm[moving] * predicted_norm[moving])
        grip = actions[:, 3] > 0
        result[phase] = {
            "transitions": int(len(obs)), "mean_motion_l2_m": float(errors.mean()),
            "direction_cosine": float(cosine.mean()) if len(cosine) else None,
            "gripper_accuracy": float(np.mean((_sigmoid(logits) >= .5) == grip)),
            "expert_motion_magnitude_m": float(target_norm.mean()),
            "predicted_motion_magnitude_m": float(predicted_norm.mean()),
        }
    return result


def feature_ranges(values: np.ndarray) -> dict[str, list[float]]:
    normalized_names = ["ee_x","ee_y","ee_z","source_x","source_y","source_z","source_up_x","source_up_y","source_up_z",
        "source_vx","source_vy","source_vz","source_wx","source_wy","source_wz","support_x","support_y","support_z",
        "support_up_x","support_up_y","support_up_z","support_vx","support_vy","support_vz","support_wx","support_wy","support_wz","gripper_opening","placement_tolerance"]
    return {name: [float(x) for x in np.percentile(values[:, i], [0, 1, 50, 99, 100])] for i, name in enumerate(normalized_names)}


def rollout_trace(env: PandaStackingEnv, policy: BCPolicyAgent, task, max_steps: int) -> tuple[dict, np.ndarray]:
    obs = env.reset(task.source_position, task.support_position, task.placement_tolerance, task.seed)
    rows=[]; observations=[]; ever_closed=False; contacts=0; switches=0; previous=None
    first_release=None; first_contact=None
    for step in range(max_steps):
        observations.append(obs.copy()); motion, logits = policy.predict_batch_with_logits(obs[None]); prob=float(_sigmoid(logits)[0]); close=prob >= .5
        if previous is not None and close != previous: switches += 1
        previous=close; ever_closed |= close
        action=np.r_[motion[0], 1.0 if close else -1.0]; obs,_=env.step(action)
        source=env.get_source_position(); support=env.get_support_position(); ee=env.get_ee_position(); d=env.get_stack_diagnostics()
        if d["source_support_contact"]: contacts += 1; first_contact = step if first_contact is None else first_contact
        if ever_closed and not close and first_release is None: first_release=step
        sv=env.get_source_velocity()[0]; tv=env.get_support_velocity()[0]
        rows.append({"step":step+1,"gripper_probability":prob,"gripper_command":int(close),"gripper_opening":d["gripper_opening"],
            "horizontal_error":d["horizontal_alignment_error"],"vertical_separation":d["vertical_separation"],"contact":d["source_support_contact"],
            "source_height":float(source[2]),"source_to_ee":(source-ee).tolist(),"ee_to_support":(ee-support).tolist(),
            "source_speed":float(np.linalg.norm(sv)),"support_speed":float(np.linalg.norm(tv)),"stable_steps":env.stable_steps})
        if env.is_success(): break
    release_row = rows[first_release] if first_release is not None else None
    summary={"task_id":task.task_id,"success":env.is_success(),"steps":len(rows),"first_release_step":None if first_release is None else first_release+1,
        "first_contact_step":None if first_contact is None else first_contact+1,"contact_steps":contacts,"command_switches":switches,
        "release_before_contact":bool(first_release is not None and (first_contact is None or first_release < first_contact)),"at_first_release":release_row,
        "final_diagnostics":env.get_stack_diagnostics(),"trace":rows}
    return summary, np.asarray(observations)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",type=Path,default=ROOT/"outputs/stacking_v1/checkpoints/uniform_bc/best.pt")
    ap.add_argument("--task-count",type=int,default=10); ap.add_argument("--output",type=Path,default=ROOT/"outputs/stacking_v1/results/s4_5/baseline_diagnostics.json"); args=ap.parse_args()
    metadata=json.loads((ROOT/"data/stacking_v1/metadata/dataset_metadata.json").read_text()); split=json.loads((ROOT/"data/stacking_v1/metadata/bc_split_v1.json").read_text())
    policy,checkpoint=load_policy_checkpoint(args.checkpoint,"cpu")
    if checkpoint["dataset_fingerprint"] != metadata["dataset_fingerprint"]: raise ValueError("checkpoint/dataset fingerprint mismatch")
    phases=phase_metrics(policy, split["validation_trajectory_ids"])
    train=[]
    for tid in split["train_trajectory_ids"]:
        train.append(load_stacking_trajectory(ROOT/"data/stacking_v1/demonstrations"/f"{tid}.npz")["observations"])
    train=np.concatenate(train)
    if hasattr(policy,"_transform"): train=policy._transform(train)
    train_normalized=(train-policy.observation_mean)/(policy.observation_std+policy.normalization_epsilon)
    cfg=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text()); tasks=load_tasks(metadata)[:args.task_count]
    traces=[]; rollout_obs=[]; expert_comparison=[]
    with PandaStackingEnv() as env:
        for task in tasks:
            trace,obs=rollout_trace(env,policy,task,int(cfg["bc"]["max_rollout_steps"])); traces.append(trace); rollout_obs.append(obs)
            expert=rollout_stacking_trajectory(env,ScriptedStackingExpert(),task,"diagnostic")
            expert_comparison.append({"task_id":task.task_id,"expert_success":bool(expert["success"]),"expert_steps":int(expert["trajectory_length"]),
                "bc_success":trace["success"],"bc_steps":trace["steps"],"expert_first_contact":int(np.flatnonzero([d["source_support_contact"] for d in json.loads("[]")])) if False else None})
    rollout=np.concatenate(rollout_obs)
    if hasattr(policy,"_transform"): rollout=policy._transform(rollout)
    rollout_normalized=(rollout-policy.observation_mean)/(policy.observation_std+policy.normalization_epsilon)
    payload={"dataset_fingerprint":metadata["dataset_fingerprint"],"checkpoint":str(args.checkpoint),"phase_metrics":phases,
        "normalization":{"epsilon":policy.normalization_epsilon,"std":policy.observation_std.tolist(),"expert_train_normalized_percentiles":feature_ranges(train_normalized),
            "bc_rollout_normalized_percentiles":feature_ranges(rollout_normalized)},"rollout_subset":{"task_indices":list(range(args.task_count)),"episodes":traces,"expert_comparison":expert_comparison}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(args.output),"phase_metrics":phases,"rollouts":[{k:v for k,v in t.items() if k!="trace"} for t in traces]},indent=2))

if __name__ == "__main__": main()
