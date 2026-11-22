#!/usr/bin/env python3
"""Train scaled-motion Uniform BC on frozen Stacking Dataset V1."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_dataset import inspect_stacking_dataset,load_stacking_transitions
from models.bc_policy import load_policy_checkpoint
from training.bc_dataset import compute_observation_statistics
from training.train_bc import train_behavior_cloning,save_history,select_device

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default=None,choices=["cpu","mps","auto"]);ap.add_argument("--overwrite",action="store_true");args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text());bc=cfg["bc"];stats=inspect_stacking_dataset(ROOT/"data/stacking_v1")
    split=json.loads((ROOT/"data/stacking_v1/metadata/bc_split_v1.json").read_text());demo=ROOT/"data/stacking_v1/demonstrations"
    train_o,train_a=load_stacking_transitions(demo,split["train_trajectory_ids"]);val_o,val_a=load_stacking_transitions(demo,split["validation_trajectory_ids"])
    mean,std=compute_observation_statistics(train_o);ckpt=ROOT/"outputs/stacking_v1/checkpoints/uniform_bc";res=ROOT/"outputs/stacking_v1/results/uniform_bc"
    if (ckpt/"best.pt").exists() and not args.overwrite:raise SystemExit("Stacking checkpoint exists; pass --overwrite explicitly")
    ckpt.mkdir(parents=True,exist_ok=True);res.mkdir(parents=True,exist_ok=True)
    history,best=train_behavior_cloning(train_observations=train_o,train_actions=train_a,validation_observations=val_o,validation_actions=val_a,
        observation_mean=mean,observation_std=std,hidden_dims=bc["hidden_dims"],batch_size=bc["batch_size"],learning_rate=bc["learning_rate"],
        epochs=bc["max_epochs"],weight_decay=bc["weight_decay"],lambda_gripper=bc["lambda_gripper"],normalization_epsilon=bc["normalization_epsilon"],
        seed=bc["seed"],device=select_device(args.device or bc["device"]),checkpoint_dir=ckpt,
        checkpoint_metadata={"task_type":"object_stacking","dataset_fingerprint":stats["dataset_fingerprint"],"split_metadata":split,
            "train_transition_count":len(train_o),"validation_transition_count":len(val_o),"training_configuration":bc},
        normalize_motion_targets=True,early_stopping_patience=bc["early_stopping_patience"])
    save_history(res/"training_history.json",history)
    agent,checkpoint=load_policy_checkpoint(ckpt/"best.pt","cpu");pred,logits=agent.predict_batch_with_logits(val_o);scale=float(checkpoint["motion_scale"])
    targets=val_a[:,:3];l2=np.linalg.norm(pred-targets,axis=1);grip=(val_a[:,3]>0);prob=1/(1+np.exp(-logits));trans=np.flatnonzero(np.r_[False,grip[1:]!=grip[:-1]])
    diagnostics={"best_metrics":best,"metric_motion_mse":float(np.mean((pred-targets)**2)),"mean_motion_l2_error":float(l2.mean()),
        "relative_motion_error":float(l2.mean()/scale),"gripper_accuracy":float(np.mean((prob>=.5)==grip)),
        "gripper_transition_accuracy":float(np.mean(((prob[trans]>=.5)==grip[trans]))) if len(trans) else None,"motion_scale":scale,"parameter_count":best["parameter_count"]}
    (res/"best_metrics.json").write_text(json.dumps(diagnostics,indent=2,sort_keys=True)+"\n")
    print("Training complete",diagnostics)
if __name__=="__main__":main()
