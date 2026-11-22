#!/usr/bin/env python3
"""Train isolated S4.5 stacking BC variants without touching S4 artifacts."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_dataset import inspect_stacking_dataset,load_stacking_transitions
from data.stacking_trajectory import load_stacking_trajectory
from models.stacking_features import RELATIVE_V1,transform_stacking_observations
from training.bc_dataset import compute_observation_statistics
from training.train_bc import train_behavior_cloning,save_history,select_device

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--variant",choices=["relative_v1","phase_balanced_v1","relative_phase_balanced_v1"],required=True);ap.add_argument("--device",default="cpu",choices=["cpu","mps","auto"]);args=ap.parse_args()
    out_name=args.variant; ckpt=ROOT/f"outputs/stacking_v1/checkpoints/s4_5/{out_name}"; res=ROOT/f"outputs/stacking_v1/results/s4_5/{out_name}"
    if (ckpt/"best.pt").exists(): raise SystemExit(f"{ckpt}/best.pt exists; S4.5 variants are immutable")
    cfg=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text());bc=cfg["bc"];stats=inspect_stacking_dataset(ROOT/"data/stacking_v1")
    split=json.loads((ROOT/"data/stacking_v1/metadata/bc_split_v1.json").read_text());demo=ROOT/"data/stacking_v1/demonstrations"
    train_o,train_a=load_stacking_transitions(demo,split["train_trajectory_ids"]);val_o,val_a=load_stacking_transitions(demo,split["validation_trajectory_ids"])
    feature_transform=None; original_train_count=len(train_o)
    if args.variant in {"phase_balanced_v1","relative_phase_balanced_v1"}:
        # Four extra copies make precision phases approximately as influential as
        # ordinary Cartesian travel, without changing validation or source data.
        critical={"ALIGN","LOWER_TO_PRECONTACT","LOWER_TO_CONTACT","RELEASE"}; extra_o=[];extra_a=[]
        for tid in split["train_trajectory_ids"]:
            t=load_stacking_trajectory(demo/f"{tid}.npz"); mask=np.isin(t["phases"],list(critical)); extra_o.append(t["observations"][mask]);extra_a.append(t["actions"][mask])
        precision_o=np.concatenate(extra_o);precision_a=np.concatenate(extra_a)
        train_o=np.concatenate([train_o]+[precision_o]*4);train_a=np.concatenate([train_a]+[precision_a]*4)
    if args.variant in {"relative_v1","relative_phase_balanced_v1"}:
        feature_transform=RELATIVE_V1; train_o=transform_stacking_observations(train_o,RELATIVE_V1);val_o=transform_stacking_observations(val_o,RELATIVE_V1)
    mean,std=compute_observation_statistics(train_o)
    ckpt.mkdir(parents=True);res.mkdir(parents=True);metadata={"task_type":"object_stacking","dataset_fingerprint":stats["dataset_fingerprint"],"split_metadata":split,
        "s4_5_variant":out_name,"phase_critical_oversampling_factor":5 if "phase_balanced" in args.variant else 1,
        "original_train_transition_count":original_train_count,"effective_train_transition_count":len(train_o),"training_configuration":bc}
    if feature_transform: metadata["feature_transform"]=feature_transform
    history,best=train_behavior_cloning(train_observations=train_o,train_actions=train_a,validation_observations=val_o,validation_actions=val_a,
        observation_mean=mean,observation_std=std,hidden_dims=bc["hidden_dims"],batch_size=bc["batch_size"],learning_rate=bc["learning_rate"],epochs=bc["max_epochs"],
        weight_decay=bc["weight_decay"],lambda_gripper=bc["lambda_gripper"],normalization_epsilon=bc["normalization_epsilon"],seed=bc["seed"],device=select_device(args.device),
        checkpoint_dir=ckpt,checkpoint_metadata=metadata,normalize_motion_targets=True,early_stopping_patience=bc["early_stopping_patience"])
    # Metadata keys are flattened into checkpoints by the shared trainer.
    save_history(res/"training_history.json",history);(res/"best_metrics.json").write_text(json.dumps(best,indent=2,sort_keys=True)+"\n");print(best)
if __name__=="__main__":main()
