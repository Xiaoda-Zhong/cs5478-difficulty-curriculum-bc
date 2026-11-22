#!/usr/bin/env python3
"""Compute and freeze all S5 stacking difficulty scores and rankings."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_trajectory import load_stacking_trajectory
from difficulty.learner import generate_oof_folds,validate_oof_folds,learner_errors,split_fingerprint
from models.bc_policy import load_policy_checkpoint
from models.stacking_features import RELATIVE_V1,transform_stacking_observations
from training.bc_dataset import compute_observation_statistics
from training.train_bc import select_device,train_fixed_epoch_behavior_cloning

FP="32a8ca20314b9a4a8ac1147b5af20b3e3b16d8b755944a99c2dcf6114c073940";CRITICAL={"ALIGN","LOWER_TO_PRECONTACT","LOWER_TO_CONTACT","RELEASE"}
def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def zscore(x):
    x=np.asarray(x,dtype=np.float64);mean=x.mean(0);std=x.std(0);return (x-mean)/(std+1e-12),{"mean":mean.tolist(),"std":std.tolist(),"epsilon":1e-12}
def load_many(ids,balanced=False):
    os=[];acts=[];ph=[]
    for tid in ids:
        t=load_stacking_trajectory(ROOT/"data/stacking_v1/demonstrations"/f"{tid}.npz");os.append(t["observations"]);acts.append(t["actions"]);ph.append(t["phases"])
    o=np.concatenate(os);a=np.concatenate(acts);p=np.concatenate(ph)
    if balanced:
        mask=np.isin(p,list(CRITICAL));o=np.concatenate([o]+[o[mask]]*4);a=np.concatenate([a]+[a[mask]]*4)
    return o,a,p
def ranks(records,key):
    ordered=sorted(records,key=lambda r:(r[key],r["trajectory_id"]));groups={"easy":[],"medium":[],"hard":[]}
    for i,r in enumerate(ordered):groups[("easy","medium","hard")[i//80]].append(r["trajectory_id"])
    payload={"score_key":key,"ordered_trajectory_ids":[r["trajectory_id"] for r in ordered],"groups":groups};payload["fingerprint"]=sha(payload);return payload
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",choices=["cpu","mps","auto"],default="cpu");args=ap.parse_args()
    split=json.loads((ROOT/"data/stacking_v1/metadata/bc_split_v1.json").read_text());assert split["dataset_fingerprint"]==FP;ids=split["train_trajectory_ids"]
    out=ROOT/"outputs/stacking_v1/results/difficulty";ckroot=ROOT/"outputs/stacking_v1/checkpoints/learner_oof";out.mkdir(parents=True,exist_ok=True);ckroot.mkdir(parents=True,exist_ok=True)
    if (out/"difficulty_rankings_v1.json").exists():raise SystemExit("S5 outputs already frozen")
    task_raw=[];expert_raw=[];base=[]
    for tid in ids:
        t=load_stacking_trajectory(ROOT/"data/stacking_v1/demonstrations"/f"{tid}.npz");ee=t["ee_positions"][0];s=t["source_initial_position"];q=t["support_initial_position"]
        task_raw.append([np.linalg.norm(ee-s),np.linalg.norm(s-q),1/float(t["placement_tolerance"])]);expert_raw.append([int(t["trajectory_length"]),float(t["ee_path_length"]),float(t["cumulative_action_magnitude"])]);base.append({"trajectory_id":tid})
    tz,tn=zscore(task_raw);ez,en=zscore(expert_raw)
    for i,r in enumerate(base):r.update({"task_components_raw":task_raw[i],"task_components_z":tz[i].tolist(),"task_difficulty":float(tz[i].mean()),"expert_components_raw":expert_raw[i],"expert_components_z":ez[i].tolist(),"expert_difficulty":float(ez[i].mean())})
    folds=generate_oof_folds(ids,5,5478);validate_oof_folds(folds,ids);fold_payload={"dataset_fingerprint":FP,"fold_seed":5478,"training_split_fingerprint":split_fingerprint(ids),"folds":folds};(out/"learner_oof_folds_v1.json").write_text(json.dumps(fold_payload,indent=2,sort_keys=True)+"\n")
    cfg=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text());bc=cfg["bc"];byid={r["trajectory_id"]:r for r in base}
    for fold in folds:
        k=int(fold["fold_index"]);path=ckroot/f"fold_{k}.pt";train=fold["training_trajectory_ids"];held=fold["held_out_trajectory_ids"]
        if not path.exists():
            o,a,_=load_many(train,True);o=transform_stacking_observations(o,RELATIVE_V1);mean,std=compute_observation_statistics(o)
            train_fixed_epoch_behavior_cloning(train_observations=o,train_actions=a,observation_mean=mean,observation_std=std,hidden_dims=[128,128],batch_size=256,learning_rate=.001,epochs=95,weight_decay=float(bc["weight_decay"]),lambda_gripper=float(bc["lambda_gripper"]),normalization_epsilon=1e-4,seed=5478+k,device=select_device(args.device),checkpoint_path=path,checkpoint_metadata={"task_type":"object_stacking","dataset_fingerprint":FP,"fold_index":k,"training_trajectory_ids":train,"held_out_trajectory_ids":held,"feature_transform":RELATIVE_V1,"phase_critical_oversampling_factor":5,"fixed_training_epochs":95})
        policy,cp=load_policy_checkpoint(path,"cpu");assert not(set(cp["training_trajectory_ids"])&set(held));scale=float(cp["motion_scale"])
        for tid in held:
            t=load_stacking_trajectory(ROOT/"data/stacking_v1/demonstrations"/f"{tid}.npz");pred,logit=policy.predict_batch_with_logits(t["observations"]);err=learner_errors(pred,logit,t["actions"],scale);byid[tid].update(err);byid[tid]["learner_difficulty"]=float(err["learner_motion_error"]);byid[tid]["oof_fold"]=k
    if len(byid)!=240 or any("learner_difficulty" not in x for x in base):raise RuntimeError("OOF coverage failure")
    rankings={name:ranks(base,f"{name}_difficulty") for name in ("task","expert","learner")}
    scores=np.array([[r[f"{x}_difficulty"] for x in ("task","expert","learner")] for r in base]);order=np.argsort(np.argsort(scores,axis=0),axis=0);corr=np.corrcoef(order.T)
    payload={"version":1,"dataset_fingerprint":FP,"training_trajectory_count":240,"rankings":rankings,"spearman":{"labels":["task","expert","learner"],"matrix":corr.tolist()},"task_normalization":tn,"expert_normalization":en,"expert_component_correlation":np.corrcoef(np.asarray(expert_raw).T).tolist(),"records":base};payload["artifact_fingerprint"]=sha(payload)
    (out/"difficulty_rankings_v1.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps({"artifact_fingerprint":payload["artifact_fingerprint"],"ranking_fingerprints":{k:v["fingerprint"] for k,v in rankings.items()},"spearman":payload["spearman"]},indent=2))
if __name__=="__main__":main()
