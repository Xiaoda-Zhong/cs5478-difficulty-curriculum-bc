#!/usr/bin/env python3
"""Train the 15 matched fixed-budget S6 stacking curriculum runs."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_trajectory import load_stacking_trajectory
from models.stacking_features import RELATIVE_V1,transform_stacking_observations
from training.bc_dataset import compute_observation_statistics
from training.curriculum_sampler import METHODS
from training.train_bc import select_device
from training.train_curriculum import create_initial_state,state_dict_fingerprint,train_curriculum_run
FP="32a8ca20314b9a4a8ac1147b5af20b3e3b16d8b755944a99c2dcf6114c073940";CRITICAL={"ALIGN","LOWER_TO_PRECONTACT","LOWER_TO_CONTACT","RELEASE"}
def load(ids,balance=False):
    oo=[];aa=[];tt=[];pp=[]
    for tid in ids:
        x=load_stacking_trajectory(ROOT/"data/stacking_v1/demonstrations"/f"{tid}.npz");n=int(x["trajectory_length"]);oo.append(x["observations"]);aa.append(x["actions"]);tt.append(np.repeat(tid,n));pp.append(x["phases"])
    o=np.concatenate(oo);a=np.concatenate(aa);t=np.concatenate(tt);p=np.concatenate(pp)
    if balance:
        m=np.isin(p,list(CRITICAL));o=np.concatenate([o]+[o[m]]*4);a=np.concatenate([a]+[a[m]]*4);t=np.concatenate([t]+[t[m]]*4)
    return transform_stacking_observations(o,RELATIVE_V1),a,t
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--methods",nargs="+",choices=METHODS);ap.add_argument("--seeds",nargs="+",type=int);ap.add_argument("--device",choices=["cpu","mps","auto"],default="cpu");ap.add_argument("--smoke",action="store_true");args=ap.parse_args()
    split=json.loads((ROOT/"data/stacking_v1/metadata/bc_split_v1.json").read_text());rankp=json.loads((ROOT/"outputs/stacking_v1/results/difficulty/difficulty_rankings_v1.json").read_text());assert split["dataset_fingerprint"]==rankp["dataset_fingerprint"]==FP
    rankings={k:v["groups"] for k,v in rankp["rankings"].items()};train_o,train_a,tids=load(split["train_trajectory_ids"],True);val_o,val_a,_=load(split["validation_trajectory_ids"],False);mean,std=compute_observation_statistics(train_o);scale=float(np.linalg.norm(train_a[:,:3],axis=1).max())
    methods=args.methods or list(METHODS);seeds=args.seeds or [5478,5479,5480];total=60 if args.smoke else 20600;steps=[20,40,60] if args.smoke else [2060,5150,10300,15450,20600]
    name="curriculum_smoke" if args.smoke else "curriculum";root=ROOT/f"outputs/stacking_v1/checkpoints/{name}";root.mkdir(parents=True,exist_ok=True);rankfile=ROOT/"outputs/stacking_v1/results/difficulty/difficulty_rankings_v1.json";rankhash=hashlib.sha256(rankfile.read_bytes()).hexdigest()
    for seed in seeds:
        initial=create_initial_state(seed,[128,128],29);fingerprint=state_dict_fingerprint(initial);ip=root/f"initial_seed_{seed}.pt"
        if ip.exists():
            old=torch.load(ip,map_location="cpu",weights_only=False);assert old["fingerprint"]==fingerprint
        else:torch.save({"fingerprint":fingerprint,"model_state_dict":initial,"training_seed":seed},ip)
        for method in methods:
            print(f"START {method} seed={seed} init={fingerprint[:12]}",flush=True)
            train_curriculum_run(method=method,seed=seed,observations=train_o,actions=train_a,transition_trajectory_ids=tids,validation_observations=val_o,validation_actions=val_a,training_ids=split["train_trajectory_ids"],validation_ids=split["validation_trajectory_ids"],rankings=rankings,initial_state=initial,observation_mean=mean,observation_std=std,motion_scale=scale,hidden_dims=[128,128],total_updates=total,stage_boundaries=[1/3,2/3],evaluation_steps=steps,batch_size=256,learning_rate=.001,weight_decay=1e-6,lambda_gripper=1.,normalization_epsilon=1e-4,resume_interval=min(1000,total),device=select_device(args.device),output_dir=root/method/f"seed_{seed}",dataset_fingerprint=FP,ranking_reference=str(rankfile.relative_to(ROOT)),ranking_fingerprint=rankhash,feature_transform=RELATIVE_V1,phase_critical_oversampling_factor=5)
if __name__=="__main__":main()
