#!/usr/bin/env python3
import argparse,sys,time,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_task_sampler import StackingTaskConfig
from envs.panda_stacking import PandaStackingEnv
from evaluation.stacking_rollout import rollout_stacking_bc
from models.bc_policy import load_policy_checkpoint
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=ROOT/"outputs/stacking_v1/checkpoints/uniform_bc/best.pt");ap.add_argument("--task-index",type=int,default=0);ap.add_argument("--hold-seconds",type=float,default=5);a=ap.parse_args()
    payload=json.loads((ROOT/"data/stacking_v1/metadata/development_eval_tasks_v1.json").read_text());x=payload["tasks"][a.task_index];t=StackingTaskConfig(x["task_id"],x["seed"],np.array(x["source_position"]),np.array(x["support_position"]),x["placement_tolerance"]);policy,_=load_policy_checkpoint(a.checkpoint,"cpu")
    with PandaStackingEnv("gui") as env:r=rollout_stacking_bc(env,policy,t,400);print(r);time.sleep(a.hold_seconds)
if __name__=="__main__":main()
