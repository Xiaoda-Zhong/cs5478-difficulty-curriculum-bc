#!/usr/bin/env python3
"""Evaluate a distinct S4.5 checkpoint on a fixed prefix or all frozen dev tasks."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from envs.panda_stacking import PandaStackingEnv
from evaluation.stacking_rollout import rollout_stacking_bc
from models.bc_policy import DebouncedGripperPolicyAgent,load_policy_checkpoint
from scripts.evaluate_stacking_bc import load_tasks
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,required=True);ap.add_argument("--task-count",type=int,default=20);ap.add_argument("--gripper-debounce",type=int,default=1);ap.add_argument("--normalization-clip",type=float,default=None);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise SystemExit("output exists; variant results are immutable")
    meta=json.loads((ROOT/"data/stacking_v1/metadata/dataset_metadata.json").read_text());cfg=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text());policy,checkpoint=load_policy_checkpoint(args.checkpoint,"cpu")
    if checkpoint["dataset_fingerprint"]!=meta["dataset_fingerprint"]:raise ValueError("fingerprint mismatch")
    base=policy.base_agent if hasattr(policy,"base_agent") else policy
    if args.normalization_clip is not None:base.normalization_clip=args.normalization_clip
    if args.gripper_debounce>1:policy=DebouncedGripperPolicyAgent(policy,args.gripper_debounce)
    tasks=load_tasks(meta)[:args.task_count];results=[]
    with PandaStackingEnv() as env:
        for i,t in enumerate(tasks,1):
            results.append(rollout_stacking_bc(env,policy,t,cfg["bc"]["max_rollout_steps"]));print(i,sum(x.success for x in results),flush=True)
    lengths=np.array([r.steps for r in results]);payload={"checkpoint":str(args.checkpoint),"normalization_clip":args.normalization_clip,"gripper_debounce":args.gripper_debounce,"task_count":len(results),"successes":sum(r.success for r in results),
        "success_rate":float(np.mean([r.success for r in results])),"mean_steps":float(lengths.mean()),"failure_reasons":dict(Counter(r.failure_reason for r in results if not r.success)),"episodes":[r.as_dict() for r in results]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(payload|{"episodes":"omitted"})
if __name__=="__main__":main()
