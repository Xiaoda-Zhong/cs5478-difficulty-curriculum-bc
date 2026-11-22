#!/usr/bin/env python3
"""Collect and freeze 300 successful physical Stacking V1 demonstrations."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_task_sampler import StackingTaskSampler
from data.stacking_trajectory import *
from envs.panda_stacking import PandaStackingEnv
from expert.stacking_expert import ScriptedStackingExpert
from training.bc_dataset import generate_trajectory_split

DATA_ROOT=ROOT/"data/stacking_v1"; DEMOS=DATA_ROOT/"demonstrations"; META=DATA_ROOT/"metadata"
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--overwrite",action="store_true");args=ap.parse_args()
    existing=list(DEMOS.glob("traj_*.npz"))+list(META.glob("*.json"))
    if existing and not args.overwrite: raise SystemExit("Stacking dataset exists; pass --overwrite explicitly")
    if args.overwrite:
        for p in existing:p.unlink()
    config=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text()); target=300; sampler=StackingTaskSampler(5478)
    logs=[]; summaries=[]; entries=[]; failures=Counter(); success=attempts=0
    with PandaStackingEnv() as env:
        while success<target and attempts<315:
            task=sampler.sample(); tid=f"traj_{success:04d}"; t=rollout_stacking_trajectory(env,ScriptedStackingExpert(),task,tid);attempts+=1
            logs.append({"attempt":attempts,**task.as_dict(),"success":bool(t["success"]),"failure_reason":str(t["failure_reason"]),"steps":int(t["trajectory_length"])})
            if not t["success"]:failures[str(t["failure_reason"])]+=1;continue
            path=DEMOS/f"{tid}.npz";save_stacking_trajectory(path,t);entry=task_entry(t);entries.append(entry)
            summaries.append({**entry,"trajectory_length":int(t["trajectory_length"]),"ee_path_length":float(t["ee_path_length"]),"cumulative_action_magnitude":float(t["cumulative_action_magnitude"])});success+=1
            if success%25==0:print(f"Collected {success}/{target} successful trajectories",flush=True)
    write(META/"collection_log.json",logs)
    if success<target:raise SystemExit(f"collection failed: {success}/{target}, failures={dict(failures)}")
    fingerprint=task_fingerprint(entries); manifest={p.name:file_sha256(p) for p in sorted(DEMOS.glob("traj_*.npz"))}
    write(META/"trajectory_sha256.json",manifest)
    lengths=[r["trajectory_length"] for r in summaries]
    metadata={"dataset_version":1,"task_type":"object_stacking","creation_timestamp_utc":datetime.now(timezone.utc).isoformat(),
              "global_dataset_seed":5478,"number_of_trajectories":300,"attempted_episodes":attempts,"failed_episodes":attempts-success,
              "failure_reasons":dict(failures),"observation_dimension":29,"action_dimension":4,
              "total_state_action_pairs":sum(lengths),"dataset_fingerprint":fingerprint,
              "task_randomization":config["task_randomization"],"expert_configuration":config["expert"],"trajectory_summaries":summaries}
    write(META/"dataset_metadata.json",metadata)
    ids=[f"traj_{i:04d}" for i in range(300)];train,val=generate_trajectory_split(ids,.8,5478)
    write(META/"bc_split_v1.json",{"dataset_fingerprint":fingerprint,"split_seed":5478,"train_fraction":.8,"train_trajectory_ids":train,"validation_trajectory_ids":val})
    print(f"Dataset fingerprint: {fingerprint}; transitions={sum(lengths)}; attempts={attempts}")
if __name__=="__main__":main()
