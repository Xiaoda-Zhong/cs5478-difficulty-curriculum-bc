#!/usr/bin/env python3
"""Resumable parallel S6 development and S7 final evaluation."""
from __future__ import annotations
import argparse,hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_task_sampler import StackingTaskConfig,StackingTaskSampler
from envs.panda_stacking import PandaStackingEnv
from evaluation.stacking_rollout import rollout_stacking_bc
from models.bc_policy import load_policy_checkpoint
from scripts.evaluate_stacking_bc import load_tasks,key
FP="32a8ca20314b9a4a8ac1147b5af20b3e3b16d8b755944a99c2dcf6114c073940";METHODS=("uniform","task","expert","learner","reverse_learner");SEEDS=(5478,5479,5480);STEPS=(2060,5150,10300,15450,20600)
def task_obj(x):return StackingTaskConfig(x["task_id"],int(x["seed"]),np.asarray(x["source_position"]),np.asarray(x["support_position"]),float(x["placement_tolerance"]))
def final_tasks(meta):
    path=ROOT/"outputs/stacking_v1/results/final_evaluation/final_tasks_seed_65478.json"
    if path.exists():return json.loads(path.read_text())
    used={key(x) for x in meta["trajectory_summaries"]};dev=json.loads((ROOT/"data/stacking_v1/metadata/development_eval_tasks_v1.json").read_text());used|={key(x) for x in dev["tasks"]}
    # Include every discoverable prior stacking task list.
    for p in ROOT.glob("outputs/**/stack*task*.json"):
        try:
            x=json.loads(p.read_text());items=x.get("tasks",[]) if isinstance(x,dict) else []
            used|={key(v) for v in items if all(k in v for k in ("source_position","support_position","placement_tolerance"))}
        except Exception:pass
    sampler=StackingTaskSampler(65478);items=[];seen=set(used)
    while len(items)<200:
        x=sampler.sample(f"stack_final_{len(items):04d}").as_dict();k=key(x)
        if k in seen:continue
        seen.add(k);items.append(x)
    core={"version":1,"seed":65478,"dataset_fingerprint":FP,"count":200,"tasks":items};core["task_list_fingerprint"]=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(core,indent=2,sort_keys=True)+"\n");return core
def worker(job):
    checkpoint,out_path,tasks=job;out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    completed={}
    if out.exists():
        for line in out.read_text().splitlines():
            if line:
                x=json.loads(line);completed[x["task_id"]]=x
    policy,cp=load_policy_checkpoint(checkpoint,"cpu")
    if cp["dataset_fingerprint"]!=FP or cp["optimizer_step"] not in STEPS:raise ValueError("checkpoint metadata mismatch")
    cfg=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text());
    with PandaStackingEnv() as env:
        for x in tasks:
            if x["task_id"] in completed:continue
            r=rollout_stacking_bc(env,policy,task_obj(x),int(cfg["bc"]["max_rollout_steps"]));completed[x["task_id"]]=r.as_dict()
            with out.open("a") as f:f.write(json.dumps(r.as_dict(),sort_keys=True)+"\n")
    return str(out),sum(x["success"] for x in completed.values()),len(completed)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--final",action="store_true");ap.add_argument("--workers",type=int,default=4);args=ap.parse_args();meta=json.loads((ROOT/"data/stacking_v1/metadata/dataset_metadata.json").read_text());assert meta["dataset_fingerprint"]==FP
    if args.final:
        payload=final_tasks(meta);tasks=payload["tasks"];steps=[20600];name="final_evaluation"
    else:
        payload=json.loads((ROOT/"data/stacking_v1/metadata/development_eval_tasks_v1.json").read_text());tasks=payload["tasks"];steps=list(STEPS);name="curriculum"
    jobs=[]
    for method in METHODS:
        for seed in SEEDS:
            for step in steps:
                cp=ROOT/f"outputs/stacking_v1/checkpoints/curriculum/{method}/seed_{seed}/step_{step:06d}.pt";out=ROOT/f"outputs/stacking_v1/results/{name}/rollouts/{method}/seed_{seed}/step_{step:06d}.jsonl";jobs.append((str(cp),str(out),tasks))
    print(f"jobs={len(jobs)} rollouts={len(jobs)*len(tasks)} workers={args.workers}",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            path,success,n=f.result();print(f"{i}/{len(jobs)} {path} success={success}/{n}",flush=True)
if __name__=="__main__":main()
