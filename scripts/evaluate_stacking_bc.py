#!/usr/bin/env python3
"""Evaluate Stacking Uniform BC on 100 frozen development tasks."""
from __future__ import annotations
import argparse,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.stacking_task_sampler import StackingTaskConfig,StackingTaskSampler
from envs.panda_stacking import PandaStackingEnv
from evaluation.stacking_rollout import rollout_stacking_bc
from models.bc_policy import load_policy_checkpoint

def key(x):return json.dumps([list(x["source_position"]),list(x["support_position"]),float(x["placement_tolerance"])],separators=(",",":"))
def load_tasks(metadata,seed=35478,count=100):
    path=ROOT/"data/stacking_v1/metadata/development_eval_tasks_v1.json";excluded={key(x) for x in metadata["trajectory_summaries"]}
    if path.exists():payload=json.loads(path.read_text())
    else:
        sampler=StackingTaskSampler(seed);items=[];seen=set(excluded)
        while len(items)<count:
            item=sampler.sample(f"stack_dev_{len(items):04d}").as_dict();k=key(item)
            if k in seen:continue
            seen.add(k);items.append(item)
        payload={"version":1,"dataset_fingerprint":metadata["dataset_fingerprint"],"seed":seed,"count":count,"tasks":items}
        path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    if payload["dataset_fingerprint"]!=metadata["dataset_fingerprint"] or set(map(key,payload["tasks"]))&excluded:raise ValueError("invalid evaluation task set")
    return [StackingTaskConfig(x["task_id"],x["seed"],np.asarray(x["source_position"]),np.asarray(x["support_position"]),x["placement_tolerance"]) for x in payload["tasks"]]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=ROOT/"outputs/stacking_v1/checkpoints/uniform_bc/best.pt");ap.add_argument("--overwrite",action="store_true");args=ap.parse_args()
    out=ROOT/"outputs/stacking_v1/results/uniform_bc/rollout_evaluation.json"
    if out.exists() and not args.overwrite:raise SystemExit("Evaluation exists; pass --overwrite explicitly")
    metadata=json.loads((ROOT/"data/stacking_v1/metadata/dataset_metadata.json").read_text());cfg=yaml.safe_load((ROOT/"configs/stacking_v1.yaml").read_text());policy,checkpoint=load_policy_checkpoint(args.checkpoint,"cpu")
    if checkpoint["dataset_fingerprint"]!=metadata["dataset_fingerprint"] or checkpoint["architecture"]["observation_dim"]!=29:raise ValueError("checkpoint mismatch")
    tasks=load_tasks(metadata);results=[];partial=out.with_suffix(".partial.jsonl")
    if args.overwrite: partial.unlink(missing_ok=True)
    if partial.exists():
        from evaluation.stacking_rollout import StackingBCRolloutResult
        results=[StackingBCRolloutResult(**json.loads(line)) for line in partial.read_text().splitlines() if line]
    completed={r.task_id for r in results}
    with PandaStackingEnv() as env:
        for i,t in enumerate(tasks,1):
            if t.task_id in completed: continue
            r=rollout_stacking_bc(env,policy,t,cfg["bc"]["max_rollout_steps"]);results.append(r)
            with partial.open("a") as f:f.write(json.dumps(r.as_dict(),sort_keys=True)+"\n")
            if i%10==0:print(i,sum(x.success for x in results),flush=True)
    successes=sum(r.success for r in results);lengths=np.array([r.steps for r in results]);errors=np.array([r.final_alignment_error for r in results])
    summary={"checkpoint":str(args.checkpoint),"checkpoint_epoch":checkpoint["epoch"],"evaluation_seed":35478,"tasks":100,"successes":successes,"success_rate":successes/100,
             "mean_rollout_length":float(lengths.mean()),"median_rollout_length":float(np.median(lengths)),"mean_final_alignment_error":float(errors.mean()),
             "failure_reasons":dict(Counter(r.failure_reason for r in results if not r.success)),"episodes":[r.as_dict() for r in results]}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");partial.unlink(missing_ok=True);print(summary|{"episodes":"omitted"})
if __name__=="__main__":main()
