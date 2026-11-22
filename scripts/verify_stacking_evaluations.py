#!/usr/bin/env python3
"""Verify uniqueness, schema, task coverage, and checkpoint metadata for S6/S7."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
FP="32a8ca20314b9a4a8ac1147b5af20b3e3b16d8b755944a99c2dcf6114c073940";METHODS=("uniform","task","expert","learner","reverse_learner");SEEDS=(5478,5479,5480);STEPS=(2060,5150,10300,15450,20600)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--final",action="store_true");ap.add_argument("--require-complete",action="store_true");a=ap.parse_args()
    name="final_evaluation" if a.final else "curriculum";expected_tasks=200 if a.final else 100;steps=(20600,) if a.final else STEPS;root=ROOT/f"outputs/stacking_v1/results/{name}/rollouts";total=0;files=0
    for method in METHODS:
        for seed in SEEDS:
            for step in steps:
                cp=ROOT/f"outputs/stacking_v1/checkpoints/curriculum/{method}/seed_{seed}/step_{step:06d}.pt";meta=torch.load(cp,map_location="cpu",weights_only=False)
                assert meta["dataset_fingerprint"]==FP and meta["optimizer_step"]==step and meta["training_seed"]==seed and meta["method_name"]==method and meta["architecture"]["observation_dim"]==29 and meta["feature_transform"]=="stacking_relative_v1" and meta["phase_critical_oversampling_factor"]==5
                path=root/method/f"seed_{seed}"/f"step_{step:06d}.jsonl";rows=[] if not path.exists() else [json.loads(x) for x in path.read_text().splitlines() if x];ids=[x["task_id"] for x in rows]
                assert len(ids)==len(set(ids));assert len(ids)<=expected_tasks
                for x in rows:assert 1<=x["steps"]<=400 and all(k in x for k in ("success","failure_reason","final_alignment_error","physical_events","final_diagnostics"))
                if a.require_complete:assert len(rows)==expected_tasks,(path,len(rows))
                total+=len(rows);files+=bool(rows)
    expected=len(METHODS)*len(SEEDS)*len(steps)*expected_tasks
    print(json.dumps({"mode":name,"nonempty_files":files,"valid_unique_records":total,"expected":expected,"remaining":expected-total,"complete":total==expected},indent=2))
if __name__=="__main__":main()
