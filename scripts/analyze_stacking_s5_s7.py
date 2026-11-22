#!/usr/bin/env python3
"""Aggregate S5–S7 results and create publication-ready tables and figures."""
from __future__ import annotations
import csv,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
METHODS=("uniform","task","expert","learner","reverse_learner");LABEL={"uniform":"Uniform","task":"Task","expert":"Expert","learner":"Learner","reverse_learner":"Reverse Learner"};SEEDS=(5478,5479,5480);STEPS=(2060,5150,10300,15450,20600)
def read_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x]
def aggregate(root,steps,tasks):
    rows=[]
    for method in METHODS:
        for seed in SEEDS:
            for step in steps:
                p=root/"rollouts"/method/f"seed_{seed}"/f"step_{step:06d}.jsonl";x=read_jsonl(p)
                if len(x)!=tasks or len({v["task_id"] for v in x})!=tasks:raise ValueError(f"incomplete {p}: {len(x)}")
                rows.append({"method":method,"seed":seed,"step":step,"successes":sum(v["success"] for v in x),"success_rate":float(np.mean([v["success"] for v in x])),"mean_rollout_length":float(np.mean([v["steps"] for v in x])),"failures":dict(Counter(v["failure_reason"] for v in x if not v["success"]))})
    return rows
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    diff=json.loads((ROOT/"outputs/stacking_v1/results/difficulty/difficulty_rankings_v1.json").read_text());fig=ROOT/"outputs/stacking_v1/figures";fig.mkdir(parents=True,exist_ok=True)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    corr=np.asarray(diff["spearman"]["matrix"]);im=plt.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm");plt.xticks(range(3),["Task","Expert","Learner"]);plt.yticks(range(3),["Task","Expert","Learner"])
    for i in range(3):
        for j in range(3):plt.text(j,i,f"{corr[i,j]:.2f}",ha="center",va="center")
    plt.colorbar(im);plt.tight_layout();plt.savefig(fig/"difficulty_correlation.png",dpi=200);plt.savefig(fig/"difficulty_correlation.pdf");plt.close()
    devroot=ROOT/"outputs/stacking_v1/results/curriculum";dev=aggregate(devroot,STEPS,100);write_csv(devroot/"development_per_seed_checkpoint.csv",dev)
    curves={}
    for m in METHODS:
        means=[np.mean([x["success_rate"] for x in dev if x["method"]==m and x["step"]==s]) for s in STEPS];curves[m]=means;plt.plot(STEPS,means,marker="o",label=LABEL[m])
    plt.xlabel("Optimizer updates");plt.ylabel("Development success rate");plt.legend(fontsize=8);plt.tight_layout();plt.savefig(fig/"curriculum_learning_curves.png",dpi=200);plt.savefig(fig/"curriculum_learning_curves.pdf");plt.close()
    finalroot=ROOT/"outputs/stacking_v1/results/final_evaluation";final=aggregate(finalroot,[20600],200);write_csv(finalroot/"final_per_seed.csv",final)
    summary=[]
    for m in METHODS:
        x=[v for v in final if v["method"]==m];rates=np.array([v["success_rate"] for v in x]);summary.append({"method":m,"mean_success":rates.mean(),"sample_sd":rates.std(ddof=1),"seed_5478":rates[0],"seed_5479":rates[1],"seed_5480":rates[2],"mean_rollout_length":np.mean([v["mean_rollout_length"] for v in x])})
    write_csv(finalroot/"final_summary.csv",summary);(finalroot/"summary.json").write_text(json.dumps({"development":dev,"final":final,"summary":summary,"learning_curve_auc":{m:float(np.trapezoid(curves[m],np.asarray(STEPS)/20600)) for m in METHODS}},indent=2,sort_keys=True)+"\n")
    xs=np.arange(5);means=[x["mean_success"] for x in summary];sd=[x["sample_sd"] for x in summary];plt.bar(xs,means,yerr=sd,capsize=4);plt.xticks(xs,[LABEL[m] for m in METHODS],rotation=20,ha="right");plt.ylabel("Final success rate");plt.tight_layout();plt.savefig(fig/"final_success.png",dpi=200);plt.savefig(fig/"final_success.pdf");plt.close()
if __name__=="__main__":main()
