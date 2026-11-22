"""Frozen Stacking V1 integrity inspection and transition loading."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from data.stacking_trajectory import load_stacking_trajectory,validate_stacking_trajectory,task_entry,task_fingerprint,file_sha256

def inspect_stacking_dataset(root:Path)->dict:
    metadata=json.loads((root/"metadata/dataset_metadata.json").read_text())
    files=sorted((root/"demonstrations").glob("traj_*.npz"))
    if len(files)!=300: raise ValueError(f"expected 300 archives, found {len(files)}")
    manifest=json.loads((root/"metadata/trajectory_sha256.json").read_text())
    entries=[]; lengths=[]; paths=[]
    for path in files:
        if manifest[path.name]!=file_sha256(path): raise ValueError(f"content hash mismatch: {path.name}")
        t=load_stacking_trajectory(path); validate_stacking_trajectory(t)
        if str(np.asarray(t["trajectory_id"]).item())!=path.stem: raise ValueError("trajectory ID mismatch")
        entries.append(task_entry(t)); lengths.append(int(t["trajectory_length"])); paths.append(float(t["ee_path_length"]))
    fingerprint=task_fingerprint(entries)
    if fingerprint!=metadata["dataset_fingerprint"]: raise ValueError("dataset fingerprint mismatch")
    if sum(lengths)!=metadata["total_state_action_pairs"]: raise ValueError("transition count mismatch")
    split=json.loads((root/"metadata/bc_split_v1.json").read_text()); train=set(split["train_trajectory_ids"]); val=set(split["validation_trajectory_ids"])
    if train&val or len(train)!=240 or len(val)!=60 or train|val!={p.stem for p in files}: raise ValueError("invalid split")
    return {"trajectories":300,"total_transitions":sum(lengths),"dataset_fingerprint":fingerprint,
            "trajectory_length":_stats(lengths),"ee_path_length":_stats(paths),"content_hashes_verified":True}

def load_stacking_transitions(demo_dir:Path,ids:list[str])->tuple[np.ndarray,np.ndarray]:
    obs=[]; actions=[]
    for tid in ids:
        t=load_stacking_trajectory(demo_dir/f"{tid}.npz"); validate_stacking_trajectory(t)
        obs.append(np.asarray(t["observations"],dtype=np.float32)); actions.append(np.asarray(t["actions"],dtype=np.float32))
    return np.concatenate(obs),np.concatenate(actions)
def _stats(v):
    a=np.asarray(v,dtype=float);return {"mean":float(a.mean()),"std":float(a.std()),"min":float(a.min()),"max":float(a.max())}
