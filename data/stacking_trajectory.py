"""Stacking expert rollout, schema validation, serialization, and hashing."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from data.stacking_task_sampler import StackingTaskConfig
from envs.panda_stacking import PandaStackingEnv
from expert.stacking_expert import ScriptedStackingExpert

def rollout_stacking_trajectory(env: PandaStackingEnv, expert: ScriptedStackingExpert,
                                task: StackingTaskConfig, trajectory_id: str) -> dict[str, Any]:
    obs = env.reset(task.source_position, task.support_position, task.placement_tolerance, task.seed)
    expert.reset(env)
    observations=[]; actions=[]; phases=[]; diagnostics=[]
    ee=[env.get_ee_position()]; sp=[env.get_source_position()]; tp=[env.get_support_position()]
    so=[env.get_source_pose()[1]]; to=[env.get_support_pose()[1]]
    sl=[]; sa=[]; tl=[]; ta=[]
    while not expert.is_done:
        observations.append(obs.copy()); phases.append(expert.current_phase.value)
        diagnostics.append(env.get_stack_diagnostics())
        sv=env.get_source_velocity(); tv=env.get_support_velocity()
        sl.append(sv[0]); sa.append(sv[1]); tl.append(tv[0]); ta.append(tv[1])
        action=expert.act(obs); actions.append(action.copy()); obs,_=env.step(action)
        ee.append(env.get_ee_position()); sp.append(env.get_source_position()); tp.append(env.get_support_position())
        so.append(env.get_source_pose()[1]); to.append(env.get_support_pose()[1])
    result=expert.result(); actions_a=np.asarray(actions,dtype=np.float32); ee_a=np.asarray(ee)
    final=env.get_stack_diagnostics()
    return {"trajectory_id":trajectory_id,"task_id":task.task_id,"task_seed":task.seed,
            "success":bool(result.success and env.is_success()),"failure_reason":result.failure_reason or "",
            "source_initial_position":task.source_position,"support_initial_position":task.support_position,
            "placement_tolerance":task.placement_tolerance,"observations":np.asarray(observations,dtype=np.float32),
            "actions":actions_a,"ee_positions":ee_a,"source_positions":np.asarray(sp),"support_positions":np.asarray(tp),
            "source_orientations":np.asarray(so),"support_orientations":np.asarray(to),
            "source_linear_velocities":np.asarray(sl),"source_angular_velocities":np.asarray(sa),
            "support_linear_velocities":np.asarray(tl),"support_angular_velocities":np.asarray(ta),
            "phases":np.asarray(phases,dtype="U32"),"trajectory_length":len(actions),
            "ee_path_length":float(np.linalg.norm(np.diff(ee_a,axis=0),axis=1).sum()),
            "cumulative_action_magnitude":float(np.linalg.norm(actions_a[:,:3],axis=1).sum()),
            "maximum_support_displacement":result.maximum_support_displacement,
            "final_diagnostics_json":json.dumps(final,sort_keys=True)}

def validate_stacking_trajectory(t: Mapping[str,Any], require_success: bool=True) -> None:
    n=int(np.asarray(t["trajectory_length"]).item())
    shapes={"observations":(n,29),"actions":(n,4),"ee_positions":(n+1,3),
            "source_positions":(n+1,3),"support_positions":(n+1,3),
            "source_orientations":(n+1,4),"support_orientations":(n+1,4),
            "source_linear_velocities":(n,3),"source_angular_velocities":(n,3),
            "support_linear_velocities":(n,3),"support_angular_velocities":(n,3),"phases":(n,)}
    for key,shape in shapes.items():
        a=np.asarray(t[key]);
        if a.shape!=shape: raise ValueError(f"{key} shape {a.shape} != {shape}")
        if key!="phases" and not np.all(np.isfinite(a)): raise ValueError(f"{key} contains non-finite values")
    if require_success and not bool(np.asarray(t["success"]).item()): raise ValueError("trajectory is not successful")
    if not np.allclose(np.asarray(t["observations"])[:,28],float(np.asarray(t["placement_tolerance"]).item())):
        raise ValueError("placement tolerance observation mismatch")

def save_stacking_trajectory(path:Path,t:Mapping[str,Any])->None:
    validate_stacking_trajectory(t); path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**t)
def load_stacking_trajectory(path:Path)->dict[str,Any]:
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def task_entry(t:Mapping[str,Any])->dict[str,Any]:
    return {"trajectory_id":str(np.asarray(t["trajectory_id"]).item()),"task_id":str(np.asarray(t["task_id"]).item()),
            "task_seed":int(np.asarray(t["task_seed"]).item()),"source_position":np.asarray(t["source_initial_position"]).tolist(),
            "support_position":np.asarray(t["support_initial_position"]).tolist(),"placement_tolerance":float(np.asarray(t["placement_tolerance"]).item())}
def task_fingerprint(entries:list[Mapping[str,Any]])->str:
    return hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def file_sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
