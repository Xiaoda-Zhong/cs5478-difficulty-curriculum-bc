"""Autonomous closed-loop rollout utilities for stacking policies."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Any
import numpy as np
from data.stacking_task_sampler import StackingTaskConfig
from envs.panda_stacking import PandaStackingEnv
from models.bc_policy import BCPolicyAgent

@dataclass
class StackingBCRolloutResult:
    task_id:str;success:bool;failure_reason:str;steps:int;final_alignment_error:float
    final_source_position:list[float];final_support_position:list[float];physical_events:dict[str,bool];max_stable_steps:int;final_diagnostics:dict[str,Any]|None=None
    def as_dict(self)->dict[str,Any]:return asdict(self)

def rollout_stacking_bc(env:PandaStackingEnv,policy:BCPolicyAgent,task:StackingTaskConfig,max_steps:int)->StackingBCRolloutResult:
    if hasattr(policy,"reset"): policy.reset()
    obs=env.reset(task.source_position,task.support_position,task.placement_tolerance,task.seed);initial_source=env.get_source_position();initial_dist=float(np.linalg.norm(initial_source[:2]-env.get_support_position()[:2]));closed=False;maxstable=0
    events={k:False for k in ["reached_source","gripper_closed","source_lifted","transported_toward_support","source_support_contact","released_after_close","gripper_retreated","instantaneous_stable"]}
    reason="timeout"
    for step in range(1,max_steps+1):
        try: action=policy.act(obs)
        except Exception: reason="invalid_policy_action";break
        obs,info=env.step(action);source=env.get_source_position();support=env.get_support_position();d=env.get_stack_diagnostics();dist=float(np.linalg.norm(source[:2]-support[:2]))
        events["reached_source"] = bool(events["reached_source"] or np.linalg.norm(env.get_ee_position()-source)<.08)
        if action[3]>0:closed=True;events["gripper_closed"]=True
        events["released_after_close"] = bool(events["released_after_close"] or (closed and action[3]<=0))
        events["source_lifted"] = bool(events["source_lifted"] or source[2]>initial_source[2]+.04)
        events["transported_toward_support"] = bool(events["transported_toward_support"] or dist<initial_dist-.05)
        events["source_support_contact"] = bool(events["source_support_contact"] or d["source_support_contact"])
        events["gripper_retreated"] = bool(events["gripper_retreated"] or d["gripper_retreat_clearance"]>=.10)
        events["instantaneous_stable"] = bool(events["instantaneous_stable"] or env.is_stack_physically_stable());maxstable=max(maxstable,env.stable_steps)
        if env.is_success():reason="success";break
        if source[2]<-.05:reason="source_left_workspace";break
    success=env.is_success();d=env.get_stack_diagnostics()
    if not success and reason=="timeout":reason=classify_failure(events)
    return StackingBCRolloutResult(task.task_id,success,reason,step,d["horizontal_alignment_error"],env.get_source_position().tolist(),env.get_support_position().tolist(),events,maxstable,d)
def classify_failure(e):
    if not e["reached_source"] or not e["gripper_closed"]:return "grasp_approach_failure"
    if not e["source_lifted"]:return "grasp_lift_failure"
    if not e["transported_toward_support"]:return "transport_alignment_failure"
    if not e["source_support_contact"]:return "placement_contact_failure"
    if not e["released_after_close"]:return "release_failure"
    if not e["gripper_retreated"]:return "retreat_failure"
    return "post_release_instability"
