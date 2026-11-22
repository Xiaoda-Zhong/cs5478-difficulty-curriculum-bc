from pathlib import Path
import json
import numpy as np
import pybullet as p
from data.stacking_dataset import inspect_stacking_dataset,load_stacking_transitions
from data.stacking_task_sampler import StackingTaskSampler
from data.stacking_trajectory import load_stacking_trajectory,validate_stacking_trajectory
from envs.panda_stacking import PandaStackingEnv

ROOT=Path(__file__).resolve().parents[1]
def test_sampler_determinism_and_tolerance():
    a,b=StackingTaskSampler(9),StackingTaskSampler(9)
    for _ in range(5):
        x,y=a.sample(),b.sample();np.testing.assert_array_equal(x.source_position,y.source_position);assert x.placement_tolerance==y.placement_tolerance
        assert .012<=x.placement_tolerance<=.015
def test_frozen_dataset_integrity():
    result=inspect_stacking_dataset(ROOT/"data/stacking_v1");assert result["trajectories"]==300 and result["content_hashes_verified"]
def test_archive_schema_and_preaction_alignment():
    t=load_stacking_trajectory(ROOT/"data/stacking_v1/demonstrations/traj_0000.npz");validate_stacking_trajectory(t)
    assert t["observations"].shape[1]==29 and t["actions"].shape[1]==4
    np.testing.assert_allclose(t["observations"][:,28],float(t["placement_tolerance"]))
    np.testing.assert_allclose(t["observations"][0,:3],t["ee_positions"][0],atol=1e-6)
def test_split_and_transition_loading():
    s=json.loads((ROOT/"data/stacking_v1/metadata/bc_split_v1.json").read_text());assert len(s["train_trajectory_ids"])==240 and len(s["validation_trajectory_ids"])==60
    o,a=load_stacking_transitions(ROOT/"data/stacking_v1/demonstrations",s["train_trajectory_ids"][:2]);assert o.shape[1]==29 and a.shape[1]==4
def test_placement_tolerance_affects_detector_and_observation():
    with PandaStackingEnv() as env:
        support=np.array([.55,.15,.025]);source=support+np.array([.0135,0,.05])
        env.reset(placement_tolerance=.012)
        p.resetBasePositionAndOrientation(env.support_cube_id,support,[0,0,0,1],physicsClientId=env.client_id)
        p.resetBasePositionAndOrientation(env.source_cube_id,source,[0,0,0,1],physicsClientId=env.client_id)
        assert env.get_observation()[28]==np.float32(.012) and not env.is_stack_geometrically_aligned()
        env.reset(placement_tolerance=.015)
        p.resetBasePositionAndOrientation(env.support_cube_id,support,[0,0,0,1],physicsClientId=env.client_id)
        p.resetBasePositionAndOrientation(env.source_cube_id,source,[0,0,0,1],physicsClientId=env.client_id)
        assert env.get_observation()[28]==np.float32(.015) and env.is_stack_geometrically_aligned()
