import numpy as np,torch
from models.bc_policy import BCPolicy,BCPolicyAgent
from training.bc_dataset import BCTransitionDataset,compute_observation_statistics
def test_29d_dataset_and_model():
    o=np.zeros((8,29),np.float32);a=np.zeros((8,4),np.float32);d=BCTransitionDataset(o,a);m=BCPolicy(29,[128,128]);motion,grip=m(d.observations);assert motion.shape==(8,3) and grip.shape==(8,1)
def test_29d_normalization_and_agent():
    o=np.arange(290,dtype=np.float32).reshape(10,29);mean,std=compute_observation_statistics(o);assert mean.shape==(29,)
    m=BCPolicy(29,[8]);agent=BCPolicyAgent(m,mean,std,1e-6,.008,motion_output_scale=.008);assert agent.act(o[0]).shape==(4,)
def test_legacy_10d_behavior_remains_supported():
    o=np.zeros((3,10),np.float32);a=np.zeros((3,4),np.float32);assert len(BCTransitionDataset(o,a))==3;assert compute_observation_statistics(o)[0].shape==(10,)
