import numpy as np
from models.bc_policy import DebouncedGripperPolicyAgent
from models.stacking_features import RELATIVE_V1,transform_stacking_observations

def test_relative_features_preserve_shape_and_use_observable_geometry():
    x=np.zeros((2,29),dtype=np.float32);x[:,0:3]=[.4,.1,.3];x[:,3:6]=[.5,-.1,.05];x[:,15:18]=[.6,.2,.025]
    y=transform_stacking_observations(x,RELATIVE_V1)
    assert y.shape==x.shape
    np.testing.assert_allclose(y[:,3:6],x[:,3:6]-x[:,0:3])
    np.testing.assert_allclose(y[:,15:18],x[:,3:6]-x[:,15:18])
    np.testing.assert_allclose(y[:,6:15],x[:,6:15])

class _Policy:
    def __init__(self, probabilities):self.probabilities=iter(probabilities)
    def predict_batch(self, observations):return np.zeros((1,3)),np.array([next(self.probabilities)])

def test_gripper_debounce_requires_three_consistent_predictions():
    p=DebouncedGripperPolicyAgent(_Policy([.9,.1,.9,.9,.9,.1,.1,.1]),3);obs=np.zeros(29)
    commands=[p.act(obs)[3] for _ in range(8)]
    assert commands==[-1,-1,-1,-1,1,1,1,-1]
