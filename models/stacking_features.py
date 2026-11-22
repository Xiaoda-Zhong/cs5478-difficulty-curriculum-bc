"""Observable-only feature transforms shared by every future stacking BC method."""
from __future__ import annotations
import numpy as np

RELATIVE_V1 = "stacking_relative_v1"

def transform_stacking_observations(observations: np.ndarray, name: str) -> np.ndarray:
    """Replace absolute object positions by task-relevant relative geometry.

    The dimensionality stays 29: EE position remains absolute, source position becomes
    source-minus-EE, and support position becomes source-minus-support. All other
    measured channels remain unchanged.
    """
    x=np.asarray(observations,dtype=np.float32)
    if x.shape[-1] != 29: raise ValueError("stacking observations must end in 29 channels")
    if name != RELATIVE_V1: raise ValueError(f"unknown stacking feature transform: {name}")
    y=x.copy(); y[...,3:6]=x[...,3:6]-x[...,0:3]; y[...,15:18]=x[...,3:6]-x[...,15:18]
    return y

