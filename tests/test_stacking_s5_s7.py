import json
from pathlib import Path
import numpy as np
from training.curriculum_sampler import eligible_trajectory_ids,stage_for_update
from training.train_curriculum import create_initial_state,state_dict_fingerprint

ROOT=Path(__file__).resolve().parents[1]

def test_stacking_rankings_are_exhaustive_equal_and_leakage_free():
    split=json.loads((ROOT/"data/stacking_v1/metadata/bc_split_v1.json").read_text())
    payload=json.loads((ROOT/"outputs/stacking_v1/results/difficulty/difficulty_rankings_v1.json").read_text())
    train=set(split["train_trajectory_ids"]);validation=set(split["validation_trajectory_ids"])
    for ranking in payload["rankings"].values():
        groups=ranking["groups"];assert [len(groups[x]) for x in ("easy","medium","hard")]==[80,80,80]
        ids=[v for x in groups.values() for v in x];assert len(ids)==len(set(ids))==240;assert set(ids)==train;assert not set(ids)&validation

def test_stacking_oof_is_disjoint_exhaustive():
    x=json.loads((ROOT/"outputs/stacking_v1/results/difficulty/learner_oof_folds_v1.json").read_text());held=[]
    for fold in x["folds"]:
        assert len(fold["held_out_trajectory_ids"])==48;assert not set(fold["held_out_trajectory_ids"])&set(fold["training_trajectory_ids"]);held+=fold["held_out_trajectory_ids"]
    assert len(held)==len(set(held))==240

def test_exact_stacking_stage_boundaries_and_matched_initialization():
    assert stage_for_update(6866,20600,[1/3,2/3])==1
    assert stage_for_update(6867,20600,[1/3,2/3])==2
    assert stage_for_update(13733,20600,[1/3,2/3])==2
    assert stage_for_update(13734,20600,[1/3,2/3])==3
    assert state_dict_fingerprint(create_initial_state(5478,[128,128],29))==state_dict_fingerprint(create_initial_state(5478,[128,128],29))
