import json
import numpy as np
import pytest
from stsr.imputation.prepare_raw_qc import classify_frames, spatially_spread, SOURCE_KIND
from stsr.imputation.runner import validate_source


def test_failed_frame_and_neighbor_cannot_supply_clean_targets():
    reference=np.array([100.,200.,300.])
    values=np.tile(reference[:,None],(1,5))
    values[:,2]=np.nan
    clean,motion,signal,dvars=classify_frames(values,reference,np.isfinite(values),
        np.zeros(5),np.zeros(5),np.ones(5),np.array([False,False,True,False,False]))
    assert clean.tolist()==[True,True,False,False,True]
    assert not motion.any()


def test_motion_and_intensity_jump_are_separate_qc_exclusions():
    reference=np.array([100.,200.,300.])
    values=np.tile(reference[:,None],(1,4)); values[:,-1]*=1.5
    clean,motion,signal,_=classify_frames(values,reference,np.ones_like(values,dtype=bool),
        np.array([0.,2.,0.,0.]),np.zeros(4),np.ones(4),np.zeros(4,dtype=bool))
    assert clean.tolist()==[True,False,True,False]
    assert motion.tolist()==[False,True,False,False]
    assert signal.tolist()==[False,False,False,True]


def test_spatial_selection_has_no_signal_dependence_and_no_duplicates():
    xyz=np.indices((7,8,9)).reshape(3,-1).T
    chosen=spatially_spread(xyz,32)
    assert len(np.unique(chosen,axis=0))==32
    np.testing.assert_array_equal(chosen,spatially_spread(xyz,32))


def test_raw_qc_cannot_masquerade_as_verified_movie_cohort(tmp_path):
    path=tmp_path/'record.npz'
    receipt=dict(source_kind=SOURCE_KIND,timing_policy='raw_acquisition_only',timing_verified=False,
        gaze_policy='unconditioned_unknown',reference_outside_analysis=True,temporal_operations=[],
        qc_frozen_before_artificial_holdouts=True)
    np.savez(path,source_kind=SOURCE_KIND,preparation_receipt=json.dumps(receipt))
    validate_source(path,dict(required_source_kind=SOURCE_KIND))
    with pytest.raises(ValueError,match='refuses'):
        validate_source(path,dict(required_source_kind='raw_framewise_spatial_v1'))
    receipt['timing_verified']=True
    np.savez(path,source_kind=SOURCE_KIND,preparation_receipt=json.dumps(receipt))
    with pytest.raises(ValueError,match='unverified movie/gaze'):
        validate_source(path,dict(required_source_kind=SOURCE_KIND))
