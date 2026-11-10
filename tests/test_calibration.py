import numpy as np
import pytest
import json
from stsr.imputation.calibration import slice_quality, choose_reference, voxel_mapping


def phantom():
    xyz=np.indices((20,20,20)); volume=100+xyz[0]+xyz[1]+.1*(xyz[2]-10)**2
    mask=np.zeros(volume.shape,bool); mask[2:-2,2:-2,2:-2]=True
    return volume,mask


def test_native_slice_qc_detects_alternating_gain_and_dropout():
    volume,mask=phantom()
    assert slice_quality(volume,mask)['passed']
    striped=volume*(1+.2*((-1.)**np.arange(20)))[None,None,:]
    assert not slice_quality(striped,mask)['passed']
    dropout=volume.copy(); dropout[:,:,8:10]=0
    assert not slice_quality(dropout,mask)['passed']


def test_reference_selector_cannot_read_scored_analysis_or_pick_striped_frame():
    volume,mask=phantom(); prefix=np.repeat(volume[...,None],11,axis=3)
    prefix[:,:,:,3:10]*=(1+.2*((-1.)**np.arange(20)))[None,None,:,None]
    index,report=choose_reference(prefix,mask)
    assert index==10 and len(report)==8
    with pytest.raises(ValueError,match='exactly the reserved prefix'):
        choose_reference(np.repeat(volume[...,None],30,axis=3),mask)
    prefix[...,10]=prefix[...,3]
    with pytest.raises(ValueError,match='no reserved-prefix'): choose_reference(prefix,mask)


def test_lps_voxel_transform_round_trip():
    affine=np.diag([-3.,3.,3.,1.]); affine[:3,3]=[90,-100,20]
    matrix=np.eye(4); matrix[:3,3]=[6,9,-3]
    converted=voxel_mapping(affine,matrix)
    np.testing.assert_allclose(converted@voxel_mapping(affine,np.linalg.inv(matrix)),np.eye(4),atol=1e-12)


def test_v2_source_and_training_gate_cannot_bypass_calibration(tmp_path):
    from stsr.imputation.runner import validate_source,validate_cohort_qa
    cfg=dict(required_source_kind='raw_framewise_qc_v2')
    receipt=dict(source_kind='raw_framewise_qc_v2',timing_policy='raw_acquisition_only',gaze_policy='unconditioned_unknown',
        timing_verified=False,qc_frozen_before_artificial_holdouts=True,reference_outside_analysis=True,temporal_operations=[],
        reference_frame=4,calibration_prefix_stop=11,brain_interior_only=True,native_slice_qc=True,
        calibration_anatomy_refined=True,unique_acquisition_sha256='a'*64)
    path=tmp_path/'record.npz'
    def save(): np.savez(path,source_kind='raw_framewise_qc_v2',preparation_receipt=json.dumps(receipt))
    save(); validate_source(path,cfg)
    receipt['reference_frame']=20; save()
    with pytest.raises(ValueError,match='reserved-prefix'): validate_source(path,cfg)
    with pytest.raises(ValueError,match='missing qa_gate'): validate_cohort_qa(tmp_path/'manifest.json',[],cfg)
