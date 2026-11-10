import numpy as np
from stsr.imputation.raw_geometry import fsl_scaled_mm, raw_to_mask_voxels, resample_mask


def test_fsl_flips_radiological_handedness_with_corner_offset():
    a=np.diag([2.,3.,4.,1.])
    matrix=fsl_scaled_mm((7,8,9),a)
    np.testing.assert_allclose(matrix@np.array([0,0,0,1]),[12,0,0,1])
    np.testing.assert_allclose(matrix@np.array([6,2,1,1]),[0,6,4,1])
    a[0,0]=-2
    np.testing.assert_allclose(fsl_scaled_mm((7,8,9),a),np.diag([2,3,4,1]))


def test_anatomical_translation_and_different_mask_grid():
    raw=np.diag([-3.,3.,3.,1.])
    anatomical=np.diag([-1.,1.,1.,1.]); anatomical[:3,3]=[10,20,30]
    mask_affine=anatomical.copy(); mask_affine[:3,3]+=anatomical[:3,:3]@np.array([2,4,6])
    transform=np.eye(4); transform[:3,3]=[1,2,3]
    mapping=raw_to_mask_voxels((10,11,12),raw,(40,50,60),anatomical,mask_affine,transform)
    # Raw (2,3,4) -> FSL (6,9,12) -> anatomical (7,11,15) -> mask (5,7,9).
    np.testing.assert_allclose(mapping@np.array([2,3,4,1]),[5,7,9,1])
    mask=np.zeros((20,20,20),dtype=np.float32); mask[5,7,9]=1
    projected=resample_mask(mask,(10,11,12),mapping)
    assert projected[2,3,4] and projected.sum()==1


def test_positive_and_negative_handedness_same_fsl_point():
    raw=np.diag([2.,2.,2.,1.])
    anat=np.diag([-1.,1.,1.,1.])
    mapping=raw_to_mask_voxels((5,5,5),raw,(20,20,20),anat,anat,np.eye(4))
    np.testing.assert_allclose(mapping@np.array([0,2,1,1]),[8,4,2,1])
