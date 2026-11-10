"""Spatial metadata conversion using FSL scaled-mm (not NIfTI world-mm)."""
import gzip
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from .prepare_dryad import member


def fsl_scaled_mm(shape, affine):
    affine=np.asarray(affine,dtype=float)
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    if affine.shape!=(4,4) or not np.isfinite(affine).all() or (spacing<=0).any():
        raise ValueError('finite nondegenerate geometry required')
    matrix=np.diag(np.r_[spacing,1.])
    if np.linalg.det(affine[:3,:3])>0:
        matrix[0,0]=-spacing[0]
        matrix[0,3]=(shape[0]-1)*spacing[0]
    return matrix


def raw_to_mask_voxels(raw_shape,raw_affine,anatomical_shape,anatomical_affine,mask_affine,flirt):
    flirt=np.asarray(flirt,dtype=float)
    if flirt.shape!=(4,4) or not np.isfinite(flirt).all() or abs(np.linalg.det(flirt))<1e-8:
        raise ValueError('finite invertible FLIRT matrix required')
    return (np.linalg.inv(mask_affine) @ anatomical_affine @
            np.linalg.inv(fsl_scaled_mm(anatomical_shape,anatomical_affine)) @
            flirt @ fsl_scaled_mm(raw_shape,raw_affine))


def resample_mask(mask,raw_shape,raw_to_mask):
    if not np.isfinite(mask).all() or mask.min()<0 or mask.max()>1+1e-5:
        raise ValueError('mask must be finite and bounded in [0,1]')
    ijk=np.indices(raw_shape).reshape(3,-1)
    points=raw_to_mask[:3,:3]@ijk+raw_to_mask[:3,3,None]
    return map_coordinates(mask,points,order=0,mode='constant',cval=0).reshape(raw_shape)>.5


def read_raw_frame(archive,name,index):
    with member(archive,name) as stream, gzip.GzipFile(fileobj=stream) as raw:
        header=nib.Nifti1Header.from_fileobj(raw)
        shape=header.get_data_shape()
        if len(shape)!=4 or not 0<=index<shape[-1]:
            raise ValueError('raw calibration index outside image')
        size=int(np.prod(shape[:3]))*header.get_data_dtype().itemsize
        target=int(header['vox_offset'])+index*size
        left=target-raw.tell()
        while left:
            block=raw.read(min(left,1024*1024))
            if not block: raise ValueError('truncated raw image')
            left-=len(block)
        data=raw.read(size)
        if len(data)!=size: raise ValueError('truncated raw frame')
        array=np.frombuffer(data,dtype=header.get_data_dtype()).reshape(shape[:3],order='F').astype(np.float32)
        slope,intercept=header.get_slope_inter()
        if slope is not None: array=array*slope+intercept
        return array,header
