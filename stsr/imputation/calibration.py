"""Reserved-prefix calibration and native-slice QC for raw BOLD imputation.

The slice metrics/thresholds are declared engineering rules, not validated
infant diagnostic cutoffs. Anatomical alignment uses no analysis BOLD image.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from .raw_spatial import image_from_array, transform_matrix


@dataclass(frozen=True)
class CalibrationConfig:
    start: int = 3
    stop: int = 11
    max_alternation: float = .04
    max_slice_residual: float = .35
    min_slices: int = 6
    anatomy_iterations: int = 150
    max_anatomy_correction_mm: float = 15.


def slice_quality(volume, brain, config=CalibrationConfig()):
    if volume.ndim!=3 or brain.shape!=volume.shape or brain.dtype!=np.bool_:
        raise ValueError('matching 3D volume and boolean brain mask required')
    if not np.isfinite(volume).all():
        return dict(known=False,passed=False,alternation=None,residual_p90=None,slices=0)
    residual=[]; indices=[]
    for k in range(1,volume.shape[2]-1):
        mask=brain[:,:,k-1]&brain[:,:,k]&brain[:,:,k+1]
        if mask.sum()<32: continue
        mid=volume[:,:,k][mask].astype(float)
        neighbor=(volume[:,:,k-1][mask].astype(float)+volume[:,:,k+1][mask])/2
        floor=max(float(np.median(np.abs(neighbor)))*.1,1e-6)
        residual.append(float(np.median((mid-neighbor)/np.maximum(abs(neighbor),floor))))
        indices.append(k)
    if len(indices)<config.min_slices:
        return dict(known=False,passed=False,alternation=None,residual_p90=None,slices=len(indices))
    alternating=float(abs(np.median(np.asarray(residual)*((-1.)**np.asarray(indices)))))
    p90=float(np.percentile(abs(np.asarray(residual)),90))
    return dict(known=True,passed=alternating<=config.max_alternation and p90<=config.max_slice_residual,
                alternation=alternating,residual_p90=p90,slices=len(indices))


def choose_reference(prefix, brain, config=CalibrationConfig()):
    """Only the reserved prefix is accepted, preventing analysis access."""
    if prefix.ndim!=4 or prefix.shape[3]!=config.stop:
        raise ValueError('provide exactly the reserved prefix, not the analysis time series')
    candidates=[]
    for index in range(config.start,config.stop):
        quality=slice_quality(prefix[...,index],brain,config)
        candidates.append(dict(raw_frame=index,**quality))
    eligible=[r for r in candidates if r['passed']]
    if not eligible: raise ValueError('no reserved-prefix frame passes native slice QC')
    chosen=min(eligible,key=lambda r:(r['alternation']/config.max_alternation+r['residual_p90']/config.max_slice_residual,r['raw_frame']))
    return chosen['raw_frame'],candidates


def project(array,mapping,shape,order=0):
    points=mapping[:3,:3]@np.indices(shape).reshape(3,-1)+mapping[:3,3,None]
    return ndimage.map_coordinates(array,points,order=order,mode='constant',cval=0,prefilter=False).reshape(shape)


def anatomical_refinement(reference, anatomy_in_raw, initial_brain, affine, config=CalibrationConfig()):
    """Correct the published affine's head pose using calibration vs anatomy.

    The published transform is an external initializer, whose original
    estimation may have used analysis frames. Independence is conditional
    on that frozen external metadata. No released functional centroid is used.
    """
    import SimpleITK as sitk
    mask=ndimage.binary_dilation(initial_brain,iterations=2)&(reference>0)
    if mask.sum()<100: raise ValueError('insufficient anatomy registration support')
    fixed=image_from_array(reference.astype(np.float32),affine)
    moving=image_from_array(anatomy_in_raw.astype(np.float32),affine)
    registration=sitk.ImageRegistrationMethod(); registration.SetNumberOfThreads(2)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricFixedMask(image_from_array(mask.astype(np.uint8),affine))
    registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetInterpolator(sitk.sitkLinear)
    transform=sitk.Euler3DTransform(); transform.SetComputeZYX(True)
    center=fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(reference.shape)-1)/2))
    transform.SetCenter(center); registration.SetInitialTransform(transform,inPlace=True)
    before=float(registration.MetricEvaluate(fixed,moving))
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.,minStep=.001,
        numberOfIterations=config.anatomy_iterations,gradientMagnitudeTolerance=1e-8)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4,2,1]); registration.SetSmoothingSigmasPerLevel([2.,1.,0.])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.Execute(fixed,moving)
    after=float(registration.MetricEvaluate(fixed,moving)); matrix=transform_matrix(transform)
    delta=matrix-np.eye(4)
    rms=float(np.sqrt(35.**2/3*np.sum(delta[:3,:3]**2)+np.sum((delta[:3,:3]@center+delta[:3,3])**2)))
    if not np.isfinite(matrix).all() or not np.isfinite(after) or after>before+.001:
        raise ValueError('anatomical MI refinement failed to improve its objective')
    if rms>config.max_anatomy_correction_mm:
        raise ValueError('anatomical correction exceeds predefined displacement limit')
    return matrix,dict(metric_before=before,metric_after=after,rms_correction_mm=rms,
        fixed_to_moving_lps=matrix.tolist(),stop=registration.GetOptimizerStopConditionDescription(),
        limitation='MI improvement is not an independent anatomical accuracy certificate')


def voxel_mapping(affine,lps_transform):
    flip=np.diag([-1.,-1.,1.,1.])
    return np.linalg.inv(affine)@flip@lps_transform@flip@affine
