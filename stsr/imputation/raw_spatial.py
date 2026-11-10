"""Independent 3-D rigid registration, with no temporal signal processing.

Each output depends only on that frame, one fixed calibration frame, and a
fixed mask. The calibration frame must never be a scored imputation target.
This is a minimal spatial preprocessing path, not a complete infant pipeline.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialConfig:
    iterations: int = 100
    learning_rate: float = 1.0
    min_step: float = 0.001
    threads: int = 2
    motion_threshold_mm: float = 3.0
    rotation_radius_mm: float = 35.0

    def validate(self):
        if self.iterations < 1 or self.threads < 1:
            raise ValueError('positive registration iteration/thread counts required')
        for name in ('learning_rate', 'min_step', 'motion_threshold_mm', 'rotation_radius_mm'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('positive finite spatial parameter: '+name)


def image_from_array(array, affine):
    import SimpleITK as sitk
    array, affine = np.asarray(array), np.asarray(affine, dtype=float)
    if array.ndim != 3 or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('3D volume and finite 4x4 affine required')
    if not np.isfinite(array).all():
        raise ValueError('registration volume must be finite')
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError('invalid homogeneous affine')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (spacing <= 0).any():
        raise ValueError('positive voxel spacing required')
    direction = affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('sheared affine requires explicit geometry handling')
    ras_to_lps = np.diag([-1., -1., 1.])
    image = sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2, 1, 0)))
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(ras_to_lps @ affine[:3, 3]))
    image.SetDirection(tuple((ras_to_lps @ direction).ravel()))
    return image


def transform_matrix(transform):
    origin = np.asarray(transform.TransformPoint((0., 0., 0.)))
    matrix = np.eye(4)
    matrix[:3, 3] = origin
    for i in range(3):
        point = np.eye(3)[i]
        matrix[:3, i] = np.asarray(transform.TransformPoint(tuple(point))) - origin
    return matrix


def register_frame(frame, reference, affine, mask, config=SpatialConfig()):
    """Return resampled values, coverage and fixed-to-moving LPS transform.

    Every optimizer is initialized independently. There is no temporal warm
    start, mean/centroid reference, slice timing, despiking, or normalization.
    """
    import SimpleITK as sitk
    config.validate()
    frame, reference, mask = np.asarray(frame), np.asarray(reference), np.asarray(mask)
    if frame.shape != reference.shape or mask.shape != reference.shape or mask.dtype != np.bool_:
        raise ValueError('matching 3-D frame/reference and boolean registration mask required')
    if mask.sum() < 32 or np.std(reference[mask]) <= 1e-6:
        raise ValueError('registration needs a nonconstant reference and at least 32 mask voxels')
    fixed = image_from_array(reference.astype(np.float32), affine)
    moving = image_from_array(frame.astype(np.float32), affine)
    fixed_mask = image_from_array(mask.astype(np.uint8), affine)
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(config.threads)
    registration.SetMetricAsCorrelation()
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetInterpolator(sitk.sitkLinear)
    transform = sitk.Euler3DTransform()
    transform.SetComputeZYX(True)
    center = fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(reference.shape)-1)/2))
    transform.SetCenter(center)
    registration.SetInitialTransform(transform, inPlace=True)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=config.learning_rate, minStep=config.min_step,
        numberOfIterations=config.iterations, gradientMagnitudeTolerance=1e-8)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1., 0.])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.Execute(fixed, moving)
    matrix = transform_matrix(transform)
    if not np.isfinite(matrix).all() or not np.isfinite(registration.GetMetricValue()):
        raise ValueError('nonfinite spatial registration result')
    resampler = sitk.ResampleImageFilter()
    resampler.SetNumberOfThreads(config.threads)
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    aligned = sitk.GetArrayFromImage(resampler.Execute(moving)).transpose(2, 1, 0)
    coverage_image = image_from_array(np.ones(frame.shape, dtype=np.float32), affine)
    coverage = sitk.GetArrayFromImage(resampler.Execute(coverage_image)).transpose(2, 1, 0) > .999
    aligned = np.where(coverage, aligned, np.nan).astype(np.float32)
    return aligned, coverage, dict(fixed_to_moving_lps=matrix.tolist(),
                                  metric=float(registration.GetMetricValue()),
                                  stop=registration.GetOptimizerStopConditionDescription(),
                                  iterations=int(registration.GetOptimizerIteration()),
                                  parameters=list(transform.GetParameters()))


def motion_summary(matrices, center_lps, config=SpatialConfig()):
    """RMS displacement of 6 points at a declared radius; not FSL/Power FD.

    Thresholding this diagnostic is a protocol choice requiring sensitivity
    analysis. It is not claimed equivalent to the released motion mask.
    """
    config.validate()
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all():
        raise ValueError('finite T x 4 x 4 transforms required')
    center = np.asarray(center_lps, dtype=float)
    points = np.vstack([np.eye(3), -np.eye(3)]) * config.rotation_radius_mm + center
    points = np.c_[points, np.ones(6)]
    moved = np.einsum('tij,pj->tpi', matrices, points)[..., :3]
    delta = np.diff(moved, axis=0)
    rms = np.r_[0., np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=-1))]
    return dict(rms_step_mm=rms, motion_bad=rms > config.motion_threshold_mm,
                definition='RMS step of six calibration-space points under fixed-to-moving transforms',
                settings=asdict(config))
