"""Raw NIfTI -> independent spatial registration -> native-time ROI records.

No released preprocessed BOLD is read. Raw-frame annotations and timing must
be explicitly verified; processed movie masks cannot be attached by length.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import nibabel as nib
import numpy as np

from .data import load_record, manifest, sha256
from .prepare_dryad import member
from .raw_spatial import SpatialConfig, image_from_array, motion_summary, register_frame

SOURCE_KIND = 'raw_framewise_spatial_v1'


def raw_geometry(image):
    if len(image.shape) != 4 or image.shape[-1] < 2:
        raise ValueError('raw input must be a 4D NIfTI')
    space, time = image.header.get_xyzt_units()
    if space != 'mm' or time not in ('sec', 'msec', 'usec'):
        raise ValueError('explicit millimeter/temporal units required')
    tr = float(image.header.get_zooms()[3]) * {'sec': 1., 'msec': .001, 'usec': 1e-6}[time]
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError('positive finite raw TR required')
    return tr


def binary_annotation(path, frames):
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.bool_ or array.shape != (frames,):
        raise ValueError('raw annotations must be boolean vectors of full raw-run length')
    return array


def checked_indices(row, frames, tr):
    timing = row.get('timing', {})
    if timing.get('verified') is not True or not timing.get('evidence'):
        raise ValueError('verified raw-to-movie timing evidence is required; no default burn-in assumption')
    for key in ('analysis_start_frame', 'analysis_stop_frame', 'reference_frame', 'burn_in_frames'):
        if type(row.get(key)) is not int:
            raise ValueError('explicit integer '+key+' required')
    start, stop, reference, burn = (row[k] for k in
                                  ('analysis_start_frame', 'analysis_stop_frame', 'reference_frame', 'burn_in_frames'))
    if not 0 <= burn <= start < stop <= frames or not 2 <= stop-start <= 256:
        raise ValueError('analysis must be a contiguous post-burn-in raw interval of 2..256 frames')
    if not burn <= reference < frames or start <= reference < stop:
        raise ValueError('calibration reference must be post-burn-in and outside the scored analysis interval')
    onset = timing.get('movie_onset_raw_seconds')
    if not isinstance(onset, (float, int)) or not np.isfinite(onset) or onset < 0 or onset >= frames*tr:
        raise ValueError('explicit movie onset in raw acquisition seconds required')
    return np.arange(start, stop, dtype=np.int64), reference


def prepare_volume(image, row, roi, registration_mask, motion_bad, gaze_bad, gaze_known,
                   config=SpatialConfig(), progress=None):
    """Pure per-run preparation from a raw image and frozen external annotations.

    Motion/gaze flags must already be on raw indices. Motion diagnostics are
    saved separately; they do not silently replace the supplied censor mask.
    """
    config.validate()
    tr = raw_geometry(image)
    frames = image.shape[-1]
    indices, reference_index = checked_indices(row, frames, tr)
    for array in (motion_bad, gaze_bad, gaze_known):
        if array.dtype != np.bool_ or array.shape != (frames,):
            raise ValueError('boolean annotations on the complete raw grid required')
    if motion_bad[reference_index]:
        raise ValueError('calibration reference is marked motion-corrupted')
    roi, registration_mask = np.asarray(roi), np.asarray(registration_mask)
    if roi.ndim != 2 or roi.shape[1] != 3 or roi.dtype.kind not in 'iu' or not 1 <= len(roi) <= 256:
        raise ValueError('explicit integer V x 3 ROI with 1..256 voxels required')
    if (roi < 0).any() or (roi >= np.array(image.shape[:3])).any() or len(np.unique(roi, axis=0)) != len(roi):
        raise ValueError('ROI indices must be unique and inside the reference grid')
    if registration_mask.dtype != np.bool_ or registration_mask.shape != image.shape[:3]:
        raise ValueError('registration mask must be boolean and match the reference grid')
    if not registration_mask[tuple(roi.T)].all():
        raise ValueError('ROI must lie inside the supplied registration mask')
    reference = np.asarray(image.dataobj[..., reference_index], dtype=np.float32)
    # Geometry and mask validation also occurs for the calibration frame.
    reference_image = image_from_array(reference, image.affine)
    values = np.empty((len(roi), len(indices)), dtype=np.float32)
    coverage = np.empty(values.shape, dtype=bool)
    transforms, quality = [], []
    for j, raw_index in enumerate(indices):
        frame = np.asarray(image.dataobj[..., int(raw_index)], dtype=np.float32)
        aligned, covered, details = register_frame(frame, reference, image.affine, registration_mask, config)
        values[:, j] = aligned[tuple(roi.T)]
        coverage[:, j] = covered[tuple(roi.T)]
        transforms.append(details['fixed_to_moving_lps'])
        quality.append(dict(raw_frame=int(raw_index), **details))
        if progress:
            progress(j+1, len(indices))
    clean = ~motion_bad[indices] & ~gaze_bad[indices] & gaze_known[indices]
    valid = np.broadcast_to(clean, values.shape).copy() & coverage & np.isfinite(values)
    recoverable = np.broadcast_to(motion_bad[indices] & ~gaze_bad[indices] & gaze_known[indices], values.shape).copy()
    if not valid.any():
        raise ValueError('no valid raw observations after annotations and spatial coverage')
    # Finite transformed corrupted values are never retained as truth.
    values = np.where(valid, values, np.nan)
    times = indices.astype(np.float32) * tr
    center = reference_image.TransformContinuousIndexToPhysicalPoint(tuple((np.array(image.shape[:3])-1)/2))
    diagnostics = motion_summary(transforms, center, config)
    receipt = dict(source_kind=SOURCE_KIND, timing_verified=True, reference_frame=reference_index,
                   reference_outside_analysis=True, temporal_operations=[],
                   spatial_operation='independent 6-DOF correlation registration; linear spatial resampling',
                   annotation_policy='supplied verified raw-frame flags; diagnostic RMS does not alter them',
                   initialization='random', spatial_config=asdict(config),
                   timing=row['timing'], roi_evidence=row.get('roi_evidence'),
                   calibration_scope='one observed raw frame excluded from every training/evaluation target')
    arrays = dict(values=values, valid=valid, recoverable=recoverable,
                  coords_mm=nib.affines.apply_affine(image.affine, roi).astype(np.float32),
                  times_s=times, raw_frame_indices=indices, voxel_ijk=roi,
                  movie_times_s=times-row['timing']['movie_onset_raw_seconds'],
                  motion_bad=motion_bad[indices], gaze_bad=gaze_bad[indices], gaze_known=gaze_known[indices],
                  padding=np.zeros(len(indices), dtype=bool), source_kind=np.array(SOURCE_KIND),
                  preparation_receipt=np.array(json.dumps(receipt, sort_keys=True)),
                  fixed_to_moving_lps=np.asarray(transforms),
                  diagnostic_rms_step_mm=diagnostics['rms_step_mm'])
    return arrays, dict(receipt=receipt, registration=quality,
                        motion_diagnostic_definition=diagnostics['definition'],
                        diagnostic_first_frame='RMS step set to zero: no preceding frame in selected interval')


def prepare(selection_path, root, output):
    selection_path, root, output = Path(selection_path).resolve(), Path(root), Path(output)
    plan = json.loads(selection_path.read_text())
    rows = plan.get('records', [])
    if not rows:
        raise ValueError('nonempty records and explicit spatial settings required')
    if not isinstance(plan.get('spatial'), dict):
        raise ValueError('explicit spatial settings required')
    config = SpatialConfig(**plan['spatial'])
    config.validate()
    families, ids = {}, set()
    # Check missing evidence before extraction/registration creates any output.
    for row in rows:
        if row.get('movie') == 'Child_Play':
            raise ValueError('Child_Play needs independently mapped continuous clips')
        ppt, movie = row['ppt'], row['movie']
        if '/' in movie or '\\' in movie or movie in ('.', '..'):
            raise ValueError('invalid movie archive name')
        if row['family'] != ppt.split('_')[0] or row['split'] not in ('train', 'val', 'test'):
            raise ValueError('invalid family or split')
        identity = movie+':'+ppt
        if identity in ids or (row['family'] in families and families[row['family']] != row['split']):
            raise ValueError('duplicate record or family leakage')
        ids.add(identity)
        families[row['family']] = row['split']
        raw_member = row['raw_member']
        if not raw_member.startswith(movie+'/raw_nifti/'+ppt+'_functional') or not raw_member.endswith(('.nii', '.nii.gz')):
            raise ValueError('only this participant\'s raw_nifti member is accepted')
        if row.get('timing', {}).get('verified') is not True or not row.get('timing', {}).get('evidence'):
            raise ValueError('verified raw-to-movie timing evidence required')
        if not row.get('annotation_evidence') or not row.get('roi_evidence'):
            raise ValueError('raw annotation and independent anatomical ROI evidence required')
        for key in ('roi_voxels', 'registration_mask', 'raw_motion_bad', 'raw_gaze_bad', 'raw_gaze_known'):
            if not (selection_path.parent / row[key]).is_file():
                raise ValueError('missing '+key)
    output.mkdir(parents=True, exist_ok=False)
    manifest_rows, receipts = [], []
    for number, row in enumerate(rows):
        def path(key):
            return (selection_path.parent / row[key]).resolve()
        with zipfile.ZipFile(root/(row['movie']+'.zip')) as archive, tempfile.TemporaryDirectory(prefix='infant-raw-') as temporary:
            # Decompress once into an uncompressed NIfTI for independent volume reads.
            import gzip
            destination = Path(temporary)/'raw.nii'
            digest = hashlib.sha256()
            with member(archive, row['raw_member']) as source:
                stream = gzip.GzipFile(fileobj=source) if row['raw_member'].endswith('.gz') else source
                with destination.open('wb') as target:
                    while True:
                        block = stream.read(1024*1024)
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
            image = nib.load(destination)
            frames = image.shape[-1]
            arrays, qc = prepare_volume(image, row, np.load(path('roi_voxels'), allow_pickle=False),
                                        np.load(path('registration_mask'), allow_pickle=False),
                                        *(binary_annotation(path(k), frames) for k in
                                          ('raw_motion_bad', 'raw_gaze_bad', 'raw_gaze_known')),
                                        config=config,
                                        progress=lambda j,n: print(json.dumps(dict(id=row['movie']+':'+row['ppt'], frame=j, total=n)), flush=True))
            target = output/f'{number:03d}_{row["movie"]}_{row["ppt"]}.npz'
            np.savez_compressed(target, **arrays)
            load_record(target)
            qc.update(raw_member=row['raw_member'], raw_uncompressed_sha256=digest.hexdigest(),
                      raw_member_crc32=archive.getinfo(row['raw_member']).CRC,
                      selection=row, annotation_sha256={k:sha256(path(k)) for k in
                      ('roi_voxels','registration_mask','raw_motion_bad','raw_gaze_bad','raw_gaze_known')},
                      record_sha256=sha256(target))
            (output/(target.stem+'.json')).write_text(json.dumps(qc, indent=2)+'\n')
            receipts.append(qc)
            manifest_rows.append(dict(id=row['movie']+':'+row['ppt'], family=row['family'],
                                      split=row['split'], path=target.name, source_kind=SOURCE_KIND))
    (output/'manifest.json').write_text(json.dumps(manifest_rows, indent=2)+'\n')
    manifest(output/'manifest.json')
    (output/'preparation.json').write_text(json.dumps(dict(source_kind=SOURCE_KIND,
        selection_sha256=sha256(selection_path), records=len(receipts), initialization='random',
        limitations='Minimal spatial path; susceptibility correction, anatomical ROI/registration QC and '
                    'scientific outcome validation remain separate requirements.'), indent=2)+'\n')
    return output/'manifest.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', required=True)
    args=parser.parse_args()
    prepare(args.selection,args.data_root,args.out)


if __name__ == '__main__':
    main()
