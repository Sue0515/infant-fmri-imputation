"""Prepare explicitly selected native-space ROI records from the Dryad ZIPs.

This is a preprocessed-data wiring pilot. Stored interpolated motion values
are discarded. It does not undo temporal filtering/interpolation already done
by the source pipeline, nor establish raw-data reconstruction ground truth.
"""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import struct
import zipfile

import nibabel as nib
import numpy as np

from .data import load_record, manifest, sha256


def member(archive, name):
    """Work around wrapped 32-bit offsets in source ZIPs, in memory only."""
    info = archive.getinfo(name)
    with open(archive.filename, 'rb') as stream:
        stream.seek(info.header_offset)
        if stream.read(4) != b'PK\x03\x04':
            candidates = []
            for offset in range(info.header_offset % 2**32, archive.start_dir, 2**32):
                stream.seek(offset)
                header = stream.read(30)
                if len(header) == 30 and header[:4] == b'PK\x03\x04':
                    length = struct.unpack('<H', header[26:28])[0]
                    if stream.read(length).decode('utf-8', errors='replace') == name:
                        candidates.append(offset)
            if len(candidates) != 1:
                raise ValueError('ambiguous ZIP member offset: ' + name)
            info.header_offset = candidates[0]
            if hasattr(info, '_end_offset'):
                info._end_offset = archive.start_dir
    return archive.open(info)


def read(archive, name):
    with member(archive, name) as stream:
        return stream.read()


def confound(archive, name, frames):
    data = np.loadtxt(io.BytesIO(read(archive, name)), ndmin=2)
    if data.shape[0] != frames or not np.isin(data, [0, 1]).all():
        raise ValueError('expected binary full-time-axis confounds: ' + name)
    return data.any(axis=1)


def read_roi(archive, name, ijk):
    """Stream one spatial volume at a time; retain original T and affine."""
    if ijk.ndim != 2 or ijk.shape[1] != 3 or ijk.dtype.kind not in 'iu' or not 1 <= len(ijk) <= 256:
        raise ValueError('voxel_ijk must contain 1..256 integer voxel coordinates')
    if len(np.unique(ijk, axis=0)) != len(ijk):
        raise ValueError('voxel coordinates must be unique')
    with member(archive, name) as raw, gzip.GzipFile(fileobj=raw) as stream:
        header = nib.Nifti1Header.from_fileobj(stream)
        shape = header.get_data_shape()
        if len(shape) != 4 or not 2 <= shape[-1] <= 256:
            raise ValueError('expected 4D NIfTI with at most 256 frames')
        if (ijk < 0).any() or (ijk >= np.array(shape[:3])).any():
            raise ValueError('ROI coordinates outside NIfTI')
        units, time_units = header.get_xyzt_units()
        if units != 'mm' or time_units not in ('sec', 'msec', 'usec'):
            raise ValueError('explicit spatial/temporal NIfTI units required')
        tr = header.get_zooms()[3] * {'sec': 1, 'msec': .001, 'usec': .000001}[time_units]
        dtype = header.get_data_dtype()
        count = int(np.prod(shape[:3]))
        indices = np.ravel_multi_index(ijk.T, shape[:3], order='F')
        offset = int(header['vox_offset'])
        remaining_header = offset - stream.tell()
        if remaining_header < 0 or len(stream.read(remaining_header)) != remaining_header:
            raise ValueError('invalid NIfTI data offset')
        slope, intercept = header.get_slope_inter()
        slope, intercept = (1., 0.) if slope is None else (slope, intercept)
        values = np.empty((len(ijk), shape[-1]), dtype=np.float32)
        for t in range(shape[-1]):
            block = stream.read(count * dtype.itemsize)
            if len(block) != count * dtype.itemsize:
                raise ValueError('truncated NIfTI volume')
            values[:, t] = np.frombuffer(block, dtype=dtype)[indices] * slope + intercept
        # Finish both streams so gzip/ZIP CRC checks cover this selected member.
        while stream.read(1024 * 1024):
            pass
        while raw.read(1024 * 1024):
            pass
    affine = header.get_best_affine()
    return values, nib.affines.apply_affine(affine, ijk).astype(np.float32), np.arange(shape[-1], dtype=np.float32) * tr


def validity(motion, eye, num_tr, *, eye_shift_tr, guard_tr):
    frames = len(motion)
    if eye.shape != motion.shape or not 1 <= num_tr <= frames:
        raise ValueError('invalid effective acquisition length')
    if not 0 <= eye_shift_tr < frames or guard_tr < 0:
        raise ValueError('nonnegative eye shift/guard required; shift must fit acquisition')
    padding = np.arange(frames) >= num_tr
    shifted = np.zeros(frames, dtype=bool)
    if eye_shift_tr:
        shifted[eye_shift_tr:] = eye[:-eye_shift_tr]
        shifted[:eye_shift_tr] = True  # no pre-movie gaze observations
    else:
        shifted = eye.copy()
    exclusion = padding | shifted
    bad = motion | exclusion
    guarded = bad.copy()
    for lag in range(1, min(guard_tr, frames - 1) + 1):
        guarded[lag:] |= bad[:-lag]
        guarded[:-lag] |= bad[lag:]
    valid = ~guarded
    # Guard-band samples remain excluded; fill explicitly marked motion only.
    recoverable = motion & ~exclusion
    return valid, recoverable, shifted, padding


def prepare(selection_path, root, output, *, eye_shift_tr, guard_tr=0):
    selection_path, root, output = Path(selection_path).resolve(), Path(root), Path(output)
    selection = json.loads(selection_path.read_text())
    if not isinstance(selection, list) or not selection:
        raise ValueError('selection must be a nonempty list')
    # Check identities before any signal extraction.
    families, ids = {}, set()
    for row in selection:
        for key in ('movie', 'ppt', 'family', 'split', 'voxel_ijk'):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError('selection needs movie, ppt, family, split, voxel_ijk strings')
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('invalid split')
        if row['movie'] == 'Child_Play':
            raise ValueError('Child_Play has removed rest intervals; continuous segment mapping is required first')
        if row['family'] != row['ppt'].split('_')[0]:
            raise ValueError('Dryad family must match the sXXXX participant prefix')
        if row['family'] in families and families[row['family']] != row['split']:
            raise ValueError('family leakage across splits')
        families[row['family']] = row['split']
        identity = (row['movie'], row['ppt'])
        if identity in ids or any('/' in v or '\\' in v or v in ('.', '..') for v in identity):
            raise ValueError('duplicate or invalid movie/participant identity')
        ids.add(identity)
    output.mkdir(parents=True, exist_ok=False)
    rows, provenance = [], []
    for index, row in enumerate(selection):
        movie, ppt = row['movie'], row['ppt']
        ijk_path = (selection_path.parent / row['voxel_ijk']).resolve()
        ijk = np.load(ijk_path, allow_pickle=False)
        archive_path = root / (movie + '.zip')
        source = f'{movie}/preprocessed_native/linear_alignment/{ppt}_Z.nii.gz'
        with zipfile.ZipFile(archive_path) as archive:
            table = list(csv.DictReader(io.StringIO(read(archive, f'{movie}/infant_participants.csv').decode('utf-8-sig'))))
            matches = [r for r in table if r['ppt'] == ppt]
            if len(matches) != 1:
                raise ValueError('one infant CSV record required: ' + ppt)
            participant = matches[0]
            try:
                gaze = float(participant['prop_TR_eye'])
            except (KeyError, ValueError):
                gaze = float('nan')
            if not np.isfinite(gaze):
                raise ValueError('missing gaze metadata; cannot label clean targets: ' + ppt)
            x, coords, times = read_roi(archive, source, ijk)
            motion = confound(archive, f'{movie}/motion_confounds/{ppt}.txt', x.shape[-1])
            eye = confound(archive, f'{movie}/eye_confounds/{ppt}.txt', x.shape[-1])
            v, recover, shifted, padding = validity(motion, eye, int(participant['num_TR']),
                                                   eye_shift_tr=eye_shift_tr, guard_tr=guard_tr)
            valid = np.broadcast_to(v, x.shape).copy() & np.isfinite(x)
            recoverable = np.broadcast_to(recover, x.shape).copy()
            # Do not retain the preprocessor's interpolation as pseudo ground truth.
            x = np.where(valid, x, np.nan)
            destination = output / f'{index:03d}_{movie}_{ppt}.npz'
            np.savez_compressed(destination, values=x, coords_mm=coords, times_s=times,
                                valid=valid, recoverable=recoverable, voxel_ijk=ijk,
                                motion_bad=motion, eye_bad_shifted=shifted, padding=padding)
            load_record(destination)
            rows.append(dict(id=movie + ':' + ppt, family=row['family'], split=row['split'], path=destination.name))
            provenance.append(dict(source_archive=str(archive_path.resolve()), member=source,
                                   member_crc32=archive.getinfo(source).CRC, participant=participant,
                                   voxel_ijk_sha256=sha256(ijk_path), record_sha256=sha256(destination),
                                   valid_values=int(valid.sum()), recoverable_values=int(recoverable.sum())))
        print(json.dumps({'id': rows[-1]['id'], 'shape': list(x.shape),
                          'valid_values': int(valid.sum()), 'recoverable_values': int(recoverable.sum())}), flush=True)
    (output / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    manifest(output / 'manifest.json')
    (output / 'preparation.json').write_text(json.dumps(dict(
        source_kind='preprocessed_native_pilot', eye_shift_tr=eye_shift_tr, guard_tr=guard_tr,
        selection_sha256=sha256(selection_path), records=provenance,
        limitations='Temporal preprocessing is already applied. This is not leakage-free raw ground truth. '
                    'ROI selection and cross-subject anatomical correspondence must be established separately.'), indent=2) + '\n')
    return output / 'manifest.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', required=True, help='JSON list of movie,ppt,family,split,voxel_ijk')
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--eye-shift-tr', type=int, required=True, help='explicit protocol choice; not assumed across studies')
    p.add_argument('--guard-tr', type=int, default=0)
    args = p.parse_args()
    prepare(args.selection, args.data_root, args.out, eye_shift_tr=args.eye_shift_tr, guard_tr=args.guard_tr)


if __name__ == '__main__':
    main()
