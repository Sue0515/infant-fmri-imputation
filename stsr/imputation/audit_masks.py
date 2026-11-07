"""Audit infant movie missingness from local ZIP metadata, without reading BOLD."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import nibabel as nib
import numpy as np
import torch

from .data import sha256
from .masks import draw_holdout, InsufficientContext
from .prepare_dryad import member, read, validity
from .profiles import describe, distribution, draw_profile, fit_profile


MOVIES = ('Aeronaut', 'Catepillar', 'Child_Play', 'Meerkats', 'Mouseforsale')


def family_split(families, seed=42):
    """Deterministic 70/15/remainder rank split, based only on family identifiers."""
    ordered = sorted(set(families), key=lambda f: hashlib.sha256(f'{seed}:{f}'.encode()).hexdigest())
    if len(ordered) < 7:
        raise ValueError('at least seven families required for this cohort split')
    ntrain, nval = int(.7*len(ordered)), max(1, int(.15*len(ordered)))
    return {f: ('train' if i < ntrain else 'val' if i < ntrain+nval else 'test') for i, f in enumerate(ordered)}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parse_mask(data, frames):
    matrix = np.loadtxt(io.BytesIO(data), ndmin=2)
    if matrix.shape[0] != frames or not np.isin(matrix, [0, 1]).all():
        raise ValueError('binary mask matching the stored NIfTI time axis required')
    return matrix.any(1)


def summarize(rows):
    motion = [describe(r['motion'], r['tr_seconds']) for r in rows]
    known = [r for r in rows if r['gaze_known']]
    return dict(records=len(rows), families=len({r['family'] for r in rows}),
                eligible_records=sum(r['eligible'] for r in rows),
                motion=distribution(motion),
                recoverable_motion_known_gaze=distribution(describe(r['recoverable'], r['tr_seconds']) for r in known),
                all_exclusions_known_gaze=distribution(describe(~np.array(r['valid']), r['tr_seconds']) for r in known))


def compare_samplers(rows, profile, draws, seed):
    if draws < 1:
        raise ValueError('positive draws per record required')
    # Reproduce the old sampler on the full stored axis, including false padding.
    # Report only the actually acquired prefix for both methods.
    empirical = [torch.tensor(r['recoverable'] + [False]*(r['stored_frames']-r['acquired_frames'])) for r in rows]
    result = {}
    mask_args = dict(gap_lengths=(1, 2, 4, 8, 16), max_gaps=3, allow_edges=True, min_observations=2)
    for method in ('legacy_random', 'legacy_mixture', 'fitted'):
        realized, proposals, fractions, attempts, failures = [], [], [], [], []
        for i, row in enumerate(rows):
            n = row['acquired_frames']
            valid = torch.tensor(row['valid'] + [False]*(row['stored_frames']-n))[None]
            times = torch.arange(valid.shape[-1]).float()*row['tr_seconds']
            acquired = torch.arange(valid.shape[-1]) < n
            for repeat in range(draws):
                sample_seed = seed + i*100000 + repeat
                try:
                    if method == 'fitted':
                        observed, target, info = draw_profile(valid, times, profile, seed=sample_seed,
                                                              exclude_family=row['family'], acquired=acquired)
                        proposals.append(info['proposal'])
                        attempts.append(info['attempts'])
                    else:
                        observed, target = draw_holdout(valid, seed=sample_seed, **mask_args,
                                                        empirical=empirical if method == 'legacy_mixture' else None)
                    realized.append(describe(target.any(0)[:n].numpy(), row['tr_seconds']))
                    fractions.append(float(target.sum()/valid.sum()))
                except InsufficientContext:
                    failures.append(dict(id=row['id'], seed=sample_seed))
        result[method] = dict(requested=len(rows)*draws, successful=len(realized), failures=failures,
                              realized_targets=distribution(realized),
                              mean_target_fraction_of_clean=float(np.mean(fractions)) if fractions else None)
        if proposals:
            result[method].update(proposals=distribution(proposals), mean_attempts=float(np.mean(attempts)),
                                  max_attempts=max(attempts))
    return result


def audit(root, output, *, eye_shift_tr=0, guard_tr=0, seed=42, draws=100):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tables, sources = {}, []
    for movie in MOVIES:
        with zipfile.ZipFile(root / (movie + '.zip')) as archive:
            name = movie + '/infant_participants.csv'
            raw = read(archive, name)
            tables[movie] = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
            sources.append(dict(archive=movie+'.zip', member=name, sha256=hashlib.sha256(raw).hexdigest()))
    families = []
    for table in tables.values():
        for row in table:
            if not re.fullmatch(r's\d+_\d+_\d+', row['ppt']):
                raise ValueError('unknown participant naming convention: '+row['ppt'])
            families.append(row['ppt'].split('_')[0])
    split = family_split(families, seed)
    # Freeze before inspecting masks. README explicitly defines sXXXX as the family ID.
    dump(output / 'cohort_split.json', dict(seed=seed, method='sha256 rank of seed:family; floor 70%/15%/remainder',
                                          family_definition='Dataset README: sXXXX family, following fields sibling/session',
                                          families=split, readme_sha256=sha256(root / 'README.md')))
    records = []
    for movie, table in tables.items():
        with zipfile.ZipFile(root / (movie + '.zip')) as archive:
            for row in table:
                ppt = row['ppt']
                native = f'{movie}/preprocessed_native/linear_alignment/{ppt}_Z.nii.gz'
                with member(archive, native) as raw, gzip.GzipFile(fileobj=raw) as stream:
                    block = stream.read(348)
                header = nib.Nifti1Header(binaryblock=block)
                frames = int(header.get_data_shape()[-1])
                unit = header.get_xyzt_units()[1]
                tr = float(header.get_zooms()[3])*{'sec': 1, 'msec': .001, 'usec': .000001}[unit]
                masks = {}
                for kind in ('motion', 'eye'):
                    name = f'{movie}/{kind}_confounds/{ppt}.txt'
                    content = read(archive, name)
                    masks[kind] = parse_mask(content, frames)
                    sources.append(dict(archive=movie+'.zip', member=name,
                                        sha256=hashlib.sha256(content).hexdigest()))
                n = int(row['num_TR'])
                try:
                    gaze_known = bool(np.isfinite(float(row['prop_TR_eye'])))
                except (KeyError, ValueError):
                    gaze_known = False
                v, recover, shifted, padding = validity(masks['motion'], masks['eye'], n,
                                                       eye_shift_tr=eye_shift_tr, guard_tr=guard_tr)
                reasons = []
                if not gaze_known:
                    reasons.append('missing_gaze_metadata')
                    v[:] = recover[:] = False
                if movie == 'Child_Play':
                    reasons.append('stitched_clips_with_removed_rest_no_segment_map')
                if gaze_known and int(v.sum()) < 2:
                    reasons.append('fewer_than_two_clean_frames')
                family = ppt.split('_')[0]
                records.append(dict(id=movie+':'+ppt, movie=movie, ppt=ppt, family=family, split=split[family],
                                    tr_seconds=tr, stored_frames=frames, acquired_frames=n,
                                    eligible=not reasons, exclusion_reasons=reasons, gaze_known=bool(gaze_known),
                                    motion=masks['motion'][:n].tolist(), eye=shifted[:n].tolist(),
                                    valid=v[:n].tolist(), recoverable=recover[:n].tolist(),
                                    native_header_sha256=hashlib.sha256(block).hexdigest(),
                                    motion_csv_fraction_error=abs(float((~masks['motion'][:n]).mean())-float(row['prop_TR_motion']))))
        print(f'{movie}: {len(table)} infant records', flush=True)
    dump(output / 'records.json', records)
    training = [r for r in records if r['eligible'] and r['split'] == 'train']
    profile = fit_profile(training)
    dump(output / 'train_mask_profile.json', profile)
    comparison = compare_samplers(training, profile, draws, seed)
    result = dict(protocol=dict(seed=seed, eye_shift_tr=eye_shift_tr, guard_tr=guard_tr,
                               acquired_interval='first CSV num_TR frames; padding excluded from all gap statistics',
                               child_play='stored-axis descriptions only; excluded from continuous-time fitting',
                               comparison_scope='eligible training families only; all-cohort descriptions never fit the sampler',
                               comparison_axis='sample on full stored axis with false padding; measure gaps within acquired prefix',
                               draws_per_training_record=draws),
                  by_movie={m: summarize([r for r in records if r['movie'] == m]) for m in MOVIES},
                  eligible_by_split={s: summarize([r for r in records if r['eligible'] and r['split'] == s])
                                     for s in ('train', 'val', 'test')},
                  excluded=[{k: r[k] for k in ('id', 'split', 'exclusion_reasons')} for r in records if not r['eligible']],
                  sources=sources, profile_sha256=profile['sha256'], sampler_comparison=comparison)
    dump(output / 'audit.json', result)
    print(json.dumps({'eligible_by_split': {s: {'records': d['records'], 'families': d['families']}
                                           for s, d in result['eligible_by_split'].items()},
                      'training_recoverable': profile['summary'],
                      'comparison': {k: {'successful': v['successful'], 'requested': v['requested'],
                                          'target_gap_seconds': v['realized_targets']['gap_seconds_quantiles'],
                                          'target_fraction_clean': v['mean_target_fraction_of_clean']}
                                     for k, v in comparison.items()}}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--eye-shift-tr', type=int, required=True)
    parser.add_argument('--guard-tr', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--draws', type=int, default=100)
    args = parser.parse_args()
    audit(args.data_root, args.out, eye_shift_tr=args.eye_shift_tr, guard_tr=args.guard_tr,
          seed=args.seed, draws=args.draws)


if __name__ == '__main__':
    main()
