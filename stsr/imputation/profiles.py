"""Fit gap templates on training families; sample without BOLD values."""
import hashlib
import json
from collections import Counter

import numpy as np
import torch

from .masks import InsufficientContext


def intervals(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError('one-dimensional temporal mask required')
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def describe(mask, tr):
    mask = np.asarray(mask, dtype=bool)
    if not len(mask) or not np.isfinite(tr) or tr <= 0:
        raise ValueError('nonempty mask and positive TR required')
    gaps = [dict(start=a, stop=b, frames=b-a, seconds=float((b-a)*tr),
                 leading=a == 0, trailing=b == len(mask)) for a, b in intervals(mask)]
    return dict(frames=len(mask), missing_frames=int(mask.sum()), fraction=float(mask.mean()),
                gap_count=len(gaps), gaps=gaps)


def distribution(descriptions):
    """Run-weighted fractions, gap-weighted lengths; explicitly distinct denominators."""
    descriptions = list(descriptions)
    gaps = [g for d in descriptions for g in d['gaps']]
    def quantiles(values):
        return {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, .9, .95, 1)} if values else {}
    return dict(runs=len(descriptions), gaps=len(gaps),
                zero_gap_runs=sum(not d['gap_count'] for d in descriptions),
                frames=sum(d['frames'] for d in descriptions),
                missing_frames=sum(d['missing_frames'] for d in descriptions),
                run_missing_fraction_quantiles=quantiles([d['fraction'] for d in descriptions]),
                run_gap_count_quantiles=quantiles([d['gap_count'] for d in descriptions]),
                gap_seconds_quantiles=quantiles([g['seconds'] for g in gaps]),
                gap_frames_histogram=dict(sorted(Counter(g['frames'] for g in gaps).items())),
                boundary_gaps=sum(g['leading'] or g['trailing'] for g in gaps),
                boundary_gap_fraction=sum(g['leading'] or g['trailing'] for g in gaps)/len(gaps) if gaps else None,
                leading_gaps=sum(g['leading'] for g in gaps), trailing_gaps=sum(g['trailing'] for g in gaps),
                runs_with_boundary_gap=sum(any(g['leading'] or g['trailing'] for g in d['gaps']) for d in descriptions))


def fit_profile(records):
    """Records contain mask metadata only; held-out/ineligible inputs are rejected."""
    templates, seen = [], set()
    for row in sorted(records, key=lambda r: r['id']):
        if row['split'] != 'train' or not row.get('eligible', True):
            raise ValueError('profile fitting accepts eligible training records only')
        if row['id'] in seen:
            raise ValueError('duplicate profile record')
        seen.add(row['id'])
        d = describe(row['recoverable'], row['tr_seconds'])
        templates.append(dict(id=row['id'], family=row['family'], **d))
    if not templates or not any(t['gaps'] for t in templates):
        raise ValueError('training profile needs nonempty motion gap templates')
    profile = dict(version=1, strategy='family_balanced_gap_templates', templates=templates,
                   families=sorted({t['family'] for t in templates}), summary=distribution(templates),
                   semantics='Recoverable motion inside acquired frames; zero-gap donors cannot supply a target. '
                             'Uniform donor family, then nonempty donor run. Preserve gap durations/count and '
                             'leading/trailing status; randomize interior positions without overlap. '
                             'Reject impossible layouts or insufficient clean targets; report attempts.')
    profile['sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    return profile


def acquisition_length(acquired, frames):
    acquired = np.ones(frames, dtype=bool) if acquired is None else np.asarray(acquired, dtype=bool)
    if acquired.shape != (frames,):
        raise ValueError('acquired must match the time axis')
    n = int(acquired.sum())
    if n < 2 or not np.array_equal(acquired, np.arange(frames) < n):
        raise ValueError('acquired must be a contiguous prefix; segment stitched acquisitions first')
    return n


def draw_profile(valid, times, profile, *, seed, acquired=None, exclude_family=None,
                 min_observations=2, max_attempts=128):
    if valid.ndim != 2 or valid.dtype != torch.bool or len(times) != valid.shape[-1]:
        raise ValueError('boolean V x T validity and matching times required')
    if min_observations < 1 or max_attempts < 1 or len(times) < 2:
        raise ValueError('positive support/attempt limits and at least two times required')
    dt = times.diff()
    if not torch.isfinite(times).all() or (dt <= 0).any() or not torch.allclose(dt, dt[0].expand_as(dt), atol=1e-6, rtol=1e-4):
        raise ValueError('uniform physical time axis required')
    domain = None if acquired is None else torch.as_tensor(acquired).cpu().numpy()
    n = acquisition_length(domain, valid.shape[-1])
    if valid[:, n:].any():
        raise ValueError('valid samples cannot lie in padding')
    tr = float(dt[0])
    donors = {}
    for template in profile['templates']:
        if template['gaps'] and template['family'] != exclude_family:
            donors.setdefault(template['family'], []).append(template)
    if not donors:
        raise InsufficientContext('no nonempty training-family donor remains')
    families = sorted(donors)
    generator = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        family = families[int(generator.integers(len(families)))]
        donor = donors[family][int(generator.integers(len(donors[family])))]
        layout = np.zeros(n, dtype=bool)
        # Boundaries first, then longest interior gaps, with at least one separating frame.
        gaps = sorted(donor['gaps'], key=lambda g: (not (g['leading'] or g['trailing']), -g['seconds']))
        placed = True
        for gap in gaps:
            length = max(1, int(np.floor(gap['seconds']/tr + .5)))
            if gap['leading'] and gap['trailing']:
                starts = [0] if length == n else []
            elif gap['leading']:
                starts = [0] if length < n else []
            elif gap['trailing']:
                starts = [n-length] if length < n else []
            else:
                starts = list(range(1, n-length))
            starts = [a for a in starts if not layout[max(0, a-1):min(n, a+length+1)].any()]
            if not starts:
                placed = False
                break
            start = starts[int(generator.integers(len(starts)))]
            layout[start:start+length] = True
        if not placed:
            continue
        hidden = torch.zeros(valid.shape[-1], dtype=torch.bool, device=valid.device)
        hidden[:n] = torch.as_tensor(layout, device=valid.device)
        condition, target = valid & ~hidden[None], valid & hidden[None]
        target &= condition.sum(-1, keepdim=True) >= min_observations
        if target.any():
            return condition, target, dict(donor_id=donor['id'], donor_family=family, attempts=attempt,
                                            proposal=describe(layout, tr),
                                            realized=describe(target.any(0)[:n].cpu().numpy(), tr),
                                            target_fraction_of_valid=float(target.sum()/valid.sum()))
    raise InsufficientContext('no fitted gap layout with sufficient clean targets; do not silently shorten gaps')


def draw_stress(valid, times, *, acquired=None, seconds, location, seed, min_observations=2):
    """One controlled long/edge gap. Unknown values never become ground truth."""
    domain = None if acquired is None else torch.as_tensor(acquired).cpu().numpy()
    n = acquisition_length(domain, valid.shape[-1])
    tr = float(times.diff()[0])
    if seconds <= 0 or location not in ('interior', 'leading', 'trailing'):
        raise ValueError('positive duration and interior/leading/trailing location required')
    length = max(1, int(np.floor(seconds/tr + .5)))
    if length >= n or (location == 'interior' and length > n-2):
        raise InsufficientContext('stress gap cannot fit with observed context')
    generator = np.random.default_rng(seed)
    starts = [0] if location == 'leading' else [n-length] if location == 'trailing' else generator.permutation(np.arange(1, n-length)).tolist()
    for start in starts:
        hidden = torch.zeros(valid.shape[-1], dtype=torch.bool, device=valid.device)
        hidden[start:start+length] = True
        condition, target = valid & ~hidden[None], valid & hidden[None]
        target &= condition.sum(-1, keepdim=True) >= min_observations
        if target.any():
            return condition, target, dict(requested_seconds=seconds, location=location,
                                            proposal=describe(hidden[:n].cpu().numpy(), tr),
                                            scored_fraction_of_proposal=float(target.sum()/(length*len(valid))))
    raise InsufficientContext('stress gap has no known targets with sufficient context')
