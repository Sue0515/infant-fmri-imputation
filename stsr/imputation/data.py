"""Strict prepared-record contract; split families before drawing any masks."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('manifest must be a nonempty JSON list')
    families, ids = {}, set()
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in ('id', 'family', 'split', 'path')):
            raise ValueError('each record needs id, family, split, and path strings')
        if row['split'] not in ('train', 'val', 'test') or row['id'] in ids:
            raise ValueError('unique ids and train/val/test splits required')
        if row['id'].split(':')[0] == 'Child_Play' or row.get('movie') == 'Child_Play':
            raise ValueError('Child_Play has removed rest intervals; prepare separately mapped continuous segments first')
        if row['family'] in families and families[row['family']] != row['split']:
            raise ValueError('family leakage across splits: ' + row['family'])
        families[row['family']] = row['split']
        ids.add(row['id'])
        row['path'] = str((path.parent / row['path']).resolve())
        if not Path(row['path']).is_file():
            raise ValueError('missing prepared record: ' + row['path'])
    # Same file must not be assigned separate identities/splits.
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('duplicate record paths in manifest')
    return rows


@dataclass
class Record:
    values: torch.Tensor
    valid: torch.Tensor
    recoverable: torch.Tensor
    coords: torch.Tensor
    times: torch.Tensor
    acquired: torch.Tensor = None

    def to(self, device):
        return Record(*(x.to(device) if x is not None else None for x in
                        (self.values, self.valid, self.recoverable, self.coords, self.times, self.acquired)))

    def analysis_view(self):
        """Exclude only the protocol tail; never remove internal missing frames.

        Retain the full record for export. A false tail is outside the analysis
        domain, not a missing interval for attention/FFT to extrapolate through.
        """
        from .profiles import acquisition_length
        domain = None if self.acquired is None else self.acquired.detach().cpu().numpy()
        n = acquisition_length(domain, len(self.times))
        if self.valid[:, n:].any() or self.recoverable[:, n:].any():
            raise ValueError('excluded tail cannot contain observations or recovery targets')
        return Record(self.values[:, :n], self.valid[:, :n], self.recoverable[:, :n],
                      self.coords, self.times[:n], None if self.acquired is None else self.acquired[:n])


def load_record(path, *, max_vertices=256, max_frames=256):
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data['values'], dtype=np.float32)
        coords = np.asarray(data['coords_mm'], dtype=np.float32)
        times = np.asarray(data['times_s'], dtype=np.float32)
        valid, recoverable = data['valid'], data['recoverable']
        if x.ndim != 2 or x.shape[0] < 1 or not 2 <= x.shape[1] <= max_frames or x.shape[0] > max_vertices:
            raise ValueError('record exceeds pilot geometry limits or is not V x T')
        if valid.dtype != np.bool_ or recoverable.dtype != np.bool_ or valid.shape != x.shape or recoverable.shape != x.shape:
            raise ValueError('valid and recoverable must be boolean V x T arrays')
        if (valid & recoverable).any() or not np.isfinite(x[valid]).all():
            raise ValueError('valid/recoverable must be disjoint and valid values finite')
        if not valid.any():
            raise ValueError('record has no valid observations')
        if coords.shape != (len(x), 3) or times.shape != (x.shape[1],) or not np.isfinite(coords).all() or not np.isfinite(times).all():
            raise ValueError('invalid geometry')
        dt = np.diff(times)
        if (dt <= 0).any() or not np.allclose(dt, dt[0], rtol=1e-4, atol=1e-6):
            raise ValueError('retain a regular acquisition grid; do not compress time')
        padding = data['padding'] if 'padding' in data else np.zeros(x.shape[-1], dtype=bool)
        if padding.dtype != np.bool_ or padding.shape != times.shape:
            raise ValueError('padding must be a boolean time vector')
        from .profiles import acquisition_length
        acquisition_length(~padding, x.shape[-1])
        if valid[:, padding].any() or recoverable[:, padding].any():
            raise ValueError('padding cannot be observed or recoverable')
        # Destroy stored motion interpolation and excluded values before modeling.
        x = np.where(valid, x, np.nan)
        return Record(torch.from_numpy(x.copy()), torch.from_numpy(valid.copy()),
                      torch.from_numpy(recoverable.copy()), torch.from_numpy(coords.copy()),
                      torch.from_numpy(times.copy()), torch.from_numpy((~padding).copy()))


def check_evaluation_split(rows, training_rows, split):
    selected = [r for r in rows if r['split'] == split]
    if split not in ('val', 'test') or not selected:
        raise ValueError('nonempty val or test split required')
    # Keep the checkpoint's reserved test families out of model selection too.
    forbidden = [r for r in training_rows if r['split'] != split]
    families = {r['family'] for r in forbidden}
    ids = {r['id'] for r in forbidden}
    paths = {r['path'] for r in forbidden}
    if any(r['family'] in families or r['id'] in ids or r['path'] in paths for r in selected):
        raise ValueError('evaluation overlaps training/model-selection identities')
    return selected
