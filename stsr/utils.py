"""
Shared helpers, one implementation each so the trainer, scripts and tests
cannot drift apart on the numbers they compare: device/AMP resolution
(autocast covers trunk + heads; physics and the CG solve stay fp32),
Pearson correlation (full rows and masked per-row), per-row masked linear
interpolation (gap-filling before spectral transforms), the shared
linear-interpolation baseline, the fine-grid upsample factor, and the
per-subject metric accumulator both objectives report through.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

# A co-tenant GPU with less than this free is not worth fighting over.
MIN_FREE_GPU_MB = 20 * 1024
AMP_MODES = ('auto', 'off', 'bf16', 'fp16', 'bfloat16', 'float16')


@dataclass(frozen=True)
class AmpConfig:
    """Resolved mixed-precision setting, carried as one value.

    Autocast covers the trunk and the readout heads only. The measurement
    model and the data-consistency solve stay fp32 — see
    physics.measurement for why a CG solve cannot live in bf16.
    """
    enabled: bool
    dtype: torch.dtype
    device: str

    @classmethod
    def resolve(cls, requested: str, device: str) -> 'AmpConfig':
        """'auto' = bf16 on CUDA, off on CPU.

        bf16 rather than fp16: same exponent range as fp32, so no loss
        scaling and no overflow bookkeeping. fp16 stays available and brings
        the gradient scaler with it.
        """
        if requested in ('bfloat16',):
            requested = 'bf16'
        elif requested in ('float16',):
            requested = 'fp16'
        if requested not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, got {requested!r}")
        # 'cuda:1' is CUDA. Comparing the raw string would silently disable
        # bf16 on every non-zero DDP rank, which is slower, uses more memory
        # and produces different numerics from every recorded baseline; and
        # torch.autocast rejects an indexed device_type outright.
        device = torch.device(device).type
        if requested == 'off' or device != 'cuda':
            return cls(False, torch.float32, device)
        dtype = torch.float16 if requested == 'fp16' else torch.bfloat16
        return cls(True, dtype, device)

    def context(self):
        """Autocast region for trunk/head compute."""
        return torch.autocast(device_type=self.device, dtype=self.dtype,
                              enabled=self.enabled)

    def scaler(self):
        """Gradient scaler; a no-op unless fp16 is in use."""
        return torch.amp.GradScaler(
            self.device, enabled=self.enabled and self.dtype is torch.float16)

    @property
    def label(self) -> str:
        return str(self.dtype).split('.')[-1] if self.enabled else 'off'


def resolve_device(requested: str) -> str:
    """'auto': cuda only if available AND nvidia-smi shows enough free memory.

    The free-memory check matters on shared login nodes, where torch reports
    CUDA as available while another tenant holds most of the card.
    """
    if requested in ('cpu', 'cuda'):
        return requested
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True).stdout
        free_mb = max(int(line) for line in out.strip().splitlines())
    except Exception:
        return 'cpu'
    return 'cuda' if free_mb >= MIN_FREE_GPU_MB else 'cpu'


def upsample_factor(tr_s: float, dt_target_s: float) -> int:
    """Smallest integer `up` with TR/up <= dt_target.

    Lives here because both ends of the latent grid need it and neither owns
    it: the readout head sizes its output by it, the measurement model sizes
    its operator by it, and the two MUST agree or native frames stop landing
    on integer fine indices.
    """
    if tr_s <= 0 or dt_target_s <= 0:
        raise ValueError(f"tr_s={tr_s}, dt_target_s={dt_target_s} must be > 0")
    return max(1, math.ceil(tr_s / dt_target_s - 1e-9))


def pearson_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along the last axis -> (...). Rows with no
    variance give 0 rather than NaN, so a constant vertex cannot poison a
    batch mean."""
    ac = a - a.mean(-1, keepdim=True)
    bc = b - b.mean(-1, keepdim=True)
    den = ac.norm(dim=-1) * bc.norm(dim=-1)
    return torch.where(den > 1e-12, (ac * bc).sum(-1) / den.clamp_min(1e-12),
                       torch.zeros_like(den))


def pearson_rows_masked(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor,
                        min_n: int = 4) -> torch.Tensor:
    """Pearson correlation along the last axis over the entries m selects
    per row -> (...). Rows with fewer than min_n selected entries, or with
    no variance among them, return NaN so the caller can drop them (a mean
    over an irregular per-row entry count must not count degenerate rows
    as zeros the way pearson_rows does for full rows)."""
    m = m.to(a.dtype)
    n = m.sum(-1)
    am = (a * m).sum(-1, keepdim=True) / n.clamp_min(1).unsqueeze(-1)
    bm = (b * m).sum(-1, keepdim=True) / n.clamp_min(1).unsqueeze(-1)
    ac = (a - am) * m
    bc = (b - bm) * m
    den = ac.norm(dim=-1) * bc.norm(dim=-1)
    r = (ac * bc).sum(-1) / den.clamp_min(1e-12)
    return torch.where((n >= min_n) & (den > 1e-12), r,
                       torch.full_like(r, float('nan')))


def masked_row_interp(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row linear interpolation of x (..., L) across its unobserved
    entries (mask 0), reading observed entries only; edges clamp to the
    nearest observed value and all-unobserved rows stay zero.

    The per-row generalization of `temporal_interp` for masks that differ
    between rows (checkerboard, tube). It exists for gap-filling a signal
    before a spectral transform: zero-filled gaps put the MASK's spectrum
    into the signal's high band (a checkerboard is a broadband comb), which
    is exactly what the HF/LF routing decomposition must not inherit.
    """
    L = x.shape[-1]
    mb = mask.to(torch.bool).expand_as(x)
    ar = torch.arange(L, device=x.device).expand_as(x)
    none = torch.full_like(ar, -1)
    prev = torch.cummax(torch.where(mb, ar, none), dim=-1).values
    nxt = (L - 1) - torch.cummax(
        torch.where(mb.flip(-1), ar, none), dim=-1).values.flip(-1)
    has_p, has_n = prev >= 0, nxt <= L - 1
    p, n = prev.clamp(min=0), nxt.clamp(max=L - 1)
    xp = x.gather(-1, p)
    xn = x.gather(-1, n)
    w = ((ar - p).to(x.dtype) / (n - p).clamp(min=1).to(x.dtype))
    interp = xp * (1.0 - w) + xn * w
    return torch.where(has_p & has_n, interp,
                       torch.where(has_p, xp,
                                   torch.where(has_n, xn,
                                               torch.zeros_like(x))))


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two flattened tensors; NaN if either is
    constant (the caller decides whether to drop or keep that)."""
    return float(pearson_rows(a.reshape(-1), b.reshape(-1))) \
        if a.std() > 1e-8 and b.std() > 1e-8 else float('nan')


def temporal_interp(x: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
    """Per-row linear interpolation of x (..., L) from the observed frame
    indices obs_t (sorted); edges clamp.

    The classical baseline every recovery number is judged against. Defined
    only for a temporally broadcast mask — when vertices are observed at
    different frames (checkerboard, tube) there is no shared grid to
    interpolate on, and callers must not substitute a per-vertex variant
    without saying so.
    """
    L = x.shape[-1]
    t = torch.arange(L, device=x.device)
    hi = torch.searchsorted(obs_t, t).clamp(1, obs_t.numel() - 1)
    lo = hi - 1
    t0, t1 = obs_t[lo], obs_t[hi]
    w = ((t - t0).to(x.dtype) / (t1 - t0).to(x.dtype)).clamp(0.0, 1.0)
    return x[..., obs_t[lo]] * (1.0 - w) + x[..., obs_t[hi]] * w


def apply_observed_mask(cond, target, batch, device, enabled: bool = True):
    """Narrow the sampling masks to the entries an item actually observed.

    The mask controller draws ONE (V, T) pattern per step and shares it
    across the batch, which is what makes the checkerboard geometry a
    property of the run rather than of a scan. But validity is per scan:
    a voxel outside this subject's field of view is stored as exactly 0.0,
    and a frame with 0.6 mm of head motion is still a number. Without this
    intersection the objective scores those zeros as if they were data, the
    data-consistency projection forces them as if they were measurements,
    and every reported metric is inflated by whatever fraction of the
    window they occupy (measured: 1.5% of served entries, 2.2% of held-out
    entries, up to 15% on the worst run).

    Returns (cond, target) unchanged when the batch carries no mask.
    """
    om = batch.get('observed_mask') if isinstance(batch, dict) else None
    if om is None or not enabled:
        return cond, target
    om = om.to(device=device, dtype=cond.dtype)
    return cond * om, target * om


class SubjectMetric:
    """Per-subject accumulator for a validation metric that is pooled as a
    plain mean over appended values.

    Exists so the pooled number and the per-subject numbers cannot drift: the
    caller feeds both inside the SAME guard, and `means()` reproduces the
    pooled mean exactly as the count-weighted mean over subjects. Values whose
    subject is unknown are dropped rather than bucketed under a placeholder,
    so a loader that ships no subject leaves the accumulator empty and the
    caller omits the per-subject keys instead of reporting a fake group.

    `add` and `extend` are NOT interchangeable: they set the weight a window
    carries. add() gives every window equal weight; extend() gives a window
    weight proportional to how many values it contributed. Each caller must
    mirror whatever its own pooled list does, or the subject means answer a
    different question than the pooled number they sit beside.
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add(self, subject: Optional[str], value: float) -> None:
        """One value for one subject; a None subject is ignored."""
        if subject is None:
            return
        self._sums[subject] = self._sums.get(subject, 0.0) + float(value)
        self._counts[subject] = self._counts.get(subject, 0) + 1

    def extend(self, subject: Optional[str], values: Iterable[float]) -> None:
        """Many values for one subject, each weighing the same as an add()."""
        if subject is None:
            return
        vals = [float(v) for v in values]
        if not vals:
            return
        self._sums[subject] = self._sums.get(subject, 0.0) + sum(vals)
        self._counts[subject] = self._counts.get(subject, 0) + len(vals)

    def means(self) -> Dict[str, float]:
        """Mean per subject, insertion-ordered by first appearance."""
        return {s: self._sums[s] / self._counts[s] for s in self._counts}

    def counts(self) -> Dict[str, int]:
        """How many values each subject contributed — the weights that turn
        `means()` back into the pooled mean."""
        return dict(self._counts)

    def __bool__(self) -> bool:
        return bool(self._counts)

    def __len__(self) -> int:
        return len(self._counts)


def batch_subjects(batch, n: int) -> List[Optional[str]]:
    """The 'subject' entry of a batch as a length-n list, or [None]*n when the
    batch carries none.

    Per-subject metrics are opt-in on the dataset: the synthetic corpus and
    the older loaders ship no 'subject' key, and a missing subject must leave
    the metric absent rather than invent a bucket. The default collate turns a
    per-item str into a list[str] of the batch length; a loader that hands
    back a bare str (one un-collated item) is broadcast, and a short or long
    list is padded/truncated so an unexpected collation can never shift a
    value onto the wrong subject.
    """
    subs = batch.get('subject') if isinstance(batch, dict) else None
    if subs is None:
        return [None] * n
    if isinstance(subs, str):
        return [subs] * n
    out: List[Optional[str]] = [str(s) for s in subs][:n]
    return out + [None] * (n - len(out))


def note_missing_subjects(metrics: Dict[str, object],
                          expected: Iterable[str]) -> Dict[str, object]:
    """Record held-out subjects that produced no per-subject value.

    A subject whose every window fails the variance guard — all its voxels
    censored by `observed_mask`, or a flat prediction — simply never reaches
    the accumulator, so it disappears from the by-subject dicts with no
    error. Silently comparing 10 subjects in one condition against 11 in
    another is exactly the failure a paired test exists to rule out, so the
    absence is written down where the reader of the metrics will see it.

    `validate()` cannot do this itself: it sees batches, not the split's
    roster. The caller that owns the dataset adds it. Mutates and returns
    `metrics`; a no-op when the run reports no per-subject values at all.
    """
    by_subject = metrics.get('recon_r_by_subject')
    if not isinstance(by_subject, dict):
        return metrics
    missing = [s for s in dict.fromkeys(expected) if s not in by_subject]
    metrics['subjects_missing'] = missing
    return metrics
