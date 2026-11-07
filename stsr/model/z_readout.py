"""
Latent-state readout: feature lattice (B, K, L, d_model) -> z on the fine
time grid (dt = TR/up, up = ceil(TR/dt_target)), preceded by n_pre pre-roll
samples so the causal kernel has history to render frame 0 from.

The sub-frame position enters as a physical offset in seconds:
z[l*up + j] = out(act(W_f feat[l] + W_p ff(j * dt))). No parameter shape
contains up, K, L or TR, so the same weights run at any TR — "0.25 s past
the frame" means the same thing on every grid, "phase 1/6" would not.
Native frame l lands exactly on fine index n_pre + l*up (integer indexing,
no interpolation). The offset term broadcasts over up, so the d_model-sized
lattice is never inflated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..utils import upsample_factor  # noqa: F401  (re-exported for callers)
from .coord_encoding import fourier_features


class ZReadoutHead(nn.Module):
    """(B, K, L, d_model) feature lattice -> (B, K, L*up) latent state.

    init_gain scales the output layer at construction. The hemodynamic kernel
    has a DC gain of sum_j k[j] ~ 10 at dt = 0.25 s, so a unit-variance BOLD
    target corresponds to a z roughly 10x smaller; starting the head near that
    scale removes an otherwise pointless few hundred steps of learning a
    constant. Pass 1.0 to disable.
    """

    def __init__(self, d_model: int, hidden: int = 64, n_freqs: int = 6,
                 min_wavelength_s: float = 0.1, max_wavelength_s: float = 8.0,
                 init_gain: float = 0.1):
        super().__init__()
        self.hidden = int(hidden)
        self.n_freqs = int(n_freqs)
        self.min_wavelength_s = float(min_wavelength_s)
        self.max_wavelength_s = float(max_wavelength_s)
        self.proj_feat = nn.Linear(d_model, hidden)
        self.proj_offset = nn.Linear(2 * n_freqs, hidden)
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.out.weight.mul_(float(init_gain))
            self.out.bias.mul_(0.0)

    def _offset_features(self, up: int, dt: float, dtype, device):
        """(up, hidden) — the sub-frame offset term, in seconds."""
        off = (torch.arange(up, dtype=dtype, device=device)
               * float(dt)).unsqueeze(-1)                      # (up, 1)
        ff = fourier_features(off, self.n_freqs, self.min_wavelength_s,
                              self.max_wavelength_s)           # (up, 2F)
        return self.proj_offset(ff)

    def _combine(self, hf: torch.Tensor, ho: torch.Tensor) -> torch.Tensor:
        """(B,K,L,h) + (up,h) -> (B,K,L,up) — split out so it can be
        checkpointed (this is the only tensor that grows by `up`)."""
        h = F.gelu(hf.unsqueeze(3) + ho.view(1, 1, 1, *ho.shape))
        return self.out(h).squeeze(-1)

    def forward(self, feat: torch.Tensor, up: int, dt: float,
                n_pre: int = 0, grad_checkpointing: bool = False
                ) -> torch.Tensor:
        """feat (B, K, L, d_model) -> z (B, K, n_pre + L*up) on the dt grid.

        Native frame l sits at fine index n_pre + l*up; within a frame, index
        offset j carries j*dt seconds.

        WHY n_pre (the pre-roll). BOLD at time t is the HRF-weighted integral
        of the preceding ~32 s of neural activity, so a rendered frame near
        the window start depends on z from BEFORE the window. Without a
        pre-roll those frames are not merely hard, they are UNRENDERABLE: the
        causal kernel is zero at lag 0, so the first rendered sample is
        identically 0 whatever z is, the row of the sampling operator is a
        zero row, and the data-consistency constraint set for measured data
        is EMPTY. CG then chases an infeasible residual and z diverges
        (observed: Gram singular values down to 7e-31, |correction| ~ 1e9).
        Carrying n_pre = n_taps - 1 extra latent samples restores full row
        rank: every observed frame sees a complete history. The pre-roll is a
        nuisance variable — never rendered into a scored frame, never
        supervised — so it is read out from the first frame's features; the
        projection is what actually fits it to the data.
        """
        if feat.ndim != 4:
            raise ValueError(
                f"feat must be (B, K, L, C), got {tuple(feat.shape)}")
        up, n_pre = int(up), int(n_pre)
        if up < 1:
            raise ValueError(f"up must be >= 1, got {up}")
        if n_pre < 0:
            raise ValueError(f"n_pre must be >= 0, got {n_pre}")
        B, K, L, _ = feat.shape
        n_pre_frames = -(-n_pre // up)                  # ceil
        if n_pre_frames:
            feat = torch.cat(
                [feat[:, :, :1].expand(-1, -1, n_pre_frames, -1), feat],
                dim=2)
        hf = self.proj_feat(feat)                              # (B,K,L',h)
        ho = self._offset_features(up, dt, feat.dtype, feat.device)
        if grad_checkpointing and self.training:
            z = torch.utils.checkpoint.checkpoint(
                self._combine, hf, ho, use_reentrant=False)
        else:
            z = self._combine(hf, ho)
        z = z.reshape(B, K, (L + n_pre_frames) * up)
        # crop so exactly n_pre samples precede native frame 0
        return z[..., n_pre_frames * up - n_pre:]
