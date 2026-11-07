"""
Continuous-coordinate readout: feature lattice + arbitrary query point
(xyz mm, tau s) -> field value. The piece that makes the lattice a hybrid
neural field — the lattice alone can only be indexed; this head can be
asked about any coordinate.

Per query: k nearest lattice vertices -> their features linearly
interpolated in time between the two bracketing frames -> per-neighbour
token [feature, fourier(xyz offset), fourier(tau offset), theta embedding]
-> MLP -> learned softmax pooling -> value. All quantities are physical
units; no parameter shape depends on V or T (defined on any grid —
accuracy off the training grid is a separate, measured question).
"""
from typing import Tuple

import torch
import torch.nn as nn

from .coord_encoding import ThetaEmbedding, fourier_features, se3_transform


class INRReadoutHead(nn.Module):
    """
    forward(feat, vertex_coords, lattice_times, theta, query_coords, query_times)
      feat:          (B, V, T, C) lattice features (C = d_feat; C=1 for a raw
                     signal lattice)
      vertex_coords: (V, 3) mm
      lattice_times: (T,) seconds, strictly increasing, uniform not required
      theta:         (V, 3) frozen HRF params, or None (zeros are used)
      query_coords:  (Q, 3) mm, shared across the batch
      query_times:   (Q,) seconds
      query_pose:    optional SE(3) pose to transform query coordinates to anatomical space
    returns (B, Q) field values.
    """
    def __init__(self, d_feat: int, d_hidden: int = 64, k_neighbors: int = 6,
                 n_freq_space: int = 8, n_freq_time: int = 8,
                 min_wavelength_mm: float = 2.0, max_wavelength_mm: float = 300.0,
                 min_wavelength_s: float = 0.25, max_wavelength_s: float = 100.0,
                 support_conditioning: bool = False):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.n_freq_space = n_freq_space
        self.n_freq_time = n_freq_time
        self.space_wl = (min_wavelength_mm, max_wavelength_mm)
        self.time_wl = (min_wavelength_s, max_wavelength_s)

        self.theta_embed = ThetaEmbedding(d_hidden)
        d_token = d_feat + 3 * 2 * n_freq_space + 2 * n_freq_time + d_hidden
        self.token_mlp = nn.Sequential(
            nn.Linear(d_token, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.pool_score = nn.Linear(d_hidden, 1)
        self.readout = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.support_proj = nn.Linear(8, d_hidden) if support_conditioning else None

    @staticmethod
    def _time_bracket(lattice_times: torch.Tensor,
                      query_times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bracketing frame indices i0, i1 (Q,) and interp weight w (Q,) toward i1.
        Queries outside [t_0, t_{T-1}] clamp to the edge frame (w=0 or 1)."""
        T = lattice_times.shape[0]
        i1 = torch.searchsorted(lattice_times, query_times).clamp(1, T - 1)
        i0 = i1 - 1
        t0, t1 = lattice_times[i0], lattice_times[i1]
        w = ((query_times - t0) / (t1 - t0)).clamp(0.0, 1.0)
        return i0, i1, w

    def forward(self, feat: torch.Tensor, vertex_coords: torch.Tensor,
                lattice_times: torch.Tensor, theta: torch.Tensor,
                query_coords: torch.Tensor, query_times: torch.Tensor, *,
                observation_support=None, query_support=None,
                query_pose: torch.Tensor | None = None) -> torch.Tensor:
        if query_pose is not None:
            query_coords = se3_transform(query_coords, query_pose, inverse=True)
        B, V, T, C = feat.shape
        Q = query_coords.shape[0]
        k = min(self.k_neighbors, V)

        # 1. k-NN vertices per query (Q, k)
        d2 = torch.cdist(query_coords, vertex_coords)  # (Q, V)
        nn_dist, nn_idx = torch.topk(d2, k, dim=1, largest=False)

        # 2. time-interpolated features of the neighbor vertices -> (B, Q, k, C)
        i0, i1, w = self._time_bracket(lattice_times, query_times)
        # index (vertex, frame) pairs directly — materializing feat[:, nn_idx]
        # as (B, Q, k, T, C) first is O(T) larger and OOMs at lattice-scale Q
        f0 = feat[:, nn_idx, i0.view(Q, 1).expand(Q, k)]  # (B, Q, k, C)
        f1 = feat[:, nn_idx, i1.view(Q, 1).expand(Q, k)]
        w_ = w.view(1, Q, 1, 1)
        f_tau = (1.0 - w_) * f0 + w_ * f1  # (B, Q, k, C)

        # 3. per-neighbor tokens in physical units
        dxyz = vertex_coords[nn_idx] - query_coords.unsqueeze(1)  # (Q, k, 3) mm
        enc_space = fourier_features(dxyz, self.n_freq_space, *self.space_wl)
        dtau = (query_times - lattice_times[i0]).unsqueeze(-1)  # (Q, 1) s
        enc_time = fourier_features(dtau, self.n_freq_time, *self.time_wl)
        enc_time = enc_time.unsqueeze(1).expand(Q, k, enc_time.shape[-1])
        if theta is None:
            theta = vertex_coords.new_zeros(V, 3)
            theta[:, 1:] = 1.0  # neutral width/amplitude for the log transform
        th = self.theta_embed(theta)[nn_idx]  # (Q, k, d_hidden)

        ctx = torch.cat([enc_space, enc_time, th], dim=-1)  # (Q, k, *)
        ctx = ctx.unsqueeze(0).expand(B, Q, k, ctx.shape[-1])
        tokens = self.token_mlp(torch.cat([f_tau, ctx], dim=-1))  # (B, Q, k, d_hidden)
        if self.support_proj is not None:
            if observation_support is None or query_support is None:
                raise ValueError('INR requires both observation and query voxel sizes/TR')
            support = torch.cat((observation_support, query_support)).to(tokens)
            if support.shape != (8,) or not torch.isfinite(support).all() or (support <= 0).any():
                raise ValueError('observation/query supports must contain eight positive finite values')
            tokens = tokens + self.support_proj(support.log())[None, None, None]

        # 4. learned pooling over neighbors, then scalar readout
        attn = torch.softmax(self.pool_score(tokens).squeeze(-1), dim=-1)  # (B, Q, k)
        pooled = (attn.unsqueeze(-1) * tokens).sum(dim=2)  # (B, Q, d_hidden)
        return self.readout(pooled).squeeze(-1)  # (B, Q)
