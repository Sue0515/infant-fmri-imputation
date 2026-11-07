"""
Spatial compression for the network body: voxel tokens -> patch tokens ->
voxel features. The volume-level foundation models all do this (SwiFT and
NeuroSTORM pool 6x6x6 voxels into one token, a 216x reduction; BrainLM
averages the brain into 424 parcels), and SwiFUN shows that a patch-token
encoder can still emit a per-voxel output if a decoder puts the detail back.

Ours differs from theirs in one way that matters: their patches are boxes on
a fixed grid, so a patch is only defined once the grid is. Here the patches
come from balanced k-means over the millimetre point cloud (SurfacePatcher),
so `patch_voxels` means the same thing on any grid and no learned parameter
shape depends on V.

    pool(x, coords, theta) -> patch signals (B, C, P, L), centroids (P, 3),
                              patch theta (P, 3)
    expand(feat_p, coords) -> per-voxel features (B, V, L, d_model)

The expansion is the piece that keeps the task well posed: a voxel's feature
is its patch's feature conditioned on the voxel's OFFSET from the patch
centroid (Fourier-encoded millimetres), so two voxels in one patch do not
collapse to the same value. Without it, pooling would delete exactly the
spatial detail a super-resolution model is asked to recover.

What compression costs is not hidden: the observation mask is pooled too, so
inside a patch the network sees the FRACTION of observed voxels rather than
which ones. That is the trade this module exists to measure.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .coord_encoding import fourier_features
from .patcher_surface import SurfacePatcher


class PatchCompressor(nn.Module):
    """Balanced k-means patch pooling with a coordinate-conditioned decoder.

    Args:
        d_model:       feature width of the network body.
        patch_voxels:  target voxels per patch token (compression factor).
        n_freqs:       Fourier bands for the within-patch offset.
        min/max_wavelength_mm: offset encoding range. The lower end sits at
            the BOLD point-spread floor; the upper end at a few patch widths.
        hidden:        decoder width (default d_model).
    """

    def __init__(self, d_model: int, patch_voxels: int = 8,
                 n_freqs: int = 8, min_wavelength_mm: float = 2.0,
                 max_wavelength_mm: float = 60.0,
                 hidden: Optional[int] = None, chunk_voxels: int = 4096):
        super().__init__()
        if patch_voxels < 1:
            raise ValueError(f'patch_voxels must be >= 1, got {patch_voxels}')
        self.patch_voxels = int(patch_voxels)
        self.n_freqs = int(n_freqs)
        self.min_wavelength = float(min_wavelength_mm)
        self.max_wavelength = float(max_wavelength_mm)
        hidden = hidden or d_model
        d_off = 3 * 2 * self.n_freqs
        self.decode = nn.Sequential(
            nn.Linear(d_model + d_off, hidden), nn.GELU(),
            nn.Linear(hidden, d_model))
        # the decoder is the memory peak of this module: its input is
        # (B, V, L, d_model + 3*2*n_freqs), so at whole-brain V it is larger
        # than everything the compressed body saved. Run it in voxel chunks
        # under recompute, which caps the live activation at one chunk.
        self.chunk_voxels = int(chunk_voxels)
        self._cache: Dict[object, SurfacePatcher] = {}

    # ------------------------------------------------------------- patches
    def patcher(self, coords_mm: torch.Tensor) -> SurfacePatcher:
        """Balanced k-means partition of this cloud, cached by content."""
        import hashlib
        key = (hashlib.sha1(
            coords_mm.detach().cpu().numpy().tobytes()).hexdigest(),
            self.patch_voxels)
        hit = self._cache.get(key)
        if hit is None:
            V = coords_mm.shape[0]
            n_patches = max(1, min(V, round(V / self.patch_voxels)))
            hit = SurfacePatcher(coords_mm.detach().float().cpu(), n_patches)
            self._cache[key] = hit
        return hit

    def n_patches(self, coords_mm: torch.Tensor) -> int:
        return self.patcher(coords_mm).n_patches

    # ---------------------------------------------------------------- pool
    def pool(self, x: torch.Tensor, coords_mm: torch.Tensor,
             theta: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """(B, C, V, L) -> (B, C, P, L), patch centroids (P, 3), patch theta.

        Members of a patch are averaged. The validity mask that `gather`
        returns keeps padded slots out of the mean, so patches of unequal
        size pool correctly.
        """
        p = self.patcher(coords_mm)
        dev = x.device
        idx = p.patch_index.to(dev)                       # (V,)
        slot_mask = p.slot_mask.to(dev)                   # (P, S)
        B, C, V, L = x.shape
        flat = x.permute(0, 1, 3, 2).reshape(B * C * L, V)
        gathered = flat[:, p.slot_index.to(dev).reshape(-1)]
        gathered = gathered.reshape(B * C * L, p.n_patches, p.max_size)
        m = slot_mask.unsqueeze(0).to(x.dtype)
        pooled = (gathered * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        pooled = pooled.reshape(B, C, L, p.n_patches).permute(0, 1, 3, 2)
        centroids = p.patch_centroids.to(dev).to(coords_mm.dtype)
        theta_p = None
        if theta is not None:
            theta_p = torch.zeros(p.n_patches, theta.shape[1],
                                  device=dev, dtype=theta.dtype)
            count = torch.zeros(p.n_patches, device=dev, dtype=theta.dtype)
            theta_p.index_add_(0, idx, theta)
            count.index_add_(0, idx, torch.ones_like(idx, dtype=theta.dtype))
            theta_p = theta_p / count.clamp_min(1.0).unsqueeze(1)
        return pooled, centroids, theta_p

    # -------------------------------------------------------------- expand
    def expand(self, feat_p: torch.Tensor, coords_mm: torch.Tensor
               ) -> torch.Tensor:
        """(B, P, L, d_model) -> (B, V, L, d_model).

        Each voxel reads its own patch's feature plus a Fourier encoding of
        its offset from the patch centroid, so voxels inside one patch stay
        distinguishable.
        """
        p = self.patcher(coords_mm)
        dev = feat_p.device
        idx = p.patch_index.to(dev)                        # (V,)
        offset = coords_mm - p.patch_centroids.to(dev).to(coords_mm.dtype)[idx]
        enc = fourier_features(offset, self.n_freqs,
                               self.min_wavelength, self.max_wavelength)
        B, P, L, C = feat_p.shape
        V = coords_mm.shape[0]
        enc = enc.to(feat_p.dtype)
        step = self.chunk_voxels if self.chunk_voxels > 0 else V

        def block(f_chunk, e_chunk):
            e = e_chunk.view(1, -1, 1, e_chunk.shape[-1]).expand(
                f_chunk.shape[0], -1, f_chunk.shape[2], -1)
            return f_chunk + self.decode(torch.cat([f_chunk, e], dim=-1))

        out = []
        for start in range(0, V, step):
            sl = slice(start, min(start + step, V))
            f_chunk = feat_p[:, idx[sl]]                   # (B, v, L, C)
            if self.training and torch.is_grad_enabled():
                out.append(torch.utils.checkpoint.checkpoint(
                    block, f_chunk, enc[sl], use_reentrant=False))
            else:
                out.append(block(f_chunk, enc[sl]))
        return torch.cat(out, dim=1)                       # (B, V, L, C)
