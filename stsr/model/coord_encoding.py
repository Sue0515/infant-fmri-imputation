"""
Coordinate conditioning in physical units. Positions enter through fixed
sin/cos formulas — never a learned table indexed by V or T — so the same
weights are defined on any vertex set and any TR, and "3 mm apart" keeps
its meaning across grids.

  SpatialCoordEncoding   xyz (mm) -> multi-scale sin/cos -> Linear -> d_model
  ContinuousTimeRoPE     times in SECONDS (never frame indices) -> rotary
                         q/k phases; attention scores depend only on
                         pairwise time differences (time-shift invariant)
  ThetaEmbedding         frozen per-vertex HRF (delay_s, width, amp) ->
                         d_model. The delay map is itself positional
                         information: each vertex's sub-TR sampling phase.
"""
import math
from typing import Tuple

import torch
import torch.nn as nn


def fourier_features(
    x: torch.Tensor,
    n_freqs: int,
    min_wavelength: float,
    max_wavelength: float,
) -> torch.Tensor:
    """
    Multi-scale sin/cos features of continuous physical inputs.

    x: (..., D) values in physical units (mm, seconds, ...). Each of the D
    input axes is encoded at n_freqs log-spaced wavelengths in
    [min_wavelength, max_wavelength] (same units as x):

        out = [sin(2*pi*x/lam_1), cos(2*pi*x/lam_1), ..., sin(2*pi*x/lam_F), ...]

    Returns (..., D * 2 * n_freqs). Fixed formula, no parameters.
    """
    assert n_freqs >= 1 and 0.0 < min_wavelength <= max_wavelength
    if n_freqs == 1:
        wavelengths = torch.tensor([min_wavelength], dtype=x.dtype, device=x.device)
    else:
        wavelengths = torch.logspace(
            math.log10(min_wavelength), math.log10(max_wavelength),
            n_freqs, dtype=x.dtype, device=x.device,
        )
    angles = 2.0 * math.pi * x.unsqueeze(-1) / wavelengths  # (..., D, F)
    feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., D, 2F)
    return feats.reshape(*x.shape[:-1], x.shape[-1] * 2 * n_freqs)


def se3_transform(coords: torch.Tensor, pose: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
                  inverse: bool = True) -> torch.Tensor:
    """Rigid-body SE(3) transformation: x_anat = T^{-1}(tau)*x_scanner or x = T(tau)*x_anat.

    coords: (..., 3) coordinates in physical units (mm).
    pose:
      - (..., 4, 4) homogeneous transformation matrix
      - (..., 6) 6-DOF vector [rx, ry, rz, tx, ty, tz] with rodrigues rotation
      - tuple (R, t) with R (..., 3, 3) and t (..., 3)
    """
    if isinstance(pose, (tuple, list)):
        R, t = pose
    elif pose.shape[-2:] == (4, 4):
        R = pose[..., :3, :3]
        t = pose[..., :3, 3]
    elif pose.shape[-1] == 6:
        rot_vec = pose[..., :3]
        t = pose[..., 3:]
        theta = torch.norm(rot_vec, dim=-1, keepdim=True).clamp_min(1e-12)
        k = rot_vec / theta
        K = torch.zeros(*pose.shape[:-1], 3, 3, device=pose.device, dtype=pose.dtype)
        K[..., 0, 1] = -k[..., 2]
        K[..., 0, 2] = k[..., 1]
        K[..., 1, 0] = k[..., 2]
        K[..., 1, 2] = -k[..., 0]
        K[..., 2, 0] = -k[..., 1]
        K[..., 2, 1] = k[..., 0]
        I = torch.eye(3, device=pose.device, dtype=pose.dtype).expand(*K.shape)
        sin_th = torch.sin(theta).unsqueeze(-1)
        cos_th = torch.cos(theta).unsqueeze(-1)
        R = I + sin_th * K + (1.0 - cos_th) * torch.matmul(K, K)
    else:
        raise ValueError(f"Unrecognized pose shape: {tuple(pose.shape)}")

    R = R.to(dtype=coords.dtype, device=coords.device)
    t = t.to(dtype=coords.dtype, device=coords.device)

    if inverse:
        # x_anat = R^T (x - t)
        diff = coords - t
        return torch.matmul(diff, R)
    else:
        # x = R x_anat + t
        return torch.matmul(coords, R.transpose(-1, -2)) + t


class SpatialCoordEncoding(nn.Module):
    """
    xyz (mm) -> d_model. Wavelengths default to 2-300 mm: the lower end sits at
    the BOLD point-spread floor (no structure below ~2.3 mm at 7T is claimable),
    the upper end spans a hemisphere.
    """
    def __init__(self, d_model: int, n_freqs: int = 16,
                 min_wavelength_mm: float = 2.0, max_wavelength_mm: float = 300.0):
        super().__init__()
        self.n_freqs = n_freqs
        self.min_wavelength = min_wavelength_mm
        self.max_wavelength = max_wavelength_mm
        self.proj = nn.Linear(3 * 2 * n_freqs, d_model)

    def forward(self, coords_mm: torch.Tensor, pose: torch.Tensor | None = None) -> torch.Tensor:
        """coords_mm (..., 3) -> (..., d_model), optionally conditioned on SE(3) pose."""
        if pose is not None:
            coords_mm = se3_transform(coords_mm, pose, inverse=True)
        return self.proj(self.encode(coords_mm))

    def encode(self, coords_mm: torch.Tensor, pose: torch.Tensor | None = None) -> torch.Tensor:
        """Fixed Fourier features, also used by independent block projections."""
        if pose is not None:
            coords_mm = se3_transform(coords_mm, pose, inverse=True)
        assert coords_mm.shape[-1] == 3, f"expected (...,3), got {tuple(coords_mm.shape)}"
        return fourier_features(
            coords_mm, self.n_freqs, self.min_wavelength, self.max_wavelength)


class ContinuousTimeRoPE(nn.Module):
    """
    Rotary positional encoding driven by real-valued times in SECONDS.

    angles(t)_i = 2*pi * t / lam_i, lam_i log-spaced in
    [min_wavelength_s, max_wavelength_s]. Because rotations of q and k compose
    to a net rotation by (t_q - t_k), post-rotation inner products depend only
    on pairwise time differences: adding a constant to every time leaves
    attention scores unchanged (tested), while changing the sampling interval
    changes them — exactly the grid-independence contract.

    Defaults 0.5-2000 s: half the 0.25 s latent grid step up to a full run.
    """
    def __init__(self, head_dim: int,
                 min_wavelength_s: float = 0.5, max_wavelength_s: float = 2000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for pairwise rotation"
        n = head_dim // 2
        wavelengths = torch.logspace(
            math.log10(min_wavelength_s), math.log10(max_wavelength_s), n,
            dtype=torch.float64)
        self.register_buffer("omega", (2.0 * math.pi / wavelengths), persistent=False)

    def cos_sin(self, times_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """times_s (L,) seconds -> cos, sin each (L, head_dim/2)."""
        assert times_s.ndim == 1
        angles = times_s.to(torch.float64).unsqueeze(-1) * self.omega  # (L, D/2)
        return angles.cos().to(times_s.dtype), angles.sin().to(times_s.dtype)


class ThetaEmbedding(nn.Module):
    """
    Per-vertex frozen HRF parameters theta (V, 3) = [delay_shift_s,
    width_scale, amplitude] -> (V, d_model). delay enters raw (seconds, signed);
    width/amplitude are positive multiplicative quantities and enter as logs.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(3, d_model)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        assert theta.shape[-1] == 3, f"expected (...,3), got {tuple(theta.shape)}"
        delay = theta[..., 0:1]
        log_width = theta[..., 1:2].clamp_min(1e-6).log()
        log_amp = theta[..., 2:3].clamp_min(1e-6).log()
        return self.proj(torch.cat([delay, log_width, log_amp], dim=-1))
