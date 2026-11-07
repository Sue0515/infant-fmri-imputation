"""
SpaceTimeAttentionNet — the network body: windowed JOINT space-time attention.

Named for what it is. Until 2026-09-10 this was `HemoFlowNet` in
stsr/model/hemoflow.py, registered as `--model hemoflow`. That name read as
flow matching, but this network is not a flow model: the deterministic and
the flow objective BOTH run on it. "HemoFlow" now names the flow-matching
objective (stsr/objectives/hemoflow.py). Checkpoints recorded under the old
name still rebuild through build_trunk(..., allow_legacy=True).

A window = (spatial patch of ~window_vertices neighbouring vertices, cut by
balanced k-means on the coordinate point cloud) x (window_time consecutive
frames); attention is full within a window. Even/odd blocks alternate two
spatial partitions (seed 0/1) and a time offset of window_time//2 — the
point-cloud analogue of shifted windows — so information crosses window
borders over the 16 blocks. Joint rather than factorized because a
traveling BOLD wavefront is a tilted ridge in (space, time) that a single
axial layer cannot attend to.

Conditioning: value channels [x_t, pinned, mask] are concatenated (1x1
conv); xyz (mm) and frozen theta are encoded and ADDED per vertex; times
(s) rotate q/k inside attention (ContinuousTimeRoPE). Optional HF/LF
routing: the gap-filled observed signal (per-row interpolation from cond
entries only, so the mask's own spectrum stays out of the bands) is split
into low/high frequency components and injected as two extra key/value
tokens per window; use_routing=False removes them. No learned parameter
shape depends on (K, L).

spatial_cond_mode='per_block' additionally passes the fixed xyz Fourier
features to every block. Each block has its own bias-free 96 -> d_model
projection, added after LayerNorm in its attention and FFN branches (before
any adaLN modulation). Coordinates remain available even if the incoming
hidden features lose them. The residual stream is not directly incremented
by the position embedding; zero adaLN gates still make each block identity.
'input' keeps the original parameter shapes and forward computation.

attn_mode: 'joint' (default) — full attention inside each space-time
window; 'factorized' — the axial alternative at IDENTICAL parameter count:
blocks cycle [space-only layout (patch x one frame, partition seed 0),
time-only layout (one vertex x `factorized_time_window` frames, offset 0),
space-only (seed 1), time-only (offset W/2)], so every block still has one
attention. Parameters are matched; per-block REACH is matched too when
factorized_time_window == window_time (the default): a joint block sees a
patch x W frames, a factorized block a patch x 1 frame or 1 vertex x W
frames. factorized_time_window=0 makes the time layout span the whole
segment (temporally global per block — a different comparison, state it).
Note the HF/LF routing prompt is the masked mean of a DC-free band over the
window, so it is identically zero in a full-length time window and carries
no information in 'time_only' blocks. A stack of alternating axial layers
CAN represent a tilted (x, t) ridge by composition (two hops); comparing
the factorized and joint conditions measures inductive bias / sample
efficiency, not representability.
Ablations: 'time_only' (each vertex alone x all frames), 'space_only'
(patch x single frame; partitions still alternate seed 0/1).

patch_voxels > 1 turns on SPATIAL COMPRESSION (compress.py): the body runs
on patch tokens instead of voxel tokens and a coordinate-conditioned
decoder restores per-voxel features, so token count falls by that factor
while every head still emits per-voxel values. 0 keeps one token per voxel.

One network body, several heads: `features()` -> feature lattice
(B, K, L, d_model); `lattice()` -> BOLD; `z_lattice()` -> latent z on the
fine grid; `field()` -> value at arbitrary (xyz mm, tau s) via the INR head.
`t_cont` is the flow time of the flow-matching objective (HemoFlow): it
conditions `features()`, and `lattice()` reads the velocity head when it is
given (BOLD-space HemoFlow). Field-space HemoFlow reads `z_lattice()` and
never uses the velocity head. The deterministic objective passes no t_cont.
"""
import hashlib
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..priors.frequency_decomposition import FrequencyDecomposition
from ..utils import masked_row_interp
from .coord_encoding import (ContinuousTimeRoPE, SpatialCoordEncoding,
                             ThetaEmbedding)
from .compress import PatchCompressor
from .inr_head import INRReadoutHead
from .patcher_surface import SurfacePatcher
from .trunk import timestep_embedding
from .z_readout import ZReadoutHead
from .observation_context import GlobalObservationEncoder, ReadGlobalObservations

_ATTN_MODES = ('joint', 'time_only', 'space_only', 'factorized')
ATTN_MODES = _ATTN_MODES
# How the flow time t of the flow-matching objective conditions the body.
# 'add' puts one embedding on the input and lets the blocks carry it; 'adaln'
# is the diffusion-transformer recipe (DiT, Peebles & Xie 2023): each block
# regresses shift/scale/gate for its own norms from t, and the gates start at
# zero so every residual branch is the identity at init. adaln is NOT a
# matched-capacity alternative: its per-block d_model -> 6*d_model modulation
# MLP adds 14.2M parameters at d_model 384 x 16 blocks (+49%). Under the
# deterministic objective there is no t, so adaln is driven by a learned
# constant and carries no information.
_COND_MODES = ('add', 'adaln')
COND_MODES = _COND_MODES
SPATIAL_COND_MODES = ('input', 'per_block')


# ------------------------------------------------------------- window layout

def build_time_windows(L: int, W: int, offset: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group frame indices 0..L-1 into windows of W frames, shifted by
    `offset` (the first/last windows may be partial). Returns
    t_idx (n_tw, W) int64 and t_mask (n_tw, W) bool (False = padding)."""
    ids = (torch.arange(L) + offset) // W
    n_tw = int(ids.max()) + 1
    t_idx = torch.zeros(n_tw, W, dtype=torch.int64)
    t_mask = torch.zeros(n_tw, W, dtype=torch.bool)
    for w in range(n_tw):
        fr = (ids == w).nonzero(as_tuple=True)[0]
        t_idx[w, :fr.numel()] = fr
        t_mask[w, :fr.numel()] = True
    return t_idx, t_mask


def build_windows(slot_index: torch.Tensor, slot_mask: torch.Tensor,
                  t_idx: torch.Tensor, t_mask: torch.Tensor, L: int
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross a spatial partition (P, S) with a temporal windowing (n_tw, W)
    into joint windows over flat (vertex * L + frame) token indices.

    Returns idx (P * n_tw, S * W) int64 (padded entries point at token 0 and
    are excluded by mask) and mask (P * n_tw, S * W) bool."""
    P, S = slot_index.shape
    n_tw, W = t_idx.shape
    flat = slot_index[:, None, :, None] * L + t_idx[None, :, None, :]
    m = slot_mask[:, None, :, None] & t_mask[None, :, None, :]
    idx = flat.reshape(P * n_tw, S * W).masked_fill(
        ~m.reshape(P * n_tw, S * W), 0)
    return idx, m.reshape(P * n_tw, S * W)


# ------------------------------------------------------------------- blocks

class WindowedJointBlock(nn.Module):
    """Pre-LN residual block: joint space-time attention inside the given
    windows (continuous-time RoPE on q/k; optional pooled HF/LF prompt
    tokens appended to keys/values) -> pointwise FFN."""

    def __init__(self, d_model: int, n_heads: int, routing: bool,
                 cond_mode: str = 'add', spatial_dim: int = 0,
                 spatial_relative_bias: bool = False):
        super().__init__()
        self.cond_mode = cond_mode
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.routing = routing
        self.norm_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        if routing:
            self.prompt_h = nn.Linear(2, d_model)
            self.prompt_l = nn.Linear(2, d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        # Each block can read the original coordinates independently of h.
        # One projection per block, shared by its attention and FFN branches;
        # never added directly to the residual stream (adaLN-Zero stays identity).
        self.spatial_proj = (nn.Linear(spatial_dim, d_model, bias=False)
                             if spatial_dim else None)
        self.relative_bias = None
        if spatial_relative_bias:
            self.relative_bias = nn.Sequential(nn.Linear(4, 32), nn.SiLU(),
                                              nn.Linear(32, n_heads, bias=False))
            nn.init.zeros_(self.relative_bias[-1].weight)
        # DiT adaLN-zero: six vectors per block (shift/scale/gate x attn/ffn).
        # Zero init => gate 0 => the block is the identity at step 0, which is
        # what makes deep conditioned stacks trainable without warm-up tricks.
        self.mod = None
        if cond_mode == 'adaln':
            self.mod = nn.Sequential(nn.SiLU(),
                                     nn.Linear(d_model, 6 * d_model))
            nn.init.zeros_(self.mod[1].weight)
            nn.init.zeros_(self.mod[1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor,
                  scale: torch.Tensor) -> torch.Tensor:
        """x (B, ..., C) scaled and shifted by per-item vectors (B, C)."""
        v = (1,) * (x.dim() - 2)
        return x * (1 + scale.reshape(scale.shape[0], *v, -1)) \
            + shift.reshape(shift.shape[0], *v, -1)

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
              ) -> torch.Tensor:
        """x (B, n_win, H, S, hd); cos/sin (n_win, S, hd/2) broadcast."""
        D = x.shape[-1]
        c = cos.unsqueeze(0).unsqueeze(2)  # (1, n_win, 1, S, hd/2)
        s = sin.unsqueeze(0).unsqueeze(2)
        xe, xo = x[..., 0:D:2], x[..., 1:D:2]
        out = torch.stack([xe * c - xo * s, xe * s + xo * c], dim=-1)
        return out.reshape(*x.shape)

    def forward(self, h: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor,
                cos_w: torch.Tensor, sin_w: torch.Tensor,
                cos_ref: torch.Tensor, sin_ref: torch.Tensor,
                praw_h: Optional[torch.Tensor],
                praw_l: Optional[torch.Tensor],
                cvec: Optional[torch.Tensor] = None,
                spatial_features: Optional[torch.Tensor] = None,
                window_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, L, C = h.shape
        pos = None
        if self.spatial_proj is not None:
            if spatial_features is None:
                raise ValueError('per_block spatial conditioning requires coordinates')
            pos = self.spatial_proj(spatial_features).view(1, K, 1, C)
        sh_a = sc_a = ga_a = sh_f = sc_f = ga_f = None
        if self.mod is not None and cvec is not None:
            sh_a, sc_a, ga_a, sh_f, sc_f, ga_f = self.mod(cvec).chunk(6, -1)
        n_win, S = idx.shape
        H, hd = self.n_heads, self.head_dim

        # ---- windowed joint attention
        yf = self.norm_attn(h)
        if pos is not None:
            yf = yf + pos
        if sh_a is not None:
            yf = self._modulate(yf, sh_a, sc_a)
        yf = yf.reshape(B, K * L, C)
        yw = yf[:, idx]                                  # (B, n_win, S, C)
        q, k, v = self.qkv(yw).chunk(3, dim=-1)

        def heads(x):
            return x.reshape(B, n_win, -1, H, hd).permute(0, 1, 3, 2, 4)

        q, k, v = heads(q), heads(k), heads(v)           # (B,n_win,H,S,hd)
        q = self._rope(q, cos_w, sin_w)
        k = self._rope(k, cos_w, sin_w)

        key_mask = mask                                   # (n_win, S)
        if self.routing and praw_h is not None:
            # masked mean of raw [component, observed] channels per window,
            # then embed: pooling first == projecting first (linear).
            denom = mask.sum(-1).clamp_min(1).to(h.dtype)  # (n_win,)
            def pool(praw):
                pw = praw.reshape(B, K * L, 2)[:, idx]     # (B,n_win,S,2)
                pw = (pw * mask[None, :, :, None].to(h.dtype)).sum(2)
                return pw / denom[None, :, None]           # (B, n_win, 2)
            emb = torch.stack(
                [self.prompt_h(pool(praw_h)), self.prompt_l(pool(praw_l))],
                dim=2)                                     # (B, n_win, 2, C)
            _, k_p, v_p = self.qkv(emb).chunk(3, dim=-1)
            # Prompt keys are anchored at the window's reference time (its
            # first frame): rotated q against UNrotated prompt keys would
            # leak the absolute time into the scores and break the global
            # time-shift invariance that RoPE otherwise guarantees.
            k_p = self._rope(heads(k_p),
                             cos_ref.unsqueeze(1).expand(-1, 2, -1),
                             sin_ref.unsqueeze(1).expand(-1, 2, -1))
            k = torch.cat([k, k_p], dim=3)                 # (+2 key tokens)
            v = torch.cat([v, heads(v_p)], dim=3)          # values unrotated
            key_mask = torch.cat(
                [mask, mask.new_ones(n_win, 2)], dim=1)

        # bool attn mask (True = attend); padded QUERIES produce junk but
        # only valid slots are scattered back, so masking keys suffices.
        bias = key_mask[None, :, None, None, :].expand(
            B, n_win, 1, 1, key_mask.shape[-1])
        if self.relative_bias is not None:
            if window_coords is None:
                raise ValueError('spatial relative bias needs window coordinates')
            delta = (window_coords[:, :, None] - window_coords[:, None, :]) / 100.
            relation = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
            # Identical positions have zero bias. Auxiliary HF/LF keys have no
            # anatomical position and receive no geometric bias.
            relation = relation.to(yw)
            geom_bias = self.relative_bias(relation) - self.relative_bias(
                torch.zeros_like(relation[..., :1, :1, :]))
            geom_bias = geom_bias.permute(0, 3, 1, 2)
            geom_bias = F.pad(geom_bias, (0, k.shape[3] - S))
            bias = geom_bias[None].expand(B, -1, -1, -1, -1).masked_fill(~bias, -torch.inf)
            attn_bias = bias.reshape(B * n_win, H, S, -1)
        else:
            attn_bias = bias.reshape(B * n_win, 1, 1, -1)
        total_b = B * n_win
        CHUNK = 8192
        if total_b > CHUNK:
            q_flat = q.reshape(total_b, H, S, hd)
            k_flat = k.reshape(total_b, H, -1, hd)
            v_flat = v.reshape(total_b, H, -1, hd)
            outs = []
            for start in range(0, total_b, CHUNK):
                end = min(start + CHUNK, total_b)
                b_mask = attn_bias[start:end] if (attn_bias.shape[0] == total_b) else attn_bias
                out_chunk = F.scaled_dot_product_attention(
                    q_flat[start:end], k_flat[start:end], v_flat[start:end],
                    attn_mask=b_mask)
                outs.append(out_chunk)
            out = torch.cat(outs, dim=0)
        else:
            out = F.scaled_dot_product_attention(
                q.reshape(B * n_win, H, S, hd),
                k.reshape(B * n_win, H, -1, hd),
                v.reshape(B * n_win, H, -1, hd),
                attn_mask=attn_bias)
        out = out.reshape(B, n_win, H, S, hd).permute(0, 1, 3, 2, 4)
        out = self.out(out.reshape(B, n_win, S, C))
        if ga_a is not None:
            out = out * ga_a.reshape(B, 1, 1, C)

        flat = h.reshape(B, K * L, C).clone()
        valid = mask.reshape(-1)
        flat[:, idx.reshape(-1)[valid]] += \
            out.reshape(B, n_win * S, C)[:, valid]
        h = flat.reshape(B, K, L, C)

        # ---- pointwise FFN
        y = self.norm_ffn(h)
        if pos is not None:
            y = y + pos
        if sh_f is not None:
            y = self._modulate(y, sh_f, sc_f)
        y = self.ffn(y)
        if ga_f is not None:
            y = y * ga_f.reshape(B, 1, 1, C)
        return h + y


# ------------------------------------------------------------- network body

class SpaceTimeAttentionNet(nn.Module):
    """
    forward(x_t, cond_pinned, cond_mask, coords_mm, times_s, theta=None,
            t_cont=None) -> (B, K, L)
      x_t:         (B, K, L) flow state (velocity mode); pass cond_pinned in
                   deterministic mode
      cond_pinned: (B, K, L) observed values, zeros where unobserved
      cond_mask:   (B, K, L) 1 = observed, 0 = to reconstruct
      coords_mm:   (K, 3) vertex coordinates, millimetres
      times_s:     (L,) sample times in SECONDS, uniform grid (the latent
                   lattice; window layout is index-based, so uniformity is
                   asserted), never frame indices
      theta:       (K, 3) frozen HRF [delay_s, width, amp] or None
      t_cont:      (B,) flow time in [0,1] -> velocity head; None ->
                   deterministic head
    """

    def __init__(self, d_model: int = 384, n_blocks: int = 16,
                 n_heads: int = 8, attn_mode: str = 'joint',
                 cond_mode: str = 'add',
                 window_vertices: int = 24, window_time: int = 8,
                 factorized_time_window: Optional[int] = None,
                 patch_voxels: int = 0,
                 use_routing: bool = True, grad_checkpointing: bool = False,
                 lf_cutoff_hz: float = 0.1, hf_cutoff_hz: float = 0.2,
                 with_inr_head: bool = True, inr_hidden: int = 64,
                 inr_k_neighbors: int = 6,
                 with_z_head: bool = True, z_hidden: int = 64,
                 z_init_gain: float = 0.1,
                 spatial_cond_mode: str = 'input',
                 spatial_relative_bias: bool = False,
                 support_conditioning: bool = False,
                 global_context_tokens: int = 0, global_context_cells: int = 128,
                 global_context_every: int = 4, attention_window_seconds: float = 0.,
                 functional_gradient: bool = False, functional_gradient_components: int = 5,
                 jepa_aux: bool = False, jepa_hidden: int = 384):
        super().__init__()
        if attn_mode not in _ATTN_MODES:
            raise ValueError(f"attn_mode must be one of {_ATTN_MODES}")
        if cond_mode not in _COND_MODES:
            raise ValueError(f"cond_mode must be one of {_COND_MODES}")
        if spatial_cond_mode not in SPATIAL_COND_MODES:
            raise ValueError(f"spatial_cond_mode must be one of {SPATIAL_COND_MODES}")
        self.spatial_cond_mode = spatial_cond_mode
        if functional_gradient_components < 1:
            raise ValueError('functional_gradient_components must be positive')
        if functional_gradient and (attn_mode != 'factorized' or patch_voxels > 1):
            raise ValueError('functional gradient requires factorized attention without patch compression')
        if global_context_tokens < 0 or global_context_cells < 1 or global_context_every < 1:
            raise ValueError('global context counts must be positive (tokens=0 disables it)')
        if attention_window_seconds < 0 or not math.isfinite(attention_window_seconds):
            raise ValueError('attention_window_seconds must be finite and nonnegative')
        if attention_window_seconds and (attn_mode != 'factorized' or factorized_time_window == 0):
            raise ValueError('physical time windows require local factorized attention')
        self.spatial_relative_bias = spatial_relative_bias
        self.support_conditioning = support_conditioning
        self.global_context_tokens = int(global_context_tokens)
        self.global_context_cells = int(global_context_cells)
        self.attention_window_seconds = float(attention_window_seconds)
        self.cond_mode = cond_mode
        self.d_model = d_model
        self.attn_mode = attn_mode
        self.window_vertices = int(window_vertices)
        self.window_time = int(window_time)
        # None -> window_time (matched per-block reach); 0 -> whole segment
        self.factorized_time_window = (
            int(window_time) if factorized_time_window is None
            else int(factorized_time_window))
        self.use_routing = use_routing
        self.grad_checkpointing = grad_checkpointing
        self.lf_cutoff_hz = lf_cutoff_hz
        self.hf_cutoff_hz = hf_cutoff_hz

        self.input_proj = nn.Conv1d(3, d_model, kernel_size=1)
        self.t_proj = nn.Linear(d_model, d_model)  # embeds the flow time; unused by det
        self.coord_enc = SpatialCoordEncoding(d_model)
        self.theta_embed = ThetaEmbedding(d_model)
        self.rope = ContinuousTimeRoPE(d_model // n_heads)
        self.blocks = nn.ModuleList([
            WindowedJointBlock(d_model, n_heads, routing=use_routing,
                               cond_mode=cond_mode,
                               spatial_dim=(self.coord_enc.proj.in_features
                                            if spatial_cond_mode == 'per_block' else 0),
                               spatial_relative_bias=spatial_relative_bias)
            for _ in range(n_blocks)
        ])
        # adaLN needs a conditioning vector even with no flow time (the
        # deterministic head): a learned null token keeps the same body
        # usable under both objectives instead of silently zeroing every gate.
        self.null_cond = nn.Parameter(torch.zeros(d_model)) \
            if cond_mode == 'adaln' else None
        self.norm_out = nn.LayerNorm(d_model)
        self.head_det = nn.Conv1d(d_model, 1, kernel_size=1)
        self.head_vel = nn.Conv1d(d_model, 1, kernel_size=1)
        self.inr_head = INRReadoutHead(
            d_feat=d_model, d_hidden=inr_hidden,
            k_neighbors=inr_k_neighbors,
            support_conditioning=support_conditioning) if with_inr_head else None
        self.z_head = ZReadoutHead(
            d_model, hidden=z_hidden,
            init_gain=z_init_gain) if with_z_head else None
        # spatial compression: 0 = one token per voxel (default), N = pool
        # ~N voxels into one body token and decode back (see compress.py)
        self.compressor = PatchCompressor(d_model, patch_voxels=patch_voxels) \
            if patch_voxels and patch_voxels > 1 else None
        self._win_cache: dict = {}
        self.support_proj = nn.Linear(4, d_model) if support_conditioning else None
        self.global_encoder = (GlobalObservationEncoder(global_context_tokens)
                               if global_context_tokens else None)
        self.global_readers = nn.ModuleDict({
            str(i): ReadGlobalObservations(d_model)
            for i in range(n_blocks)
            if global_context_tokens and (i % global_context_every == 0)})
        self.functional_proj = None
        if functional_gradient:
            # Preserve both the initial function and the RNG stream of the
            # local/global baseline when adding this zero-initialized path.
            with torch.random.fork_rng(devices=[]):
                self.functional_proj = nn.Linear(functional_gradient_components + 1,
                                                 d_model, bias=False)
            nn.init.zeros_(self.functional_proj.weight)
        self.jepa_predictor = None
        if jepa_aux:
            if attn_mode != 'factorized' or patch_voxels > 1 or jepa_hidden < 1:
                raise ValueError('JEPA requires factorized attention, no spatial compression and positive hidden size')
            from ..jepa import JepaPredictor
            with torch.random.fork_rng(devices=[]):
                self.jepa_predictor = JepaPredictor(d_model, jepa_hidden)

    # -------------------------------------------------------- window layout
    def _layouts(self, coords_mm: torch.Tensor, L: int, times_s=None):
        """Per-parity window layouts [(idx, mask), ...] for this vertex set
        and frame count; deterministic, cached by content hash."""
        K = coords_mm.shape[0]
        time_window = self.factorized_time_window
        if self.attention_window_seconds:
            if times_s is None or len(times_s) < 2:
                raise ValueError('physical time windows require times in seconds')
            dt = float(times_s[1] - times_s[0])
            time_window = max(1, min(L, math.floor(self.attention_window_seconds / dt + 1e-6) + 1))
        key = (hashlib.sha1(
            coords_mm.detach().cpu().numpy().tobytes()).hexdigest(),
            K, L, self.attn_mode, self.window_vertices, self.window_time,
            time_window)
        hit = self._win_cache.get(key)
        if hit is not None:
            return hit

        if self.attn_mode == 'time_only':
            slot = torch.arange(K).unsqueeze(1)
            smask = torch.ones(K, 1, dtype=torch.bool)
            t_idx, t_mask = build_time_windows(L, L, 0)
            layouts = [build_windows(slot, smask, t_idx, t_mask, L)]
        elif self.attn_mode == 'factorized':
            # axial: [space(seed 0), time(off 0), space(seed 1), time(off W/2)]
            # cycling over the blocks — one attention per block, same
            # parameters as joint; time windows of factorized_time_window
            # frames (0 = the whole segment), shifted like joint's
            n_patches = max(1, round(K / self.window_vertices))
            coords_cpu = coords_mm.detach().float().cpu()
            slot = torch.arange(K).unsqueeze(1)
            smask = torch.ones(K, 1, dtype=torch.bool)
            Wt = time_window or L
            t_one, tm_one = build_time_windows(L, 1, 0)
            layouts = []
            for parity, off in ((0, 0), (1, Wt // 2)):
                patcher = SurfacePatcher(coords_cpu, n_patches, seed=parity)
                layouts.append(build_windows(
                    patcher.slot_index, patcher.slot_mask, t_one, tm_one, L))
                t_w, tm_w = build_time_windows(L, Wt, off % max(Wt, 1))
                layouts.append(build_windows(slot, smask, t_w, tm_w, L))
        else:
            n_patches = max(1, round(K / self.window_vertices))
            coords_cpu = coords_mm.detach().float().cpu()
            W = 1 if self.attn_mode == 'space_only' else self.window_time
            layouts = []
            # space_only alternates the two spatial partitions (seed 0/1)
            # exactly like joint; the time offset is moot at W=1. With a
            # single layout (pre-2026-08-31) all 16 blocks attended inside
            # the SAME ~window_vertices patch, so that condition measured a frozen
            # receptive field, not spatial-only attention.
            offsets = [0, 0] if self.attn_mode == 'space_only' \
                else [0, self.window_time // 2]
            for parity, off in enumerate(offsets):
                patcher = SurfacePatcher(coords_cpu, n_patches, seed=parity)
                t_idx, t_mask = build_time_windows(L, W, off % max(W, 1))
                layouts.append(build_windows(
                    patcher.slot_index, patcher.slot_mask, t_idx, t_mask, L))
        self._win_cache[key] = layouts
        return layouts

    # -------------------------------------------------------------- forward
    def features(self, x_t: torch.Tensor, cond_pinned: torch.Tensor,
                 cond_mask: torch.Tensor, coords_mm: torch.Tensor,
                 times_s: torch.Tensor, theta: Optional[torch.Tensor] = None,
                 t_cont: Optional[torch.Tensor] = None, *,
                 observation_support=None, global_context=None,
                 functional_coordinates=None) -> torch.Tensor:
        """One pass of the network body -> the feature lattice (B, K, L, d_model): the
        shared substrate of the lattice heads AND the field readout (compute
        it once when both outputs are needed)."""
        B, K, L = x_t.shape
        assert coords_mm.shape == (K, 3), \
            f"coords_mm must be (K,3), got {tuple(coords_mm.shape)}"
        assert times_s.shape == (L,), \
            f"times_s must be (L,), got {tuple(times_s.shape)}"
        if L > 1:
            steps = times_s.diff()
            assert torch.allclose(steps, steps[0], rtol=1e-4), \
                "times_s must be a uniform grid (window layout is index-based)"

        chan = torch.stack([x_t, cond_pinned, cond_mask], dim=1)  # (B,3,K,L)
        if theta is None:
            theta = coords_mm.new_zeros(K, 3)
            theta[:, 1:] = 1.0  # neutral width/amp under the log transform
        # SPATIAL COMPRESSION (optional): the body runs on patch tokens and
        # the decoder puts per-voxel detail back, so every head below still
        # sees (B, V, L, C). Coordinates and theta become patch-level too.
        coords_body, theta_body = coords_mm, theta
        if self.compressor is not None:
            chan, coords_body, theta_body = self.compressor.pool(
                chan, coords_mm, theta)
            K = chan.shape[2]
        cond_pinned, cond_mask = chan[:, 1], chan[:, 2]

        h = self.input_proj(chan.reshape(B, 3, K * L))
        cvec = None
        if self.cond_mode == 'adaln':
            cvec = (self.t_proj(timestep_embedding(t_cont, self.d_model))
                    if t_cont is not None
                    else self.null_cond.expand(B, self.d_model))
        elif t_cont is not None:
            t_emb = self.t_proj(timestep_embedding(t_cont, self.d_model))
            h = h + t_emb.unsqueeze(-1)
        h = h.reshape(B, self.d_model, K, L).permute(0, 2, 3, 1)  # (B,K,L,C)

        spatial_features = None
        if self.spatial_cond_mode == 'per_block':
            spatial_features = self.coord_enc.encode(coords_body)
            spatial_row = self.coord_enc.proj(spatial_features)
        else:
            spatial_row = self.coord_enc(coords_body)
        row = spatial_row + self.theta_embed(theta_body)
        h = h + row.view(1, K, 1, self.d_model)
        if self.functional_proj is not None:
            if (functional_coordinates is None or
                    functional_coordinates.shape != (K, self.functional_proj.in_features) or
                    not torch.isfinite(functional_coordinates).all()):
                raise ValueError('functional gradient requires finite support-averaged coordinates')
            h = h + self.functional_proj(functional_coordinates.to(h))[None, :, None, :]
        if self.support_proj is not None:
            if observation_support is None or observation_support.shape != (4,) or not (
                    torch.isfinite(observation_support).all() and (observation_support > 0).all()):
                raise ValueError('support conditioning requires positive voxel sizes and TR')
            h = h + self.support_proj(observation_support.to(h).log())[None, None, None]
        summaries = present = None
        if self.global_encoder is not None:
            if global_context is None:
                raise ValueError('global context requires observation-only full-field summaries')
            if global_context['times'].shape != times_s.shape or not torch.allclose(
                    global_context['times'], times_s):
                raise ValueError('local and global observations must use the same physical times')
            summaries, present = self.global_encoder(global_context)

        praw_h = praw_l = None
        if self.use_routing:
            dt = float(times_s[1] - times_s[0]) if L > 1 else 1.0
            nyquist = 0.5 / dt
            hf = min(self.hf_cutoff_hz, 0.99 * nyquist)
            lf = min(self.lf_cutoff_hz, 0.5 * hf)
            decomp = FrequencyDecomposition(sampling_rate=dt, lf_cutoff=lf,
                                            hf_cutoff=hf)
            # Gap-fill before the FFT: zero-filled hidden entries put the
            # MASK's spectrum into the HF band (a checkerboard is a broadband
            # comb). The fill reads cond entries only — no leakage — and the
            # mask still rides along as its own prompt channel below.
            x_lf, x_hf = decomp(masked_row_interp(cond_pinned, cond_mask))
            praw_h = torch.stack([x_hf, cond_mask], dim=-1)  # (B,K,L,2)
            praw_l = torch.stack([x_lf, cond_mask], dim=-1)

        cos_L, sin_L = self.rope.cos_sin(times_s)  # (L, hd/2)
        layouts = self._layouts(coords_body, L, times_s)
        dev = h.device
        per_parity = []
        for idx, mask in layouts:
            idx, mask = idx.to(dev), mask.to(dev)
            t_of_tok = idx % L
            # slot 0 of every window row is always a valid token (patches and
            # time windows are non-empty and fill slots front-to-back), so it
            # serves as the window's reference time for prompt-key rotation.
            per_parity.append(
                (idx, mask, cos_L[t_of_tok], sin_L[t_of_tok],
                 cos_L[t_of_tok[:, 0]], sin_L[t_of_tok[:, 0]]))

        for b, block in enumerate(self.blocks):
            if str(b) in self.global_readers:
                h = self.global_readers[str(b)](h, summaries, present)
            idx, mask, cos_w, sin_w, cos_r, sin_r = \
                per_parity[b % len(per_parity)]
            window_coords = coords_body[idx // L] if self.spatial_relative_bias else None
            if self.grad_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, idx, mask, cos_w, sin_w, cos_r, sin_r,
                    praw_h, praw_l, cvec, spatial_features, window_coords, use_reentrant=False)
            else:
                h = block(h, idx, mask, cos_w, sin_w, cos_r, sin_r,
                          praw_h, praw_l, cvec, spatial_features, window_coords)
        h = self.norm_out(h)                    # (B, K_body, L, C)
        if self.compressor is not None:
            h = self.compressor.expand(h, coords_mm)
        return h                                # (B, V, L, C)

    def lattice(self, feat: torch.Tensor,
                t_cont: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feature lattice -> (B, K, L) via the deterministic head
        (t_cont=None) or the velocity head. Pass the SAME t_cont that
        produced `feat` — it selects the matching head."""
        B, K, L, C = feat.shape
        hf = feat.permute(0, 3, 1, 2).reshape(B, C, K * L)
        head = self.head_det if t_cont is None else self.head_vel
        return head(hf).reshape(B, K, L)

    def z_lattice(self, feat: torch.Tensor, up: int, dt: float,
                  n_pre: int = 0) -> torch.Tensor:
        """Feature lattice -> latent neural state (B, K, n_pre + L*up) on the
        fine dt grid — the physics-path counterpart of `lattice()`.

        `lattice()` emits BOLD directly and so has no measurement model under
        it; this emits the quantity the hemodynamic operator renders FROM, so
        H_theta(z) is what gets compared to data and the exact DC projection
        has something to project. Native frame l sits at fine index
        n_pre + l*up (see ZReadoutHead on why the pre-roll is mandatory once
        anything is projected)."""
        if self.z_head is None:
            raise RuntimeError(
                "z_lattice needs with_z_head=True at construction")
        return self.z_head(feat, up, dt, n_pre=n_pre,
                           grad_checkpointing=self.grad_checkpointing)

    def forward(self, x_t: torch.Tensor, cond_pinned: torch.Tensor,
                cond_mask: torch.Tensor, coords_mm: torch.Tensor,
                times_s: torch.Tensor, theta: Optional[torch.Tensor] = None,
                t_cont: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.lattice(
            self.features(x_t, cond_pinned, cond_mask, coords_mm, times_s,
                          theta, t_cont), t_cont)

    def field(self, feat: torch.Tensor, coords_mm: torch.Tensor,
              times_s: torch.Tensor, theta: Optional[torch.Tensor],
              query_coords: torch.Tensor, query_times: torch.Tensor, *,
              observation_support=None, query_support=None
              ) -> torch.Tensor:
        """Continuous-coordinate query of the hybrid field: arbitrary
        (xyz mm, tau s) against the feature lattice from `features()`.
        Returns (B, Q). Answers between lattice points are trustworthy only
        inside the physical band (module docstring)."""
        if self.inr_head is None:
            raise RuntimeError(
                "field queries need with_inr_head=True at construction")
        return self.inr_head(feat, coords_mm, times_s, theta,
                             query_coords, query_times,
                             observation_support=observation_support, query_support=query_support)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
