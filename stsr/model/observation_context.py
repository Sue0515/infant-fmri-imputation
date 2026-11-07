"""Optional STSR context paths; defaults preserve the archived architecture."""
import math

import torch
from torch import nn

from .coord_encoding import fourier_features


DESIGN_DEFAULTS = dict(spatial_relative_bias=False, support_conditioning=False,
    global_context_tokens=0, global_context_cells=128, global_context_every=4,
    attention_window_seconds=0., functional_gradient=False, functional_gradient_components=5,
    jepa_aux=False, jepa_hidden=384)


def support_vector(basis, times):
    """Three voxel edge lengths (mm) and sampling interval (s), not time averaging."""
    return torch.cat((basis.norm(dim=0), times.diff().median().reshape(1)))


def pool_observations(values, valid, coords, times, max_cells):
    """Bounded spatial summaries of OBSERVATIONS, keeping every time separately.

    Fixed geometric bins never use targets or signal correlations. Missing values
    contribute neither to sums nor to denominators. The local path keeps detail.
    """
    side = max(1, math.floor(max_cells ** (1 / 3) + 1e-6))
    extent = (coords.amax(0) - coords.amin(0)).clamp_min(1e-6)
    bins = ((coords - coords.amin(0)) / extent * side).long().clamp_max(side - 1)
    _, group = torch.unique(bins, dim=0, return_inverse=True)
    count = torch.bincount(group).to(values)
    n = len(count)
    good = valid.bool() & torch.isfinite(values)
    total = values.new_zeros(len(values), n, values.shape[-1])
    weight = torch.zeros_like(total)
    total.index_add_(1, group, torch.where(good, values, 0.))
    weight.index_add_(1, group, good.to(values))
    centres = coords.new_zeros(n, 3).index_add_(0, group, coords) / count[:, None]
    return dict(values=total / weight.clamp_min(1), valid=weight > 0,
                coverage=weight / count[None, :, None], coords=centres, times=times)


class GlobalObservationEncoder(nn.Module):
    """Learned queries read geometric observation bins independently at each time."""
    def __init__(self, tokens, width=64):
        super().__init__()
        self.input_proj = nn.Linear(2, width)
        self.position_proj = nn.Linear(48, width)
        self.queries = nn.Parameter(torch.randn(tokens, width) * .02)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, context):
        value, good = context['values'], context['valid']
        b, v, t = value.shape
        value = torch.where(good, value, 0.)
        content = torch.stack((value, context['coverage']), -1)
        position = fourier_features(context['coords'], 8, 2., 300.)
        h = self.input_proj(content) + self.position_proj(position)[None, :, None]
        h = h.permute(0, 2, 1, 3).reshape(b * t, v, -1)
        valid = good.permute(0, 2, 1).reshape(b * t, v)
        present = valid.any(-1)
        # SDPA/MHA must never see an entirely masked row; empty frames return zero.
        valid = valid.clone()
        valid[~present, 0] = True
        h = torch.where(present[:, None, None], h, 0.)
        q = self.queries.to(h)[None].expand(b * t, -1, -1)
        out, _ = self.attention(q, h, h, key_padding_mask=~valid, need_weights=False)
        out = torch.where(present[:, None, None], self.norm(out + q), 0.)
        return out.reshape(b, t, len(self.queries), -1), present.reshape(b, t)


class ReadGlobalObservations(nn.Module):
    """Each local voxel reads the global summaries at its own physical time."""
    def __init__(self, d_model, width=64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.out = nn.Linear(width, d_model, bias=False)
        nn.init.zeros_(self.out.weight)  # new path initially leaves local features intact

    def forward(self, h, summaries, present):
        b, v, t, d = h.shape
        q = self.query(self.norm(h)).permute(0, 2, 1, 3).reshape(b * t, v, -1)
        kv = summaries.reshape(b * t, summaries.shape[2], -1)
        update, _ = self.attention(q, kv, kv, need_weights=False)
        update = self.out(update).reshape(b, t, v, d).permute(0, 2, 1, 3)
        return h + update * present[:, None, :, None].to(update)
