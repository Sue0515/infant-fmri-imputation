"""Observation-only STSR encoder + INR for native-grid irregular imputation.

No HRF deconvolution, temporal decimation, or spatial SR is performed here.
The uniform acquisition lattice is retained; its observation mask is arbitrary.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TemporalImputer(nn.Module):
    def __init__(self, net, query_chunk=1024, normalization='record'):
        super().__init__()
        if not hasattr(net, 'features') or getattr(net, 'inr_head', None) is None:
            raise ValueError('STSR features() and an INR readout are required')
        if query_chunk < 1:
            raise ValueError('query_chunk must be positive')
        if (getattr(net, 'support_proj', None) is not None or
                getattr(net, 'global_encoder', None) is not None or
                getattr(net, 'functional_proj', None) is not None):
            raise ValueError('first imputation path uses local STSR without extra context/atlas inputs')
        self.net, self.query_chunk = net, int(query_chunk)
        if normalization not in ('record', 'voxel'):
            raise ValueError('normalization must be record or voxel')
        self.normalization = normalization

    def encode(self, values, observed, coords, times):
        if values.ndim != 3 or observed.shape != values.shape or observed.dtype != torch.bool:
            raise ValueError('values and boolean observed must have shape B x V x T')
        if coords.shape != (values.shape[1], 3) or times.shape != (values.shape[2],):
            raise ValueError('coordinate/time geometry does not match values')
        if not torch.isfinite(coords).all() or len(times) < 2 or not torch.isfinite(times).all():
            raise ValueError('finite geometry and at least two acquisition times required')
        dt = times.diff()
        if (dt <= 0).any() or not torch.allclose(dt, dt[0].expand_as(dt), rtol=1e-4, atol=1e-6):
            raise ValueError('retain the uniform acquisition grid; mark gaps instead of deleting times')
        if not torch.isfinite(values[observed]).all():
            raise ValueError('observed values must be finite; mark invalid samples false')
        if (observed.sum((1, 2)) == 0).any():
            raise ValueError('each sample needs at least one observation')
        # Never multiply NaNs by zero, nor use hidden values in normalization.
        safe = torch.where(observed, values, 0.).float()
        axes = (1, 2) if self.normalization == 'record' else (2,)
        count = observed.sum(axes, keepdim=True).clamp_min(1)
        mean = safe.sum(axes, keepdim=True) / count
        var = torch.where(observed, (safe - mean).square(), 0.).sum(axes, keepdim=True) / count
        scale = var.clamp_min(1e-8).sqrt()
        condition = torch.where(observed, (safe - mean) / scale, 0.)
        features = self.net.features(condition, condition, observed.to(condition), coords, times, None)
        return dict(features=features, mean=mean, scale=scale, safe=safe,
                    observed=observed, coords=coords, times=times)

    def query_statistics(self, encoded, vertices):
        if self.normalization == 'voxel':
            return encoded['mean'][:, vertices, 0], encoded['scale'][:, vertices, 0]
        return encoded['mean'].flatten(1), encoded['scale'].flatten(1)

    def query(self, encoded, vertices, query_times, preserve_observed=True):
        """B x Q prediction at requested vertices/times within the scan.

        Non-native queries are estimates without measured fine-time targets.
        Known samples are copied only for exact native-grid queries.
        """
        coords, times = encoded['coords'], encoded['times']
        vertices = torch.as_tensor(vertices, device=coords.device)
        if vertices.is_floating_point() and (not torch.isfinite(vertices).all() or (vertices != vertices.round()).any()):
            raise ValueError('query vertices must be integer indices')
        vertices = vertices.long()
        query_times = torch.as_tensor(query_times, device=times.device, dtype=times.dtype)
        if vertices.ndim != 1 or query_times.shape != vertices.shape or not len(vertices):
            raise ValueError('nonempty, equally sized vertex and time vectors required')
        if (vertices < 0).any() or (vertices >= len(coords)).any():
            raise ValueError('query vertex is outside the supplied geometry')
        if (not torch.isfinite(query_times).all() or (query_times < times[0]).any()
                or (query_times > times[-1]).any()):
            raise ValueError('query times must lie within the acquisition interval')
        predictions = []
        for start in range(0, len(vertices), self.query_chunk):
            v, q = vertices[start:start+self.query_chunk], query_times[start:start+self.query_chunk]
            def read(f, v=v, q=q):
                return self.net.field(f, coords, times, None, coords[v], q)
            if self.training and torch.is_grad_enabled():
                predictions.append(checkpoint(read, encoded['features'], use_reentrant=False))
            else:
                predictions.append(read(encoded['features']))
        prediction = torch.cat(predictions, -1).float()
        mean, scale = self.query_statistics(encoded, vertices)
        prediction = prediction * scale + mean
        if preserve_observed:
            idx = torch.searchsorted(times, query_times).clamp_max(len(times)-1)
            exact = query_times == times[idx]
            copy = encoded['observed'][:, vertices, idx] & exact[None]
            prediction = torch.where(copy, encoded['safe'][:, vertices, idx], prediction)
        return prediction

    def forward(self, values, observed, coords, times):
        encoded = self.encode(values, observed, coords, times)
        vertices = torch.arange(len(coords), device=coords.device).repeat_interleave(len(times))
        return self.query(encoded, vertices, times.repeat(len(coords))).reshape_as(values)

    def loss(self, values, condition, target, coords, times):
        if (condition.dtype != torch.bool or condition.shape != values.shape or
                target.dtype != torch.bool or target.shape != values.shape or (condition & target).any()):
            raise ValueError('condition and target must be disjoint boolean masks of value shape')
        if not target.any() or not torch.isfinite(values[target]).all():
            raise ValueError('a nonempty finite observed holdout target is required')
        encoded = self.encode(values, condition, coords, times)
        # Only decode query locations scored by at least one item in the batch.
        pairs = target.any(0).nonzero()
        prediction = self.query(encoded, pairs[:, 0], times[pairs[:, 1]], preserve_observed=False)
        truth = values[:, pairs[:, 0], pairs[:, 1]]
        score = target[:, pairs[:, 0], pairs[:, 1]]
        error = torch.where(score, prediction - torch.where(score, truth, 0.), 0.)
        # Observation-only scales: held-out amplitudes cannot affect the input.
        _, scale = self.query_statistics(encoded, pairs[:, 0])
        normal_error = error / scale
        loss = normal_error.square().sum() / score.sum()
        return loss, {'loss': float(loss.detach()), 'target_values': int(score.sum())}
