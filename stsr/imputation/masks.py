"""Whole-volume holdouts on the original time grid, intersected with real validity."""
import torch


class InsufficientContext(ValueError):
    pass


def draw_holdout(valid, *, seed, gap_lengths=(1, 2, 4, 8), max_gaps=3,
                 allow_edges=False, empirical=None, min_observations=2, empirical_probability=.5):
    """Return (condition, target) for one V x T record/window.

    Empirical patterns must come from TRAINING subjects only. They describe
    temporal missingness, never BOLD values. A random contiguous segment of a
    longer pattern is used; the time axis is never compressed or wrapped.
    Existing invalid values can be in neither condition nor target.
    """
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError('valid must be a boolean V x T tensor')
    if max_gaps < 1 or min_observations < 1:
        raise ValueError('positive max_gaps and min_observations required')
    if not 0 <= empirical_probability <= 1:
        raise ValueError('empirical_probability must be within [0, 1]')
    T = valid.shape[-1]
    margin = 0 if allow_edges else 1
    lengths = [int(n) for n in gap_lengths if 0 < n < T - 2 * margin + 1]
    if any(int(n) != n or n < 1 for n in gap_lengths) or not lengths:
        raise ValueError('gap lengths must be positive integers fitting the window')
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    patterns = [] if empirical is None else list(empirical)
    for p in patterns:
        p = torch.as_tensor(p)
        if p.ndim != 1 or not ((p == 0) | (p == 1)).all():
            raise ValueError('empirical masks must be binary time vectors')
    for _ in range(128):
        hidden = torch.zeros(T, dtype=torch.bool)
        use_empirical = bool(patterns) and (bool(torch.randint(2, (), generator=g)) if empirical_probability == .5
                                            else bool(torch.rand((), generator=g) < empirical_probability))
        if use_empirical:
            p = torch.as_tensor(patterns[int(torch.randint(len(patterns), (), generator=g))]).bool().cpu()
            if len(p) >= T:
                start = int(torch.randint(len(p) - T + 1, (), generator=g))
                hidden = p[start:start + T].clone()
            else:
                start = int(torch.randint(T - len(p) + 1, (), generator=g))
                hidden[start:start + len(p)] = p
            if not allow_edges:
                hidden[0] = hidden[-1] = False
        else:
            for _ in range(1 + int(torch.randint(max_gaps, (), generator=g))):
                length = lengths[int(torch.randint(len(lengths), (), generator=g))]
                start = margin + int(torch.randint(T - 2 * margin - length + 1, (), generator=g))
                hidden[start:start + length] = True
        hidden = hidden.to(valid.device)
        condition = valid & ~hidden[None]
        target = valid & hidden[None]
        # Temporal comparisons require same-voxel evidence for every scored row.
        target &= condition.sum(-1, keepdim=True) >= min_observations
        if target.any():
            return condition, target
    raise InsufficientContext('no valid holdout with sufficient observed context')


def context_distances(observed, times):
    """Distances in seconds to real observations, including one-sided gaps."""
    if observed.shape[-1] != len(times) or observed.dtype != torch.bool:
        raise ValueError('observed mask must match times')
    t = times.expand_as(observed)
    left = torch.where(observed, t, -torch.inf).cummax(-1).values
    right = torch.where(observed, t, torch.inf).flip(-1).cummin(-1).values.flip(-1)
    return {'left_seconds': t - left, 'right_seconds': right - t,
            'two_sided': torch.isfinite(left) & torch.isfinite(right),
            'any_context': torch.isfinite(left) | torch.isfinite(right)}
