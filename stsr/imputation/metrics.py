"""Matched-denominator baselines and temporal recovery metrics."""
import torch
from .masks import context_distances


def interpolate(values, observed, times, method='linear'):
    if method not in ('linear', 'mean_included'):
        raise ValueError('unknown interpolation baseline')
    result = torch.full_like(values, float('nan'))
    for b in range(len(values)):
        for v in range(values.shape[1]):
            ids = observed[b, v].nonzero().flatten()
            if not len(ids):
                continue
            hi = torch.searchsorted(times[ids], times).clamp_max(len(ids)-1)
            lo = (hi - 1).clamp_min(0)
            # Outside the bracket, use the single nearest observation.
            lo = torch.where(times >= times[ids[-1]], hi, lo)
            left, right = ids[lo], ids[hi]
            if method == 'linear':
                weight = ((times-times[left])/(times[right]-times[left]).clamp_min(1e-12)).clamp(0,1)
            else:
                weight = torch.full_like(times, .5)
            row = (1-weight)*values[b,v,left] + weight*values[b,v,right]
            result[b,v] = torch.where(observed[b,v], values[b,v], row)
    return result


def score(prediction, truth, target, observed, times, acquired=None):
    if not target.any() or not torch.isfinite(prediction[target]).all() or not torch.isfinite(truth[target]).all():
        raise ValueError('all methods must score the same nonempty finite target set')
    err = prediction[target]-truth[target]
    correlations=[]
    for p,y,m in zip(prediction.flatten(0,1),truth.flatten(0,1),target.flatten(0,1)):
        if m.sum()<3: continue
        p,y=p[m],y[m]; p=p-p.mean(); y=y-y.mean()
        den=p.norm()*y.norm()
        if den>1e-12: correlations.append(float((p*y).sum()/den))
    support=context_distances(observed,times)
    groups={}
    bracket=support['left_seconds']+support['right_seconds']
    one_sided = support['any_context'] & ~support['two_sided']
    for name,m in [('two_sided',support['two_sided']),('one_sided',one_sided),
                  ('leading', ~torch.isfinite(support['left_seconds']) & torch.isfinite(support['right_seconds'])),
                  ('trailing', torch.isfinite(support['left_seconds']) & ~torch.isfinite(support['right_seconds']))]:
        selected=target&m
        groups[name]={'n':int(selected.sum()),'mse':float((prediction[selected]-truth[selected]).square().mean()) if selected.any() else None}
    for lo,hi in [(0,4),(4,8),(8,16),(16,float('inf'))]:
        selected=target&support['two_sided']&(bracket>lo)&(bracket<=hi)
        groups['bracket_%s_%ss'%(lo,hi)]={'n':int(selected.sum()),'mse':float((prediction[selected]-truth[selected]).square().mean()) if selected.any() else None}
    # Actual unobserved interval containing a target, including real exclusions,
    # but excluding padding. Unlike bracket span this also measures edge gaps.
    domain = torch.ones(len(times), dtype=torch.bool, device=observed.device) if acquired is None else acquired.to(observed.device)
    missing = ~observed & domain
    index = torch.arange(len(times), device=observed.device).expand_as(observed)
    previous = torch.where(~missing, index, -1).cummax(-1).values
    following = torch.where(~missing, index, len(times)).flip(-1).cummin(-1).values.flip(-1)
    duration = (following-previous-1).clamp_min(0) * (times[1]-times[0])
    for lo, hi in [(0,2), (2,8), (8,16), (16,32), (32,float('inf'))]:
        for suffix, context in [('', torch.ones_like(target)), ('_one_sided', one_sided)]:
            selected = target & domain & context & (duration > lo) & (duration <= hi)
            groups['missing_%s_%ss%s'%(lo,hi,suffix)] = {
                'n': int(selected.sum()),
                'mse': float((prediction[selected]-truth[selected]).square().mean()) if selected.any() else None}
    return {'n':int(target.sum()),'mse':float(err.square().mean()),'mae':float(err.abs().mean()),
            'voxel_r':sum(correlations)/len(correlations) if correlations else None,
            'voxel_r_rows':len(correlations),'support':groups}
