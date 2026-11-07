"""JEPA-style auxiliary fine-support feature prediction for deterministic STSR.

The student receives ONLY degraded observations. An EMA teacher sees the native
training target under no_grad; it is never an inference input. This is an auxiliary
objective beside BOLD reconstruction, not a reproduction of Brain-JEPA.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .model.observation_context import support_vector

JEPA_DEFAULTS = dict(jepa_aux=False, jepa_hidden=384, jepa_weight=.1,
    jepa_warmup_updates=1000, jepa_queries=256, jepa_ema_start=.996,
    jepa_ema_end=1., jepa_ema_updates=20000, jepa_variance_weight=0.,
    jepa_variance_floor=.5)
JEPA_LOSS_KEYS = tuple(k for k in JEPA_DEFAULTS if k not in ('jepa_aux','jepa_hidden'))


class JepaPredictor(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim+4,hidden),nn.GELU(),nn.Linear(hidden,dim))

    def forward(self, features, geom, indices):
        b,v,t=indices
        parent=geom.group[v]
        query=geom.times[t]
        right=torch.searchsorted(geom.coarse_times,query).clamp(1,len(geom.coarse_times)-1)
        left=right-1
        fraction=(query-geom.coarse_times[left])/(geom.coarse_times[right]-geom.coarse_times[left])
        context=torch.lerp(features[b,parent,left],features[b,parent,right],fraction[:,None].to(features))
        relative=(geom.coords[v]-geom.coarse_coords[parent])@torch.linalg.inv(geom.coarse_basis).T
        offset=torch.cat([relative,fraction[:,None]],-1).to(context)
        return self.net(torch.cat([context,offset],-1))


class JepaAuxiliary:
    """Teacher deliberately lives outside the trainable nn.Module hierarchy."""
    def __init__(self, student, options):
        self.options={k:options.get(k,v) for k,v in JEPA_DEFAULTS.items()}
        o=self.options
        if not (0 <= o['jepa_ema_start'] <= o['jepa_ema_end'] <= 1):
            raise ValueError('JEPA EMA must satisfy 0 <= start <= end <= 1')
        if any(not math.isfinite(float(v)) for v in o.values()):
            raise ValueError('JEPA options must be finite')
        if min(o['jepa_queries'],o['jepa_hidden'],o['jepa_ema_updates']) < 1:
            raise ValueError('JEPA query/hidden/EMA counts must be positive')
        if min(o['jepa_weight'],o['jepa_variance_weight'],o['jepa_warmup_updates']) < 0 or o['jepa_variance_floor'] <= 0:
            raise ValueError('invalid JEPA loss weights/warmup/variance floor')
        self.teacher=copy.deepcopy(student)
        self.teacher.jepa_predictor=None
        self.teacher.eval().requires_grad_(False)
        self.updates=0
        self.query_draws=0

    def select(self, score):
        # A private CPU generator preserves the BOLD augmentation/noise stream.
        gen=torch.Generator().manual_seed(7919+self.query_draws)
        self.query_draws+=1
        rows=[]
        for b in range(len(score)):
            positions=score[b].nonzero()
            select=torch.randperm(len(positions),generator=gen)[:self.options['jepa_queries']].to(positions.device)
            vt=positions[select]
            rows.append(torch.cat([torch.full_like(vt[:,:1],b),vt],-1))
        return torch.cat(rows).T

    @property
    def weight(self):
        warmup=self.options['jepa_warmup_updates']
        return self.options['jepa_weight']*min(1.,(self.updates+1)/max(1,warmup))

    @property
    def momentum(self):
        fraction=min(1.,self.updates/self.options['jepa_ema_updates'])
        return self.options['jepa_ema_start']+fraction*(self.options['jepa_ema_end']-self.options['jepa_ema_start'])

    @torch.no_grad()
    def update(self, student, *, successful=True):
        if not successful:
            return
        momentum=self.momentum
        live=dict(student.named_parameters())
        for name,param in self.teacher.named_parameters():
            param.lerp_(live[name].detach(),1.-momentum)
        buffers=dict(student.named_buffers())
        for name,value in self.teacher.named_buffers():
            value.copy_(buffers[name])
        self.updates+=1
        self.teacher.eval()

    def checkpoint(self):
        return dict(schema='jepa_aux_v1',teacher=self.teacher.state_dict(),updates=self.updates,
                    query_draws=self.query_draws,options=self.options)

    def restore(self, saved):
        if not saved or saved.get('schema')!='jepa_aux_v1' or saved.get('options')!=self.options:
            raise ValueError('missing or incompatible JEPA teacher state on resume')
        if saved['updates']<0 or saved['query_draws']<0:
            raise ValueError('invalid JEPA counters')
        self.teacher.load_state_dict(saved['teacher'],strict=True)
        self.teacher.eval().requires_grad_(False)
        self.updates=int(saved['updates']);self.query_draws=int(saved['query_draws'])

    def loss(self, objective, features, prepared):
        g=prepared['geom'];indices=prepared['jepa_indices'];b,v,t=indices
        dim=features.shape[-1]
        token_count=prepared.get('jepa_denominator')
        sample_count=prepared.get('jepa_sample_denominator')
        if token_count is None:
            token_count=objective.denom_fn(features.new_tensor(len(b)*dim))
        if sample_count is None:
            counts=torch.bincount(b,minlength=len(features))
            sample_count=objective.denom_fn((counts>=2).sum().float())
        if not len(b):
            # Keep all predictor parameters inside DDP's graph even on an empty
            # rank. The pooled group, not each rank, determines target validity.
            zero=sum(p.sum()*0 for p in objective.net.jepa_predictor.parameters())
            return zero,dict(loss_jepa=0.,loss_jepa_variance=0.,jepa_target_std=0.,jepa_prediction_std=0.,
                jepa_queries=0.,jepa_weight=self.weight,jepa_ema=self.momentum)
        with objective.amp.context():
            prediction=objective.net.jepa_predictor(features,g,indices).float()
        with torch.no_grad(),objective.amp.context():
            teacher=self.teacher
            self.teacher.eval()
            clean=prepared['clean'];valid=prepared['jepa_native_valid']
            latent=teacher.features(clean,clean,valid.to(clean),g.coords,g.times,None,
                observation_support=support_vector(g.basis,g.times),
                global_context=prepared.get('jepa_teacher_context'),
                functional_coordinates=(objective.functional_atlas.support_features(g.coords,g.basis)
                    if objective.functional_atlas is not None else None))
            target=F.layer_norm(latent[b,v,t].float(),(dim,)).detach()
        embedding=F.smooth_l1_loss(prediction,target,reduction='sum')/token_count.clamp_min(1.)
        # Optional per-sample feature variance floor, distinct from core JEPA.
        # This decomposes across accumulated microbatches; no batch-statistic
        # differences are introduced by an unequal spatial crop size.
        variance=prediction.sum()*0
        for i in b.unique():
            value=prediction[b==i]
            if len(value)>=2:
                std=(value.var(0,unbiased=False)+1e-4).sqrt()
                variance=variance+F.relu(self.options['jepa_variance_floor']-std).mean()
        variance=variance/sample_count.clamp_min(1.)
        loss=self.weight*(embedding+self.options['jepa_variance_weight']*variance)
        return loss,dict(loss_jepa=float(embedding.detach()),loss_jepa_variance=float(variance.detach()),
            jepa_target_std=float(target.std(0,unbiased=False).mean()),
            jepa_prediction_std=float(prediction.detach().std(0,unbiased=False).mean()),
            jepa_queries=float(len(b)),jepa_weight=self.weight,jepa_ema=self.momentum)
