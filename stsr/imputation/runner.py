"""Small full-run/ROI pilot, reusing the STSR trunk and INR readout."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from stsr.model.spacetime_attention import SpaceTimeAttentionNet
from .data import check_evaluation_split, load_record, manifest, sha256
from .masks import InsufficientContext, context_distances, draw_holdout
from .metrics import interpolate, score
from .model import TemporalImputer
from .profiles import acquisition_length, describe, draw_profile, draw_stress, fit_profile


def build_model(config, device):
    return TemporalImputer(SpaceTimeAttentionNet(**config['model']),
                           config.get('query_chunk', 1024)).to(device)


def load_rows(rows, config, device):
    return [(r, load_record(r['path'], **config.get('limits', {})).to(device)) for r in rows]


def profile_from_records(records):
    metadata = []
    for row, record in records:
        domain = None if record.acquired is None else record.acquired.cpu().numpy()
        n = acquisition_length(domain, len(record.times))
        metadata.append(dict(id=row['id'], family=row['family'], split=row['split'],
                             tr_seconds=float(record.times.diff()[0]),
                             recoverable=record.recoverable.any(0)[:n].cpu().tolist()))
    return fit_profile(metadata)


def sample_mask(row, record, config, seed, profile=None, empirical=None):
    strategy = config.get('mask_strategy', 'legacy')
    if strategy == 'fitted':
        if profile is None:
            raise ValueError('fitted strategy requires the checkpoint training mask profile')
        return draw_profile(record.valid, record.times, profile, seed=seed, acquired=record.acquired,
                            exclude_family=row['family'], min_observations=config['mask'].get('min_observations', 2))
    if strategy != 'legacy':
        raise ValueError('mask_strategy must be legacy or fitted')
    observed, target = draw_holdout(record.valid, seed=seed, empirical=empirical, **config['mask'])
    domain = None if record.acquired is None else record.acquired.cpu().numpy()
    n = acquisition_length(domain, len(record.times))
    return observed, target, dict(strategy='legacy', realized=describe(target.any(0)[:n].cpu().numpy(),
                                                                      float(record.times.diff()[0])))


def case_metrics(model, record, observed, target):
    x, observed, target = record.values[None], observed[None], target[None]
    predictions = {'stsr': model(x, observed, record.coords, record.times)}
    for method in ('linear', 'mean_included'):
        predictions[method] = interpolate(x, observed, record.times, method)
    return {method: score(p, x, target, observed, record.times, acquired=record.acquired)
            for method, p in predictions.items()}


def stress_evaluation(model, records, config):
    results, unsupported = [], []
    spec = config['stress_evaluation']
    with torch.no_grad():
        for i, (row, record) in enumerate(records):
            for seconds in spec['seconds']:
                for location in ('interior', 'leading', 'trailing'):
                    label = f'{seconds}s_{location}'
                    try:
                        observed, target, info = draw_stress(record.valid, record.times, acquired=record.acquired,
                                                             seconds=seconds, location=location, seed=config['seed']+200000+i,
                                                             min_observations=config['mask'].get('min_observations', 2))
                    except InsufficientContext as exc:
                        unsupported.append(dict(id=row['id'], case=label, reason=str(exc)))
                        continue
                    results.append(dict(id=row['id'], case=label, mask=info,
                                        metrics=case_metrics(model, record, observed, target)))
    summary = {}
    for seconds in spec['seconds']:
        for location in ('interior', 'leading', 'trailing'):
            label = f'{seconds}s_{location}'
            entries = [r for r in results if r['case'] == label]
            summary[label] = dict(scored_records=len(entries), unsupported_records=len(records)-len(entries), methods={})
            for method in ('stsr', 'linear', 'mean_included'):
                n = sum(r['metrics'][method]['n'] for r in entries)
                summary[label]['methods'][method] = dict(n=n, mse=sum(r['metrics'][method]['mse']*r['metrics'][method]['n']
                                                                                    for r in entries)/n if n else None)
    return dict(summary=summary, records=results, unsupported=unsupported,
                scope='Controlled sensitivity tests, separate from checkpoint selection. Score only known values within the requested gap.')


def evaluate(model, records, config):
    model.eval()
    eval_config = dict(config, mask_strategy=config.get('evaluation_mask_strategy', config.get('mask_strategy', 'legacy')))
    results = []
    with torch.no_grad():
        for i, (row, record) in enumerate(records):
            for repeat in range(config.get('eval_masks', 3)):
                # Fixed across epochs/methods; no empirical patterns from held-out families.
                seed = config['seed'] + 100000 + i * 1000 + repeat
                try:
                    observed, target, mask_info = sample_mask(row, record, eval_config, seed,
                                                              profile=getattr(model, 'mask_profile', None))
                except InsufficientContext as exc:
                    raise ValueError('cannot evaluate record ' + row['id']) from exc
                results.append({'id': row['id'], 'seed': seed, 'mask': mask_info,
                                'metrics': case_metrics(model, record, observed, target)})
    if not results:
        raise ValueError('empty evaluation')
    summary = {}
    for method in ('stsr', 'linear', 'mean_included'):
        entries = [r['metrics'][method] for r in results]
        n = sum(e['n'] for e in entries)
        summary[method] = {'n': n, 'mse': sum(e['mse'] * e['n'] for e in entries) / n,
                           'mae': sum(e['mae'] * e['n'] for e in entries) / n}
        support = {}
        for group in entries[0]['support']:
            groups = [e['support'][group] for e in entries]
            count = sum(g['n'] for g in groups)
            support[group] = dict(n=count, mse=sum(g['mse']*g['n'] for g in groups if g['n'])/count if count else None)
        summary[method]['support'] = support
    result = {'summary': summary, 'records': results}
    if config.get('stress_evaluation'):
        result['stress'] = stress_evaluation(model, records, config)
    return result


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def train(manifest_path, config, output, device='cpu'):
    if config['steps'] < 1 or config['eval_every'] < 1 or config.get('eval_masks', 3) < 1:
        raise ValueError('positive steps/evaluation counts required')
    rows = manifest(manifest_path)
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    if not train_rows or not val_rows:
        raise ValueError('family-separated train and val records required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config['seed'])
    records = load_rows(train_rows, config, device)
    validation = load_rows(val_rows, config, device)
    # Only training-family missingness contributes to the empirical bank.
    empirical = [r.recoverable.any(0).cpu() for _, r in records]
    needs_profile = config.get('mask_strategy') == 'fitted' or config.get('evaluation_mask_strategy') == 'fitted'
    profile = profile_from_records(records) if needs_profile else None
    if config.get('mask_strategy') == 'fitted' and len({t['family'] for t in profile['templates'] if t['gaps']}) < 2:
        raise ValueError('fitted training requires at least two nonempty donor families')
    model = build_model(config, device)
    model.mask_profile = profile
    if model.mask_profile is not None:
        write_json(output / 'mask_profile.json', model.mask_profile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    source_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(source_root), 'status', '--porcelain'], text=True,
                                             stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = 'unknown', True
    provenance = {'config': config, 'manifest': rows, 'manifest_sha256': sha256(manifest_path),
                  'input_sha256': {r['id']: sha256(r['path']) for r in train_rows + val_rows},
                  'base_revision': revision, 'dirty': dirty, 'mask_profile': model.mask_profile,
                  'empirical_families': sorted({r['family'] for r in train_rows}),
                  'source_sha256': {str(p.relative_to(Path(__file__).resolve().parents[2])): sha256(p)
                                    for p in Path(__file__).resolve().parents[1].rglob('*.py')}}
    write_json(output / 'provenance.json', provenance)
    best, history = float('inf'), []
    generator = torch.Generator().manual_seed(config['seed'])
    for step in range(1, config['steps'] + 1):
        model.train()
        skipped = []
        # Retry another training record, but fail explicitly if none can supply targets.
        for index in torch.randperm(len(records), generator=generator).tolist():
            row, r = records[index]
            try:
                observed, target, mask_info = sample_mask(row, r, config, config['seed'] + step * 1000 + index,
                                                          profile=model.mask_profile, empirical=empirical)
                break
            except InsufficientContext:
                skipped.append(row['id'])
                continue
        else:
            raise ValueError('no training record has sufficient clean context/targets')
        optimizer.zero_grad(set_to_none=True)
        loss, stats = model.loss(r.values[None], observed[None], target[None], r.coords, r.times)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        with (output / 'mask_audit.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step, id=row['id'], skipped_records=skipped,
                                         target_values=int(target.sum()), mask=mask_info), allow_nan=False)+'\n')
        if step == 1 or step % config['eval_every'] == 0 or step == config['steps']:
            result = evaluate(model, validation, config)
            current = result['summary']['stsr']['mse']
            history.append({'step': step, 'train': stats, 'validation': result['summary']})
            if current < best:
                best = current
                torch.save({'model': model.state_dict(), 'provenance': provenance,
                            'step': step, 'validation_mse': best}, output / 'best.pt')
                write_json(output / 'validation.json', result)
            write_json(output / 'history.json', history)
            print(json.dumps(history[-1]), flush=True)
    return history


def restore(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = saved['provenance']['config']
    model = build_model(config, device)
    model.load_state_dict(saved['model'])
    model.mask_profile = saved['provenance'].get('mask_profile')
    model.eval()
    return model, saved


def infer(model, record):
    """Fill only explicitly recoverable slots with same-voxel observed support."""
    observed = record.valid[None]
    eligible = record.recoverable & (record.valid.sum(-1, keepdim=True) >= 2)
    with torch.no_grad():
        prediction = model(record.values[None], observed, record.coords, record.times)[0]
    filled = torch.where(record.valid, record.values,
                         torch.where(eligible, prediction, float('nan')))
    support = context_distances(record.valid, record.times)
    return {'values': filled.cpu().numpy(), 'valid': record.valid.cpu().numpy(),
            'imputed': eligible.cpu().numpy(), 'recoverable': record.recoverable.cpu().numpy(),
            'coords_mm': record.coords.cpu().numpy(), 'times_s': record.times.cpu().numpy(),
            'padding': np.zeros(len(record.times), dtype=bool) if record.acquired is None else (~record.acquired).cpu().numpy(),
            **{k: v.cpu().numpy() for k, v in support.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('train')
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('evaluate')
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('infer')
    p.add_argument('--record', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    for p in sub.choices.values():
        p.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.action == 'train':
        train(args.manifest, yaml.safe_load(Path(args.config).read_text()), args.out, args.device)
        return
    if Path(args.out).exists():
        raise ValueError('output exists: ' + args.out)
    model, saved = restore(args.checkpoint, args.device)
    config = saved['provenance']['config']
    if args.action == 'evaluate':
        rows = check_evaluation_split(manifest(args.manifest), saved['provenance']['manifest'], args.split)
        result = evaluate(model, load_rows(rows, config, args.device), config)
        result.update(checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest),
                      input_sha256={r['id']: sha256(r['path']) for r in rows}, split=args.split)
        write_json(args.out, result)
    else:
        record = load_record(args.record, **config.get('limits', {})).to(args.device)
        result = infer(model, record)
        # Use a file handle so numpy does not silently append a different suffix.
        with open(args.out, 'xb') as stream:
            np.savez_compressed(stream, **result, checkpoint_sha256=sha256(args.checkpoint),
                                input_sha256=sha256(args.record))


if __name__ == '__main__':
    main()
