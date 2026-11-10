"""Small full-run/ROI pilot, reusing the STSR trunk and INR readout."""
import argparse
import hashlib
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
    if config.get('initialization', 'random') != 'random' or any(
            config.get(key) is not None for key in ('pretrained', 'pretrained_checkpoint', 'init_checkpoint', 'resume')):
        raise ValueError('this experiment requires random initialization; pretrained/resume weights are not accepted')
    return TemporalImputer(SpaceTimeAttentionNet(**config['model']),
                           config.get('query_chunk', 1024), config.get('normalization', 'record')).to(device)


def validate_source(path, config):
    required = config.get('required_source_kind')
    if required:
        with np.load(path, allow_pickle=False) as data:
            kind = str(data['source_kind'].item()) if 'source_kind' in data else None
            receipt = json.loads(str(data['preparation_receipt'].item())) if 'preparation_receipt' in data else {}
            if kind != required or receipt.get('source_kind') != required:
                raise ValueError('raw experiment refuses preprocessed or unverified source: '+str(path))
            if required in ('raw_framewise_qc_v1','raw_framewise_qc_v2'):
                if (receipt.get('timing_policy')!='raw_acquisition_only' or
                        receipt.get('gaze_policy')!='unconditioned_unknown' or
                        receipt.get('timing_verified') is not False or
                        receipt.get('qc_frozen_before_artificial_holdouts') is not True or
                        receipt.get('reference_outside_analysis') is not True or
                        receipt.get('temporal_operations')!=[]):
                    raise ValueError('raw QC cohort requires explicit unverified movie/gaze scope and independent spatial processing')
                if required=='raw_framewise_qc_v2':
                    reference=receipt.get('reference_frame')
                    if (receipt.get('brain_interior_only') is not True or receipt.get('native_slice_qc') is not True or
                            receipt.get('calibration_anatomy_refined') is not True or
                            receipt.get('calibration_prefix_stop')!=11 or not isinstance(reference,int) or
                            not 3<=reference<11 or not receipt.get('unique_acquisition_sha256')):
                        raise ValueError('raw QC v2 requires reserved-prefix calibration, brain interior and native-slice QC')
            elif (receipt.get('timing_verified') is not True or receipt.get('reference_outside_analysis') is not True
                    or receipt.get('temporal_operations') != []):
                raise ValueError('verified timing, held-out-independent reference and no temporal processing required')


def load_rows(rows, config, device):
    for row in rows:
        validate_source(row['path'], config)
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
    predictions = {'stsr': predict_record(model, record, observed[0])[None]}
    for method in ('linear', 'mean_included'):
        predictions[method] = interpolate(x, observed, record.times, method)
    return {method: score(p, x, target, observed, record.times, acquired=record.acquired)
            for method, p in predictions.items()}


def predict_record(model, record, observed):
    """Model only the analysis prefix and restore an excluded NaN tail."""
    view = record.analysis_view()
    n = len(view.times)
    if observed.shape != record.valid.shape or (observed & ~record.valid).any():
        raise ValueError('conditioning must be a subset of valid observations')
    prediction = model(view.values[None], observed[None, :, :n], view.coords, view.times)[0]
    return torch.nn.functional.pad(prediction, (0, len(record.times)-n), value=float('nan'))


def evaluation_seed(config, identity, case='primary', repeat=0):
    """Versioned, identity-based masks, independent of model initialization."""
    payload = json.dumps(['infant-eval-v2', config.get('evaluation_seed', 42), identity, case, repeat])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') & ((1 << 63)-1)


def stress_evaluation(model, records, config):
    results, unsupported = [], []
    spec = config['stress_evaluation']
    with torch.no_grad():
        for row, record in records:
            for seconds in spec['seconds']:
                for location in ('interior', 'leading', 'trailing'):
                    label = f'{seconds}s_{location}'
                    seed = evaluation_seed(config, row['id'], label)
                    try:
                        observed, target, info = draw_stress(record.valid, record.times, acquired=record.acquired,
                                                             seconds=seconds, location=location, seed=seed,
                                                             min_observations=config['mask'].get('min_observations', 2))
                    except InsufficientContext as exc:
                        unsupported.append(dict(id=row['id'], case=label, reason=str(exc)))
                        continue
                    results.append(dict(id=row['id'], family=row['family'], case=label, seed=seed, mask=info,
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


def family_macro_scores(results,method):
    """Equal weight per family after averaging masks and acquisitions."""
    grouped={}
    for row in results:
        grouped.setdefault(row['family'],{}).setdefault(row['id'],[]).append(row['metrics'][method]['normalized_mse'])
    families={f:sum(sum(v)/len(v) for v in records.values())/len(records) for f,records in grouped.items()}
    if not families: raise ValueError('family macro requires nonempty results')
    return sum(families.values())/len(families),families


def training_order(records,generator,sampling='record'):
    if sampling=='record': return torch.randperm(len(records),generator=generator).tolist()
    if sampling!='family': raise ValueError('training_sampling must be record or family')
    groups={}
    for i,(row,_) in enumerate(records): groups.setdefault(row['family'],[]).append(i)
    keys=sorted(groups); order=[]
    for i in torch.randperm(len(keys),generator=generator).tolist():
        members=groups[keys[i]]
        order.extend(members[j] for j in torch.randperm(len(members),generator=generator).tolist())
    return order


def evaluate(model, records, config):
    model.eval()
    eval_config = dict(config, mask_strategy=config.get('evaluation_mask_strategy', config.get('mask_strategy', 'legacy')))
    results = []
    with torch.no_grad():
        for row, record in records:
            for repeat in range(config.get('eval_masks', 3)):
                # Fixed across epochs/methods; no empirical patterns from held-out families.
                seed = evaluation_seed(config, row['id'], repeat=repeat)
                try:
                    observed, target, mask_info = sample_mask(row, record, eval_config, seed,
                                                              profile=getattr(model, 'mask_profile', None))
                except InsufficientContext as exc:
                    raise ValueError('cannot evaluate record ' + row['id']) from exc
                results.append({'id': row['id'], 'family': row['family'], 'seed': seed, 'mask': mask_info,
                                'metrics': case_metrics(model, record, observed, target)})
    if not results:
        raise ValueError('empty evaluation')
    summary = {}
    for method in ('stsr', 'linear', 'mean_included'):
        entries = [r['metrics'][method] for r in results]
        n = sum(e['n'] for e in entries)
        summary[method] = {'n': n, 'mse': sum(e['mse'] * e['n'] for e in entries) / n,
                           'mae': sum(e['mae'] * e['n'] for e in entries) / n,
                           'normalized_mse': sum(e['normalized_mse'] * e['n'] for e in entries) / n}
        support = {}
        for group in entries[0]['support']:
            groups = [e['support'][group] for e in entries]
            count = sum(g['n'] for g in groups)
            support[group] = dict(n=count, mse=sum(g['mse']*g['n'] for g in groups if g['n'])/count if count else None)
        summary[method]['support'] = support
        macro,families=family_macro_scores(results,method)
        summary[method]['family_macro_normalized_mse']=macro
        summary[method]['family_normalized_mse']=families
    result = {'summary': summary, 'records': results,
              'evaluation_protocol': dict(version=2, seed=config.get('evaluation_seed', 42),
                                          masks='sha256 of evaluation seed, record id, case and repeat')}
    if config.get('stress_evaluation'):
        result['stress'] = stress_evaluation(model, records, config)
    return result


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def validate_cohort_qa(manifest_path, rows, config):
    """Raw QC training requires a current, input-bound QA decision."""
    if config.get('required_source_kind') not in ('raw_framewise_qc_v1','raw_framewise_qc_v2'):
        return
    path = Path(manifest_path).with_name('qa_gate.json')
    if not path.exists():
        raise ValueError('raw QC training blocked: missing qa_gate.json')
    gate = json.loads(path.read_text())
    if gate.get('status') != 'approved_for_training':
        raise ValueError('raw QC training blocked by dataset QA: '+str(gate.get('blockers', [])))
    if gate.get('manifest_sha256') != sha256(manifest_path):
        raise ValueError('raw QC QA manifest hash mismatch')
    if gate.get('input_sha256') != {r['id']: sha256(r['path']) for r in rows}:
        raise ValueError('raw QC QA input hashes mismatch')
    identities = gate.get('raw_identity_sha256', {})
    if set(identities) != {r['id'] for r in rows} or len(set(identities.values())) != len(rows):
        raise ValueError('raw QC QA requires unique raw acquisition identities')
    for row in rows:
        qc_path = Path(row['path']).with_name('qc.json')
        if not qc_path.exists() or json.loads(qc_path.read_text()).get('raw_uncompressed_sha256') != identities[row['id']]:
            raise ValueError('raw QC acquisition identity does not match QC receipt')


def train(manifest_path, config, output, device='cpu'):
    if config['steps'] < 1 or config['eval_every'] < 1 or config.get('eval_masks', 3) < 1:
        raise ValueError('positive steps/evaluation counts required')
    selection_metric = config.get('selection_metric', 'mse')
    if selection_metric not in ('mse', 'normalized_mse','family_macro_normalized_mse'):
        raise ValueError('unknown checkpoint selection metric')
    rows = manifest(manifest_path)
    validate_cohort_qa(manifest_path, rows, config)
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
                  'initialization': 'random', 'pretrained_checkpoint': None,
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
        for index in training_order(records,generator,config.get('training_sampling','record')):
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
        view = r.analysis_view()
        n = len(view.times)
        loss, stats = model.loss(view.values[None], observed[None, :, :n], target[None, :, :n],
                                 view.coords, view.times)
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
            current = result['summary']['stsr'][selection_metric]
            history.append({'step': step, 'train': stats, 'validation': result['summary']})
            if current < best:
                best = current
                torch.save({'model': model.state_dict(), 'provenance': provenance,
                            'step': step, 'validation_mse': result['summary']['stsr']['mse'],
                            'selection_metric': selection_metric, 'selection_value': best}, output / 'best.pt')
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
        prediction = predict_record(model, record, observed[0])
    if not torch.isfinite(prediction[eligible]).all():
        raise ValueError('nonfinite predictions in recoverable locations')
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
        validate_source(args.record, config)
        record = load_record(args.record, **config.get('limits', {})).to(args.device)
        result = infer(model, record)
        # Use a file handle so numpy does not silently append a different suffix.
        with open(args.out, 'xb') as stream:
            np.savez_compressed(stream, **result, checkpoint_sha256=sha256(args.checkpoint),
                                input_sha256=sha256(args.record))


if __name__ == '__main__':
    main()
