"""Cohort/availability EDA and sparse-probe QC; no model fitting or split changes."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np

from .audit_masks import MOVIES, dump
from .data import sha256
from .prepare_dryad import member, read, read_roi
from .profiles import intervals


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def quantiles(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, 1)} if values else {}


def clean_windows(valid, lengths=(1, 2, 4, 8, 16)):
    valid = np.asarray(valid, dtype=bool)
    result = {}
    for length in lengths:
        starts = np.array([], dtype=int)
        if len(valid) >= length and valid.sum()-length >= 2:
            starts = np.flatnonzero(np.convolve(valid.astype(int), np.ones(length, dtype=int), 'valid') == length)
        result[str(length)] = dict(starts=int(len(starts)), leading=bool(0 in starts), trailing=bool(len(valid)-length in starts))
    return result


def enrich(records, metadata):
    rows = []
    for r in records:
        meta = metadata[(r['movie'], r['ppt'])]
        motion, eye, valid = (np.asarray(r[k], dtype=bool) for k in ('motion','eye','valid'))
        known = r['gaze_known']
        categories = dict(clean=0, motion_only=0, eye_only=0, overlap=0, gaze_unknown=0)
        if known:
            categories.update(clean=int((~motion & ~eye).sum()), motion_only=int((motion & ~eye).sum()),
                              eye_only=int((~motion & eye).sum()), overlap=int((motion & eye).sum()))
        else:
            categories['gaze_unknown'] = len(motion)
        continuous = r['movie'] != 'Child_Play'
        longest = max((b-a for a,b in intervals(valid)), default=0)*r['tr_seconds'] if known and continuous else None
        rows.append(dict(**r, subject='_'.join(r['ppt'].split('_')[:2]), session=r['ppt'],
                         age_months=number(meta.get('age')), sex=meta.get('sex','').strip() or 'unknown',
                         location=meta.get('location','').strip() or 'unknown', categories=categories,
                         usable_frames=int(valid.sum()) if known else None,
                         usable_seconds=float(valid.sum()*r['tr_seconds']) if known else None,
                         usable_fraction=float(valid.mean()) if known else None, longest_clean_seconds=longest,
                         windows=clean_windows(valid) if r['eligible'] else None))
    return rows


def summarize(rows):
    subjects, sessions = defaultdict(list), defaultdict(list)
    for r in rows:
        subjects[r['subject']].append(r)
        sessions[r['session']].append(r)
    entry_ages = [min(r['age_months'] for r in group if r['age_months'] is not None) for group in subjects.values()
                  if any(r['age_months'] is not None for r in group)]
    states = Counter()
    for r in rows:
        states.update(r['categories'])
    opportunities = {}
    eligible = [r for r in rows if r['windows'] is not None]
    for length in ('1','2','4','8','16'):
        available = [r for r in eligible if r['windows'][length]['starts']]
        opportunities[length] = dict(records=len(available), denominator_records=len(eligible),
            families=len({r['family'] for r in available}), starts=sum(r['windows'][length]['starts'] for r in available),
            leading_records=sum(r['windows'][length]['leading'] for r in available),
            trailing_records=sum(r['windows'][length]['trailing'] for r in available))
    return dict(records=len(rows), families=len({r['family'] for r in rows}), subjects=len(subjects), sessions=len(sessions),
        subjects_with_repeat_sessions=sum(len({r['session'] for r in group}) > 1 for group in subjects.values()),
        records_per_subject=dict(Counter(len(g) for g in subjects.values())),
        entry_age_months=quantiles(entry_ages), record_age_months=quantiles(r['age_months'] for r in rows),
        missing_age_records=sum(r['age_months'] is None for r in rows),
        subject_sex_counts=dict(Counter(g[0]['sex'] for g in subjects.values())),
        session_metadata_conflicts=[s for s,g in sessions.items() if len({r['age_months'] for r in g if r['age_months'] is not None}) > 1 or len({r['sex'] for r in g}) > 1],
        acquired_frames=sum(r['acquired_frames'] for r in rows), excluded_tail_frames=sum(r['stored_frames']-r['acquired_frames'] for r in rows),
        joint_state_frames=dict(states), usable_seconds=quantiles(r['usable_seconds'] for r in rows),
        usable_fraction=quantiles(r['usable_fraction'] for r in rows), longest_clean_seconds=quantiles(r['longest_clean_seconds'] for r in rows),
        missing_gaze_records=[r['id'] for r in rows if not r['gaze_known']], clean_windows=opportunities)


def select_signals(rows, per_movie):
    if per_movie < 0:
        raise ValueError('nonnegative sample count required')
    training = sorted([r for r in rows if r['eligible'] and r['split'] == 'train'], key=lambda r:r['id'])
    selected = []
    for movie in MOVIES:
        selected.extend([r for r in training if r['movie'] == movie][:per_movie])
    if per_movie and training:
        selected.extend([min(training,key=lambda r:(r['acquired_frames'],r['id'])),
                         max(training,key=lambda r:(sum(r['motion']),r['id']))])
    return list({r['id']:r for r in selected}.values())


def signal_qc(root, row):
    source = f"{row['movie']}/preprocessed_native/linear_alignment/{row['ppt']}_Z.nii.gz"
    with zipfile.ZipFile(Path(root)/(row['movie']+'.zip')) as archive:
        with member(archive,source) as raw, gzip.GzipFile(fileobj=raw) as stream:
            header = nib.Nifti1Header.from_fileobj(stream)
        shape = header.get_data_shape()
        axes = [np.unique(np.linspace(.25*(n-1), .75*(n-1), min(6,n)).round().astype(int)) for n in shape[:3]]
        ijk = np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        x, coords, times = read_roi(archive,source,ijk)
        crc = archive.getinfo(source).CRC
    if x.shape[-1] != row['stored_frames'] or not np.isclose(float(times[1]-times[0]), row['tr_seconds']):
        raise ValueError('geometry differs from the mask audit')
    n = row['acquired_frames']
    clean = x[:,:n][:,np.asarray(row['valid'],dtype=bool)]
    support = np.isfinite(clean).all(1) & (clean != 0).any(1)
    std = np.std(clean[support],axis=1,ddof=1) if clean.shape[-1] > 1 else []
    rms, delta = [], [None]
    for t in range(x.shape[-1]):
        v = x[support,t]
        v = v[np.isfinite(v)]
        rms.append(float(np.sqrt(np.mean(v.astype(float)**2))) if len(v) else None)
        if t:
            d = x[support,t]-x[support,t-1]
            d = d[np.isfinite(d)]
            delta.append(float(np.sqrt(np.mean(d.astype(float)**2))) if len(d) else None)
    return dict(id=row['id'], split=row['split'], source_member=source, member_crc32=crc, shape=list(shape),
        tr_seconds=row['tr_seconds'], acquired_frames=n, probe_voxels=len(x), nonzero_clean_probe_voxels=int(support.sum()),
        voxel_sizes_mm=[float(v) for v in header.get_zooms()[:3]], affine_mm=header.get_best_affine().tolist(),
        probe_std=quantiles(std), probe_rms=rms, probe_delta_rms=delta,
        probe_allzero_frames=np.flatnonzero(np.isfinite(x).all(0)&(x==0).all(0)).tolist(),
        probe_nonfinite_frames=np.flatnonzero(~np.isfinite(x).all(0)).tolist(),
        probe_identical_previous_frames=[t for t in range(1,x.shape[1]) if np.array_equal(x[:,t],x[:,t-1],equal_nan=True)],
        excluded_tail_probe_nonzero_frames=(np.flatnonzero((np.isfinite(x[:,n:])&(x[:,n:]!=0)).any(0))+n).tolist(),
        scope='Fixed central grid probe, at most 216 voxels, not an anatomical ROI or whole-volume QC. '
              'Support selected from clean observations for descriptive QC only. Z series tSNR is not interpreted.')


def plots(rows, signals, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x = np.arange(len(MOVIES))
    names = [m.replace('_',' ') for m in MOVIES]
    axes[0,0].bar(x-.18,[sum(r['movie']==m for r in rows) for m in MOVIES],width=.35,label='Movie records')
    axes[0,0].bar(x+.18,[len({r['subject'] for r in rows if r['movie']==m}) for m in MOVIES],width=.35,label='Unique infants')
    axes[0,0].set(xticks=x,xticklabels=names,title='Records and participants',ylabel='Count')
    axes[0,0].legend(frameon=False)
    ages = defaultdict(list)
    for r in rows:
        if r['age_months'] is not None: ages[r['subject']].append(r['age_months'])
    axes[0,1].hist([min(a) for a in ages.values()],bins=10,color='#21877f')
    axes[0,1].set(title='Youngest reported age per infant',xlabel='Months',ylabel='Unique infants')
    bottom = np.zeros(len(MOVIES))
    for key,color in [('clean','#21877f'),('motion_only','#db8040'),('eye_only','#e4b550'),('overlap','#b94e59'),('gaze_unknown','#999999')]:
        values = [100*sum(r['categories'][key] for r in rows if r['movie']==m)/sum(r['acquired_frames'] for r in rows if r['movie']==m) for m in MOVIES]
        axes[1,0].bar(x,values,bottom=bottom,label=key.replace('_',' '),color=color)
        bottom += values
    axes[1,0].set(xticks=x,xticklabels=names,title='Joint motion / gaze state',ylabel='Acquired frames (%)')
    axes[1,0].legend(frameon=False,fontsize=8,ncol=2)
    groups = [[r for r in rows if r['eligible'] and r['split']==s] for s in ('train','val','test')]
    axes[1,1].boxplot([[r['usable_seconds'] for r in g] for g in groups])
    axes[1,1].set_xticks([1,2,3],labels=[f'{s}\n{len(g)} runs / {len({r["family"] for r in g})} families' for s,g in zip(('train','val','test'),groups)])
    axes[1,1].set(title='Eligible continuous runs',ylabel='Usable sample duration (s)')
    for ax in (axes[0,0],axes[1,0]): ax.tick_params(axis='x',rotation=15)
    fig.suptitle('Infant movie fMRI — initial EDA')
    fig.supxlabel('Descriptive only. Unknown gaze is not clean data; Child Play has removed rest intervals.',fontsize=9)
    for ext in ('png','pdf'): fig.savefig(output/f'cohort_overview.{ext}',dpi=160)
    plt.close(fig)
    if signals:
        fig,axes=plt.subplots(len(signals),1,figsize=(11,2.3*len(signals)),layout='constrained',squeeze=False)
        lookup={r['id']:r for r in rows}
        for ax,qc in zip(axes[:,0],signals):
            row=lookup[qc['id']]; tr=qc['tr_seconds']; t=np.arange(len(qc['probe_rms']))*tr
            ax.plot(t,qc['probe_rms'],label='Probe RMS',linewidth=1)
            ax.plot(t,qc['probe_delta_rms'],label='Adjacent difference RMS',linewidth=1)
            for mask,color in [(row['motion'],'#db8040'),(row['eye'],'#e4b550')]:
                for a,b in intervals(mask): ax.axvspan(a*tr,b*tr,color=color,alpha=.18)
            ax.axvspan(qc['acquired_frames']*tr,len(t)*tr,color='gray',alpha=.2)
            ax.set(title=qc['id']+' (training)',ylabel='Stored signal units')
        axes[0,0].legend(frameon=False,fontsize=8)
        axes[-1,0].set_xlabel('Stored time (s); orange: motion, yellow: gaze, gray: protocol-excluded tail')
        fig.suptitle('Preprocessed native series: sparse spatial probe QC')
        for ext in ('png','pdf'): fig.savefig(output/f'signal_qc.{ext}',dpi=140)
        plt.close(fig)


def run(root, audit_dir, output, per_movie=1):
    root,audit_dir,output=Path(root),Path(audit_dir),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((audit_dir/'audit.json').read_text())
    records=json.loads((audit_dir/'records.json').read_text())
    expected={(s['archive'],s['member']):s['sha256'] for s in audit['sources']}
    metadata,sources,adult_counts={},{},{}
    for movie in MOVIES:
        with zipfile.ZipFile(root/(movie+'.zip')) as archive:
            for group in ('infant','adult'):
                name=f'{movie}/{group}_participants.csv'; raw=read(archive,name)
                digest=hashlib.sha256(raw).hexdigest(); sources[name]=digest
                if group=='infant' and digest!=expected[(movie+'.zip',name)]: raise ValueError('metadata changed since audit')
                table=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
                if group=='infant': metadata.update({(movie,r['ppt']):r for r in table})
                else: adult_counts[movie]=len(table)
    if len(records)!=len(metadata): raise ValueError('record inventory differs from audit')
    rows=enrich(records,metadata)
    result=dict(infants=summarize(rows),by_movie={m:summarize([r for r in rows if r['movie']==m]) for m in MOVIES},
        eligible_by_split={s:summarize([r for r in rows if r['eligible'] and r['split']==s]) for s in ('train','val','test')},
        adult_record_counts=adult_counts,incomplete_continuous_records=[dict(id=r['id'],frames=r['acquired_frames'],split=r['split']) for r in rows if r['movie']!='Child_Play' and r['acquired_frames']<90],
        protocol=audit['protocol'],metadata_sha256=sources,audit_sha256=sha256(audit_dir/'audit.json'),records_sha256=sha256(audit_dir/'records.json'),
        scope='All-cohort descriptive metadata; eligible training-only sparse signal QC. No model fitting or mask refitting.')
    dump(output/'summary.json',result); dump(output/'records.json',rows)
    fields=['id','subject','family','session','movie','split','eligible','age_months','sex','location','stored_frames','acquired_frames','usable_frames','usable_seconds','usable_fraction','longest_clean_seconds','gaze_known']
    with (output/'records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    selected=select_signals(rows,per_movie)
    print(json.dumps(dict(cohort=result['infants'],signal_selection=[r['id'] for r in selected])),flush=True)
    signals=[]
    for row in selected:
        signals.append(signal_qc(root,row)); dump(output/'signal_qc.json',signals)
        print(json.dumps(dict(signal=row['id'],probe_zero_frames=signals[-1]['probe_allzero_frames'],probe_nonfinite_frames=signals[-1]['probe_nonfinite_frames'])),flush=True)
    plots(rows,signals,output)
    dump(output/'completion.json',dict(complete=True,signal_records=len(signals),files={p.name:sha256(p) for p in output.iterdir() if p.is_file()}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True); p.add_argument('--audit-dir',required=True); p.add_argument('--out',required=True)
    p.add_argument('--signal-samples-per-movie',type=int,default=1)
    a=p.parse_args(); run(a.data_root,a.audit_dir,a.out,a.signal_samples_per_movie)


if __name__=='__main__': main()
