"""Raw-acquisition BOLD imputation cohort, without movie/gaze conditioning.

This exploratory protocol does not claim recovered original burn-in metadata,
verified movie alignment, clean gaze, or equivalence to the released censoring.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import nibabel as nib
import numpy as np
from scipy import ndimage

from .data import load_record, manifest, sha256
from .prepare_dryad import member
from .prepare_raw import raw_geometry
from .raw_geometry import raw_to_mask_voxels, resample_mask
from .raw_spatial import SpatialConfig, register_frame, image_from_array, motion_summary

SOURCE_KIND='raw_framewise_qc_v1'
PROTOCOL=dict(version=1, reference_frame=10, analysis_start_frame=11, max_frames=256,
    max_voxels=256, min_voxels=32, min_valid_frames=30, min_valid_fraction=.4,
    max_rms_step_mm=1., max_rms_from_reference_mm=3., min_correlation=.8,
    max_fractional_dvars=.10, min_frame_coverage=.98,
    gaze_policy='unconditioned_unknown', timing_policy='raw_acquisition_only',
    prefix_policy='exclude first 11 raw frames; not a reconstruction of original burn-in',
    roi_policy='released anatomical occipital ROI projected by frozen published spatial transform; nearest neighbor > 0.5')


def write_json(path,data):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def spatially_spread(ijk,count):
    ijk=np.asarray(ijk)
    if len(ijk)<=count: return ijk
    chosen=[int(np.argmin(np.sum((ijk-ijk.mean(0))**2,axis=1)))]
    distance=np.full(len(ijk),np.inf)
    for _ in range(count-1):
        distance=np.minimum(distance,np.sum((ijk-ijk[chosen[-1]])**2,axis=1))
        chosen.append(int(np.argmax(distance)))
    return ijk[chosen]


def reference_support(reference):
    if reference.ndim!=3 or not np.isfinite(reference).all():
        raise ValueError('finite 3D reference required')
    positive=reference[reference>0]
    if not len(positive): raise ValueError('empty reference')
    support=reference>np.percentile(positive,35)
    support=ndimage.binary_fill_holes(ndimage.binary_closing(support,iterations=1))
    components,_=ndimage.label(support)
    counts=np.bincount(components.ravel()); counts[0]=0
    if not counts.any(): raise ValueError('empty reference support after morphology')
    return components==int(counts.argmax())


def classify_frames(values,reference_values,coverage,step,absolute,correlation,failed):
    """QC is frozen before artificial holdouts; amplitudes are never filtered."""
    floor=max(float(np.percentile(np.abs(reference_values),10)),1e-6)
    scale=np.maximum(np.abs(reference_values),floor)
    sequence=np.column_stack([reference_values,values])
    diffs=np.diff(sequence,axis=1)/scale[:,None]
    dvars=np.sqrt(np.mean(np.where(np.isfinite(diffs),diffs,0.)**2,axis=0))
    # Missing/failed frames make the next temporal-difference QC unknown as well.
    dvars_known=np.isfinite(sequence[:,:-1]).all(0)&np.isfinite(sequence[:,1:]).all(0)
    motion=(step>PROTOCOL['max_rms_step_mm'])|(absolute>PROTOCOL['max_rms_from_reference_mm'])
    signal=(failed|~dvars_known|~np.isfinite(correlation)|~np.isfinite(step)|~np.isfinite(absolute)|
            (dvars>PROTOCOL['max_fractional_dvars'])|
            (correlation<PROTOCOL['min_correlation'])|
            (coverage.mean(0)<PROTOCOL['min_frame_coverage']))
    clean=~motion&~signal
    return clean,motion,signal,dvars


def prepare_one(task):
    row,support_root,data_root,output=task
    output=Path(output); source=Path(support_root)/row['receipt']
    original=json.loads(source.read_text()); movie,ppt=row['id'].split(':')
    for item in original['sources']:
        path=source.parent/item['local']
        if path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']:
            raise ValueError('source support integrity mismatch: '+str(path))
    folder=output/movie/ppt; folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/'qc.json'
    if receipt_path.exists():
        previous=json.loads(receipt_path.read_text())
        if previous.get('status')=='prepared' and sha256(folder/'record.npz')==previous['record_sha256']:
            return previous
        raise ValueError('incomplete or excluded output exists; inspect before retry: '+str(folder))
    record=dict(id=row['id'],family=original['family'],split=row['split'],source_receipt=str(source),
                source_receipt_sha256=sha256(source),protocol=PROTOCOL,status='preparing')
    if len(original['raw_geometry'])!=1: raise ValueError('one raw run required')
    raw_name=original['raw_geometry'][0]['raw_member']
    spatial=SpatialConfig(iterations=100,threads=2)
    record['spatial_config']=asdict(spatial)
    with zipfile.ZipFile(Path(data_root)/(movie+'.zip')) as archive, tempfile.TemporaryDirectory(prefix='infant-qc-') as scratch:
        destination=Path(scratch)/'raw.nii'; digest=hashlib.sha256()
        with member(archive,raw_name) as stream, gzip.GzipFile(fileobj=stream) as expanded, destination.open('wb') as target:
            while True:
                block=expanded.read(1024*1024)
                if not block: break
                target.write(block); digest.update(block)
        image=nib.load(destination); tr=raw_geometry(image)
        start=PROTOCOL['analysis_start_frame']; stop=min(image.shape[-1],start+PROTOCOL['max_frames'])
        if stop-start<PROTOCOL['min_valid_frames']: raise ValueError('run too short after fixed prefix exclusion')
        reference=np.asarray(image.dataobj[...,PROTOCOL['reference_frame']],dtype=np.float32)
        fixed_support=reference_support(reference)
        roi_image=nib.load(source.parent/'released_occipital.nii.gz')
        anatomical=original['anatomical_header']
        mapping=raw_to_mask_voxels(image.shape[:3],image.affine,anatomical['shape'],np.array(anatomical['affine']),
                    roi_image.affine,np.loadtxt(source.parent/'released_raw_to_highres_0.mat'))
        roi=resample_mask(roi_image.get_fdata(dtype=np.float32),image.shape[:3],mapping)
        if roi.sum()<PROTOCOL['min_voxels']: raise ValueError('too few projected occipital voxels')
        roi_support=float(fixed_support[roi].mean())
        # Eroded anatomical interiors reduce partial-volume boundary sensitivity.
        interior=ndimage.binary_erosion(roi)&fixed_support
        if interior.sum()<PROTOCOL['min_voxels']: raise ValueError('too few ROI interior voxels inside reference support')
        if roi_support<.6: raise ValueError('projected ROI/reference intensity support mismatch')
        ijk=spatially_spread(np.argwhere(interior),PROTOCOL['max_voxels'])
        ref_values=reference[tuple(ijk.T)]
        values=np.full((len(ijk),stop-start),np.nan,dtype=np.float32)
        covered=np.zeros(values.shape,dtype=bool)
        matrices=[]; registration=[]; correlations=[]; failed=[]
        for j,index in enumerate(range(start,stop)):
            frame=np.asarray(image.dataobj[...,index],dtype=np.float32)
            try:
                aligned,coverage,details=register_frame(frame,reference,image.affine,fixed_support,spatial)
                values[:,j]=aligned[tuple(ijk.T)]; covered[:,j]=coverage[tuple(ijk.T)]
                region=fixed_support&coverage
                a,b=aligned[region],reference[region]
                a=a-a.mean(); b=b-b.mean()
                corr=float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12))
                if not np.isfinite(corr): raise ValueError('nonfinite reference correlation')
                matrices.append(details['fixed_to_moving_lps'])
                failed.append(False); correlations.append(corr)
                registration.append(dict(raw_frame=index,**details))
            except (RuntimeError,ValueError) as exc:
                values[:,j]=np.nan; covered[:,j]=False
                matrices.append(np.eye(4).tolist()); failed.append(True); correlations.append(0.)
                registration.append(dict(raw_frame=index,error=str(exc)))
            if (j+1)%10==0 or index==stop-1:
                write_json(folder/'progress.json',dict(id=row['id'],completed=j+1,total=stop-start))
        matrices=np.asarray(matrices)
        fixed=image_from_array(reference,image.affine)
        center=fixed.TransformContinuousIndexToPhysicalPoint(tuple((np.array(image.shape[:3])-1)/2))
        step=motion_summary(np.concatenate([np.eye(4)[None],matrices]),center,spatial)['rms_step_mm'][1:]
        points=np.vstack([np.eye(3),-np.eye(3)])*spatial.rotation_radius_mm+center
        homogeneous=np.c_[points,np.ones(6)]
        moved=np.einsum('tij,pj->tpi',matrices,homogeneous)[...,:3]
        absolute=np.sqrt(np.mean(np.sum((moved-points)**2,axis=-1),axis=-1))
        clean,motion,signal,dvars=classify_frames(values,ref_values,covered,step,absolute,np.array(correlations),np.array(failed))
        valid=np.broadcast_to(clean,values.shape).copy()&covered&np.isfinite(values)
        recoverable=np.broadcast_to(motion,values.shape).copy()
        # Only frames with a finite registration can receive a motion label.
        unknown_motion=np.array(failed)|np.r_[False,np.array(failed)[:-1]]
        recoverable[:,unknown_motion]=False
        record.update(raw_member=raw_name,raw_uncompressed_sha256=digest.hexdigest(),
            raw_member_crc32=archive.getinfo(raw_name).CRC,raw_shape=list(image.shape),tr_seconds=tr,
            analysis_start_frame=start,analysis_stop_frame=stop,roi_support_fraction=roi_support,
            selected_voxels=len(ijk),projected_roi_voxels=int(roi.sum()),
            valid_frames=int(clean.sum()),analysis_frames=len(clean),motion_frames=int(motion.sum()),
            signal_qc_frames=int(signal.sum()),failed_registration_frames=int(sum(failed)),
            registration=registration,rms_step_mm=step.tolist(),rms_from_reference_mm=absolute.tolist(),
            fractional_dvars=dvars.tolist(),reference_correlation=correlations,
            raw_to_roi_voxels=mapping.tolist())
        np.save(folder/'roi_voxels.npy',ijk); np.save(folder/'registration_mask.npy',fixed_support)
        for name,array in [('reference_frame10',reference),('occipital_candidate',roi.astype(np.uint8))]:
            derivative=nib.Nifti1Image(array,image.affine)
            derivative.header.set_xyzt_units('mm')
            nib.save(derivative,folder/(name+'.nii.gz'))
        if clean.sum()<PROTOCOL['min_valid_frames'] or clean.mean()<PROTOCOL['min_valid_fraction']:
            record['status']='excluded_insufficient_qc_observations'; write_json(receipt_path,record); return record
        receipt=dict(source_kind=SOURCE_KIND,temporal_operations=[],reference_frame=10,reference_outside_analysis=True,
            timing_verified=False,timing_policy='raw_acquisition_only',gaze_policy='unconditioned_unknown',
            qc_protocol=PROTOCOL,source_receipt_sha256=sha256(source),spatial_config=asdict(spatial),
            scope='Exploratory motion-censored raw BOLD imputation; no movie decoding or gaze-conditioned claim',
            qc_frozen_before_artificial_holdouts=True,spatial_metadata='frozen released anatomical mask and affine',
            limitations='Spatial alignment and QC are automated; whole-cohort expert anatomical review remains required.')
        indices=np.arange(start,stop,dtype=np.int64)
        path=folder/'record.npz'
        np.savez_compressed(path,values=np.where(valid,values,np.nan),valid=valid,recoverable=recoverable,
            coords_mm=nib.affines.apply_affine(image.affine,ijk).astype(np.float32),times_s=(indices*tr).astype(np.float32),
            raw_frame_indices=indices,voxel_ijk=ijk,padding=np.zeros(len(indices),dtype=bool),
            source_kind=np.array(SOURCE_KIND),preparation_receipt=np.array(json.dumps(receipt,sort_keys=True)),
            gaze_known=np.zeros(len(indices),dtype=bool),motion_bad=motion,signal_qc_bad=signal,
            fixed_to_moving_lps=matrices,diagnostic_rms_step_mm=step)
        load_record(path)
        record.update(status='prepared',path=str(path.relative_to(output)),record_sha256=sha256(path))
        write_json(receipt_path,record)
        return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support',required=True); p.add_argument('--data-root',required=True)
    p.add_argument('--out',required=True); p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int); args=p.parse_args()
    support=Path(args.support); output=Path(args.out); output.mkdir(parents=True,exist_ok=False)
    index=json.loads((support/'index.json').read_text())
    rows=index['records_index'][:args.limit] if args.limit else index['records_index']
    write_json(output/'protocol.json',dict(protocol=PROTOCOL,split_sha256=index['split_sha256'],
                                          source_index_sha256=sha256(support/'index.json')))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(prepare_one,(row,str(support),args.data_root,str(output))):row for row in rows}
        for future in as_completed(jobs):
            row=jobs[future]
            try: result=future.result()
            except Exception as exc:
                result=dict(id=row['id'],split=row['split'],status='failed',error=str(exc))
            results.append(result)
            write_json(output/'progress.json',dict(completed=len(results),total=len(rows),
                prepared=sum(r['status']=='prepared' for r in results),last={k:result.get(k) for k in ('id','status','error')}))
            print(json.dumps({k:result.get(k) for k in ('id','status','valid_frames','analysis_frames','error')}),flush=True)
    results.sort(key=lambda r:r['id'])
    selected=[dict(id=r['id'],family=r['family'],split=r['split'],path=r['path'],source_kind=SOURCE_KIND)
              for r in results if r['status']=='prepared']
    write_json(output/'manifest.json',selected)
    if selected: manifest(output/'manifest.json')
    summary=dict(status='prepared_exploratory_cohort' if selected else 'no_eligible_records',
        prepared=len(selected),attempted=len(rows),splits={s:sum(r['split']==s for r in selected) for s in ('train','val','test')},
        original_family_split_sha256=index['split_sha256'],movie_gaze_alignment_verified=False,
        source_kind=SOURCE_KIND,records=results)
    write_json(output/'summary.json',summary)


if __name__=='__main__': main()
