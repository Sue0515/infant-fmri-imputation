"""Unique raw acquisition preparation, reserved-prefix calibration, native QC."""
import argparse,gzip,hashlib,json,tempfile,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage
from stsr.imputation.prepare_dryad import member
from stsr.imputation.data import load_record,sha256
from stsr.imputation.prepare_raw import raw_geometry
from stsr.imputation.prepare_raw_qc import PROTOCOL as OLD,write_json,reference_support,spatially_spread,classify_frames
from stsr.imputation.raw_geometry import raw_to_mask_voxels
from stsr.imputation.raw_spatial import SpatialConfig,register_frame
from stsr.imputation.calibration import CalibrationConfig,choose_reference,slice_quality,project,anatomical_refinement,voxel_mapping

DATA=Path('/pscratch/sd/s/suincho/data/infant_movies_dryad_jm63xsjm3')
SUMMARY=DATA/'derived/raw_qc_v1_20261108/summary.json'
SOURCE_KIND='raw_framewise_qc_v2'
CAL=CalibrationConfig()
PROTOCOL=dict(OLD,version=2,reference_frame=None,calibration=asdict(CAL),
    roi_policy='eroded occipital AND eroded released anatomical brain, refined with calibration-only anatomy MI',
    recovery_policy='all QC-rejected samples with finite registration and field-of-view coverage; unknown first temporal difference excluded',
    native_slice_axis=2,initial_motion_step_known=False)

def save_image(path,values,affine):
    image=nib.Nifti1Image(values,affine); image.header.set_xyzt_units('mm'); nib.save(image,path)

def extract(z,name,path):
    with member(z,name) as src, path.open('wb') as dst:
        for b in iter(lambda:src.read(1024*1024),b''): dst.write(b)
    return sha256(path)

def prepare(task):
    old,aliases,out,calibration_only=task
    movie,ppt=old['id'].split(':'); folder=Path(out)/movie/ppt; folder.mkdir(parents=True,exist_ok=False)
    source=Path(old['source_receipt']); receipt=json.loads(source.read_text())
    record=dict(id=old['id'],family=old['family'],split=old['split'],aliases=aliases,
        raw_member=old['raw_member'],raw_uncompressed_sha256=old['raw_uncompressed_sha256'],
        source_receipt=str(source),source_receipt_sha256=sha256(source),source_kind=SOURCE_KIND,protocol=PROTOCOL)
    try:
        for item in receipt['sources']:
            if sha256(source.parent/item['local'])!=item['sha256']: raise ValueError('support hash mismatch')
        with tempfile.TemporaryDirectory(prefix='infant-raw-v2-') as tmp, zipfile.ZipFile(DATA/(movie+'.zip')) as z:
            rawpath=Path(tmp)/'raw.nii'; digest=hashlib.sha256()
            with member(z,old['raw_member']) as src,gzip.GzipFile(fileobj=src) as gz,rawpath.open('wb') as dst:
                for b in iter(lambda:gz.read(4*1024*1024),b''): dst.write(b); digest.update(b)
            if digest.hexdigest()!=old['raw_uncompressed_sha256']: raise ValueError('raw hash mismatch')
            raw=nib.load(rawpath); tr=raw_geometry(raw); affine=raw.affine; shape=raw.shape[:3]
            brain_name=f'{movie}/transformation_ants/{ppt}/highres_brain.nii.gz'
            brain_path=Path(tmp)/'brain.nii.gz'; brain_hash=extract(z,brain_name,brain_path)
            anatomy=nib.load(brain_path); anat=anatomy.get_fdata(dtype=np.float32)
            if not np.isfinite(anat).all(): raise ValueError('nonfinite anatomical image')
            roi=nib.load(source.parent/'released_occipital.nii.gz'); roi_data=roi.get_fdata(dtype=np.float32)
            ah=receipt['anatomical_header']
            init=raw_to_mask_voxels(shape,affine,ah['shape'],np.array(ah['affine']),roi.affine,
                np.loadtxt(source.parent/'released_raw_to_highres_0.mat'))
            brain_init=np.linalg.inv(anatomy.affine)@roi.affine@init
            initial_anatomy=project(anat,brain_init,shape,order=1)
            initial_brain=project((anat>0).astype(np.uint8),brain_init,shape)>0
            prefix=np.asarray(raw.dataobj[...,:CAL.stop],dtype=np.float32)
            record['calibration_candidates']=[dict(raw_frame=i,**slice_quality(prefix[...,i],ndimage.binary_erosion(initial_brain),CAL)) for i in range(CAL.start,CAL.stop)]
            chosen,candidates=choose_reference(prefix,ndimage.binary_erosion(initial_brain),CAL)
            reference=prefix[...,chosen]
            residual,alignment=anatomical_refinement(reference,initial_anatomy,initial_brain,affine,CAL)
            correction=voxel_mapping(affine,residual)
            mapping=init@correction; brain_mapping=brain_init@correction
            projected_roi=project(roi_data,mapping,shape)>.5
            projected_brain=project((anat>0).astype(np.uint8),brain_mapping,shape)>0
            brain_interior=ndimage.binary_erosion(projected_brain)
            ref_quality=slice_quality(reference,brain_interior,CAL)
            if not ref_quality['passed']: raise ValueError('selected reference fails refined-brain slice QC')
            fixed_support=brain_interior&reference_support(reference)
            roi_interior=ndimage.binary_erosion(projected_roi)&brain_interior&fixed_support
            if roi_interior.sum()<OLD['min_voxels']: raise ValueError('insufficient brain-interior occipital voxels')
            ijk=spatially_spread(np.argwhere(roi_interior),OLD['max_voxels'])
            if not brain_interior[tuple(ijk.T)].all(): raise ValueError('selected voxel outside eroded brain')
            record.update(reference_frame=chosen,reference_quality=ref_quality,anatomical_alignment=alignment,
                anatomical_brain_member=brain_name,anatomical_brain_sha256=brain_hash,
                raw_to_roi_voxels=mapping.tolist(),raw_to_brain_voxels=brain_mapping.tolist(),
                selected_voxels=len(ijk),selected_outside_brain=0,raw_shape=list(raw.shape),tr_seconds=tr)
            for name,array in [('reference',reference),('brain_mask',projected_brain.astype(np.uint8)),
                               ('occipital_candidate',projected_roi.astype(np.uint8)),('anatomy_in_reference',project(anat,brain_mapping,shape,order=1))]:
                save_image(folder/(name+'.nii.gz'),array,affine)
            np.save(folder/'roi_voxels.npy',ijk); np.save(folder/'registration_mask.npy',fixed_support)
            if calibration_only:
                record['status']='calibrated'; write_json(folder/'qc.json',record); return record
            start=CAL.stop; stop=min(raw.shape[3],start+OLD['max_frames']); n=stop-start
            if n<OLD['min_valid_frames']: raise ValueError('insufficient raw analysis duration')
            values=np.full((len(ijk),n),np.nan,np.float32); covered=np.zeros(values.shape,bool)
            matrices=[]; registrations=[]; correlations=[]; failed=[]; slice_metrics=[]
            spatial=SpatialConfig(iterations=100,threads=2)
            for j,t in enumerate(range(start,stop)):
                frame=np.asarray(raw.dataobj[...,t],dtype=np.float32)
                try:
                    aligned,coverage,details=register_frame(frame,reference,affine,fixed_support,spatial)
                    matrix=np.array(details['fixed_to_moving_lps'])
                    native_mask=project(brain_interior.astype(np.uint8),voxel_mapping(affine,np.linalg.inv(matrix)),shape)>0
                    quality=slice_quality(frame,native_mask,CAL)
                    region=fixed_support&coverage
                    corr=float(np.corrcoef(aligned[region],reference[region])[0,1])
                    if not np.isfinite(corr): raise ValueError('nonfinite reference correlation')
                    values[:,j]=aligned[tuple(ijk.T)]; covered[:,j]=coverage[tuple(ijk.T)]
                    matrices.append(matrix); registrations.append(dict(raw_frame=t,**details)); failed.append(False); correlations.append(corr); slice_metrics.append(quality)
                except (RuntimeError,ValueError) as exc:
                    matrices.append(np.eye(4)); registrations.append(dict(raw_frame=t,error=str(exc))); failed.append(True); correlations.append(0.)
                    slice_metrics.append(dict(known=False,passed=False,alternation=None,residual_p90=None,slices=0))
                if (j+1)%10==0 or j==n-1: write_json(folder/'progress.json',dict(id=old['id'],completed=j+1,total=n))
            matrices=np.asarray(matrices); failed=np.array(failed)
            flip=np.diag([-1.,-1.,1.,1.]); center=(flip@affine@np.r_[(np.array(shape)-1)/2,1])[:3]
            def rms(delta):
                return np.sqrt(35.**2/3*np.sum(delta[:,:3,:3]**2,axis=(1,2))+np.sum((np.einsum('tij,j->ti',delta[:,:3,:3],center)+delta[:,:3,3])**2,axis=1))
            step=np.r_[0.,rms(np.diff(matrices,axis=0))]; absolute=rms(matrices-np.eye(4))
            clean,motion,signal,dvars=classify_frames(values,reference[tuple(ijk.T)],covered,step,absolute,np.array(correlations),failed)
            artifact=np.array([not r['passed'] for r in slice_metrics])
            signal |= artifact; signal[0]=True; dvars[0]=0.
            clean=~motion&~signal
            valid=clean[None,:]&covered&np.isfinite(values)
            recoverable=(~clean)[None,:]&covered&np.isfinite(values)&~failed[None,:]
            recoverable[:,0]=False
            record.update(analysis_start_frame=start,analysis_stop_frame=stop,analysis_frames=n,valid_frames=int(clean.sum()),
                motion_frames=int(motion.sum()),signal_qc_frames=int(signal.sum()),native_slice_qc_frames=int(artifact.sum()),
                failed_registration_frames=int(failed.sum()),registration=registrations,native_slice_quality=slice_metrics,
                rms_step_mm=step.tolist(),rms_from_reference_mm=absolute.tolist(),fractional_dvars=dvars.tolist(),
                reference_correlation=correlations,spatial_config=asdict(spatial))
            if clean.sum()<OLD['min_valid_frames'] or clean.mean()<OLD['min_valid_fraction']:
                record['status']='excluded_insufficient_qc_observations'; write_json(folder/'qc.json',record); return record
            prep_receipt=dict(source_kind=SOURCE_KIND,temporal_operations=[],reference_frame=chosen,
                reference_outside_analysis=True,calibration_prefix_stop=CAL.stop,timing_verified=False,
                timing_policy='raw_acquisition_only',gaze_policy='unconditioned_unknown',qc_frozen_before_artificial_holdouts=True,
                unique_acquisition_sha256=old['raw_uncompressed_sha256'],brain_interior_only=True,native_slice_qc=True,
                calibration_anatomy_refined=True,initial_spatial_metadata='published affine; independence conditional on external metadata',
                protocol=PROTOCOL,scope='Exploratory raw BOLD temporal imputation; no movie/gaze/cortical-gray-matter claim')
            indices=np.arange(start,stop,dtype=np.int64); path=folder/'record.npz'
            np.savez_compressed(path,values=np.where(valid,values,np.nan),valid=valid,recoverable=recoverable,
                coords_mm=nib.affines.apply_affine(affine,ijk).astype(np.float32),times_s=(indices*tr).astype(np.float32),
                raw_frame_indices=indices,voxel_ijk=ijk,padding=np.zeros(n,bool),gaze_known=np.zeros(n,bool),
                motion_bad=motion,signal_qc_bad=signal,native_slice_bad=artifact,fixed_to_moving_lps=matrices,
                diagnostic_rms_step_mm=step,source_kind=np.array(SOURCE_KIND),preparation_receipt=np.array(json.dumps(prep_receipt,sort_keys=True)))
            load_record(path); record.update(status='prepared',path=str(path.relative_to(out)),record_sha256=sha256(path))
    except ValueError as exc:
        record.update(status='excluded_calibration_or_input_qc',error=str(exc))
    except Exception as exc:
        record.update(status='failed',error=repr(exc))
    write_json(folder/'qc.json',record); return record

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--workers',type=int,default=1)
    p.add_argument('--ids',nargs='*'); p.add_argument('--calibration-only',action='store_true'); a=p.parse_args()
    output=Path(a.out); output.mkdir(parents=True,exist_ok=False)
    rows=json.loads(SUMMARY.read_text())['records']; groups={}
    for row in rows: groups.setdefault(row['raw_uncompressed_sha256'],[]).append(row)
    tasks=[]
    for group in groups.values():
        group.sort(key=lambda r:r['id']); row=group[0]
        if a.ids and row['id'] not in a.ids: continue
        assert len({r['family'] for r in group})==1 and len({r['split'] for r in group})==1
        tasks.append((row,[r['id'] for r in group],str(output),a.calibration_only))
    write_json(output/'protocol.json',dict(protocol=PROTOCOL,source_summary_sha256=sha256(SUMMARY),
        unique_candidates=len(tasks),source_kind=SOURCE_KIND,scope='Engineering protocol; native artifact thresholds require scientific sensitivity analysis'))
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(prepare,task) for task in tasks]
        for future in as_completed(futures):
            r=future.result(); results.append(r)
            write_json(output/'progress.json',dict(completed=len(results),total=len(tasks),last={k:r.get(k) for k in ['id','status','error']}))
            print(json.dumps({k:r.get(k) for k in ['id','status','reference_frame','valid_frames','native_slice_qc_frames','error']}),flush=True)
    results.sort(key=lambda r:r['id'])
    selected=[{k:r[k] for k in ['id','family','split','path','source_kind']} for r in results if r['status']=='prepared']
    write_json(output/'manifest.json',selected)
    write_json(output/'summary.json',dict(status='pending_independent_validation',records=results,attempted=len(tasks),prepared=len(selected),failed=sum(r['status']=='failed' for r in results)))
    write_json(output/'qa_gate.json',dict(status='blocked',blockers=['independent validation pending'],manifest_sha256=sha256(output/'manifest.json')))

if __name__=='__main__': main()
