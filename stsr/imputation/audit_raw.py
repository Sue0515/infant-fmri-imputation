"""Inventory raw movie geometry/timing without inventing missing alignment."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import nibabel as nib

from .audit_masks import MOVIES
from .prepare_dryad import member, read
from .raw_spatial import SpatialConfig
from dataclasses import asdict


def audit(root, split_path, output):
    root, split_path, output = Path(root), Path(split_path), Path(output)
    split = json.loads(split_path.read_text())['families']
    rows, burn_files, sources = [], [], []
    for movie in MOVIES:
        with zipfile.ZipFile(root/(movie+'.zip')) as archive:
            names = [n for n in archive.namelist() if not n.startswith('__MACOSX/') and not n.endswith('/')]
            burn_files.extend(dict(archive=movie+'.zip', member=n) for n in names if 'burn' in n.lower())
            table_name = movie+'/infant_participants.csv'
            content = read(archive, table_name)
            sources.append(dict(archive=movie+'.zip', member=table_name, sha256=hashlib.sha256(content).hexdigest()))
            for participant in csv.DictReader(io.StringIO(content.decode('utf-8-sig'))):
                ppt = participant['ppt']
                raw_names = [n for n in names if n.startswith(movie+'/raw_nifti/'+ppt+'_functional') and n.endswith(('.nii','.nii.gz'))]
                raw_files, timing_files = [], []
                for name in raw_names:
                    with member(archive, name) as raw:
                        stream = gzip.GzipFile(fileobj=raw) if name.endswith('.gz') else raw
                        header = nib.Nifti1Header.from_fileobj(stream)
                    unit = header.get_xyzt_units()[1]
                    tr = float(header.get_zooms()[3])*{'sec':1.,'msec':.001,'usec':1e-6}.get(unit, float('nan'))
                    raw_files.append(dict(member=name, shape=list(header.get_data_shape()), tr_seconds=tr,
                                          crc32=archive.getinfo(name).CRC, compressed_bytes=archive.getinfo(name).compress_size))
                for name in names:
                    if name.startswith(movie+'/raw_timing/'+ppt+'_functional') and name.endswith('.txt'):
                        content=read(archive,name)
                        events=[list(map(float,line.split())) for line in content.decode().splitlines() if line.strip()]
                        timing_files.append(dict(member=name, events=events, sha256=hashlib.sha256(content).hexdigest()))
                reasons=['raw_frame_motion_gaze_annotations_not_verified', 'anatomical_roi_not_selected',
                         'raw_movie_time_origin_and_run_burn_in_not_verified']
                if movie=='Child_Play':
                    reasons.append('continuous_clip_mapping_required')
                if len(raw_files)!=1 or len(timing_files)!=1 or any(len(t['events'])!=1 for t in timing_files):
                    reasons.append('multiple_or_missing_raw_runs_or_events_require_explicit_mapping')
                rows.append(dict(id=movie+':'+ppt, movie=movie, ppt=ppt, family=ppt.split('_')[0],
                                 split=split[ppt.split('_')[0]], csv_num_tr=int(participant['num_TR']),
                                 raw_files=raw_files, raw_timing=timing_files, ready=False, blockers=reasons))
    # README advertises these files; inventory all archives before declaring absent.
    with zipfile.ZipFile(root/'0-predict_retinotopy.zip') as archive:
        burn_files.extend(dict(archive='0-predict_retinotopy.zip',member=n) for n in archive.namelist()
                          if 'burn' in n.lower() and not n.startswith('__MACOSX/') and not n.endswith('/'))
    output.mkdir(parents=True,exist_ok=False)
    result=dict(records=rows, burn_in_files=burn_files, sources=sources,
                summary=dict(infant_records=len(rows), continuous_movie_records=sum(r['movie']!='Child_Play' for r in rows),
                             raw_files=sum(len(r['raw_files']) for r in rows), ready_records=0),
                preserved_split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                note='Header/timing inventory only. No inference of burn-in from length and no processed-to-raw mask copying.')
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    candidates=[]
    for row in rows:
        if row['movie']=='Child_Play' or len(row['raw_files'])!=1:
            continue
        candidates.append(dict(movie=row['movie'],ppt=row['ppt'],family=row['family'],split=row['split'],
            raw_member=row['raw_files'][0]['member'],analysis_start_frame=None,analysis_stop_frame=None,
            reference_frame=None,burn_in_frames=None,
            timing=dict(verified=False,evidence='',movie_onset_raw_seconds=None,source_files=row['raw_timing']),
            annotation_evidence='',roi_evidence='',roi_voxels='',registration_mask='',
            raw_motion_bad='',raw_gaze_bad='',raw_gaze_known=''))
    (output/'selection_unverified.json').write_text(json.dumps(dict(spatial=asdict(SpatialConfig()),records=candidates),indent=2)+'\n')
    print(json.dumps(dict(**result['summary'],burn_in_files=len(burn_files)),indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--split',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    audit(args.data_root,args.split,args.out)


if __name__=='__main__':
    main()
