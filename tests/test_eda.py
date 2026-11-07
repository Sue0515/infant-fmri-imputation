import gzip
import json
import zipfile

import nibabel as nib
import numpy as np

from stsr.imputation.data import sha256
from stsr.imputation.eda import clean_windows, enrich, select_signals, signal_qc, summarize


def record(movie='Aeronaut', ppt='s0001_1_1', known=True):
    motion, eye = np.zeros(10, dtype=bool), np.zeros(10, dtype=bool)
    motion[[2,3]], eye[[3,4]] = True, True
    return dict(id=movie+':'+ppt, movie=movie, ppt=ppt, family=ppt.split('_')[0], split='train',
                gaze_known=known, eligible=known and movie!='Child_Play', tr_seconds=2., stored_frames=13,
                acquired_frames=10, motion=motion.tolist(), eye=eye.tolist(),
                valid=(~(motion|eye) if known else np.zeros(10,dtype=bool)).tolist())


def test_cohort_deduplicates_infants_sessions_and_families():
    rows=[record(),record('Meerkats'),record(ppt='s0001_1_2')]
    meta={(r['movie'],r['ppt']):dict(age=8 if r['ppt'].endswith('_2') else 6,sex='F') for r in rows}
    enriched=enrich(rows,meta); result=summarize(enriched)
    assert (result['records'],result['subjects'],result['sessions'],result['families'])==(3,1,2,1)
    assert result['entry_age_months']['0.5']==6
    assert result['subjects_with_repeat_sessions']==1
    assert enriched[0]['categories']==dict(clean=7,motion_only=1,eye_only=1,overlap=1,gaze_unknown=0)
    assert enriched[0]['usable_seconds']==14
    assert sum(result['joint_state_frames'].values())==result['acquired_frames']


def test_unknown_gaze_and_child_play_do_not_claim_clean_windows():
    rows=[record(known=False),record('Child_Play')]
    a,b=enrich(rows,{(r['movie'],r['ppt']):dict(age=6,sex='F') for r in rows})
    assert a['usable_seconds'] is None and a['categories']['gaze_unknown']==10
    assert a['windows'] is None and b['windows'] is None and b['longest_clean_seconds'] is None


def test_windows_require_full_observations_and_remaining_context():
    result=clean_windows([1,1,0,1,1,1,0,1,1])
    assert result['2']==dict(starts=4,leading=True,trailing=True)
    assert result['4']['starts']==0
    assert clean_windows([1,1,1],lengths=(2,))['2']['starts']==0


def test_signal_selection_is_stable_and_train_only():
    rows=[record(),record('Meerkats'),dict(record('Meerkats','s0002_1_1'),split='test'),
          dict(record('Aeronaut','s0003_1_1'),split='val'),record('Child_Play')]
    selected=select_signals(rows,1)
    assert {r['id'] for r in selected}=={rows[0]['id'],rows[1]['id']}
    assert selected==select_signals(rows[::-1],1)
    assert select_signals(rows,0)==[]


def test_signal_probe_flags_nan_padding_and_duplicates_without_mutating_source(tmp_path):
    x=np.broadcast_to(np.arange(1,9,dtype=np.float32),(3,4,2,8)).copy()
    x[...,2]=x[...,1]; x[...,6:]=0; x[0,1,0,3]=np.nan
    img=nib.Nifti1Image(x,np.diag([3.,3.,3.,1.]))
    img.header.set_xyzt_units('mm','sec'); img.header.set_zooms((3.,3.,3.,2.))
    path=tmp_path/'Movie.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('Movie/preprocessed_native/linear_alignment/s0001_1_1_Z.nii.gz',gzip.compress(img.to_bytes()))
    before=sha256(path)
    row=dict(id='Movie:s0001_1_1',movie='Movie',ppt='s0001_1_1',split='train',stored_frames=8,acquired_frames=6,tr_seconds=2.,valid=[True]*6)
    result=signal_qc(tmp_path,row)
    assert result['probe_allzero_frames']==[6,7]
    assert result['probe_nonfinite_frames']==[3]
    assert result['probe_identical_previous_frames']==[2,7]
    assert result['excluded_tail_probe_nonzero_frames']==[]
    assert sha256(path)==before
    json.dumps(result,allow_nan=False)
