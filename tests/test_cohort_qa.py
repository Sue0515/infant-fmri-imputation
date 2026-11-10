import gzip
import json
import zipfile
import nibabel as nib
import numpy as np
import pytest
from stsr.imputation import prepare_raw_qc as prep
from stsr.imputation.data import sha256
from stsr.imputation.runner import validate_cohort_qa


def test_empty_or_nonfinite_reference_cannot_become_whole_volume_mask():
    for values in [np.ones((8,8,8)), np.full((8,8,8),np.nan)]:
        with pytest.raises(ValueError): prep.reference_support(values)


@pytest.mark.parametrize('field',['correlation','step','absolute'])
def test_unknown_qc_metric_cannot_become_clean_target(field):
    args=dict(values=np.full((3,4),100.),reference_values=np.full(3,100.),
              coverage=np.ones((3,4),bool),step=np.zeros(4),absolute=np.zeros(4),
              correlation=np.ones(4),failed=np.zeros(4,bool))
    args[field][1]=np.nan
    clean,_,signal,_=prep.classify_frames(**args)
    assert clean.tolist()==[True,False,True,True]
    assert signal[1]


def test_post_registration_failure_keeps_one_entry_per_frame(tmp_path,monkeypatch):
    source=tmp_path/'support'; source.mkdir()
    xyz=np.indices((12,12,12)).sum(0).astype(np.float32)+100
    image=nib.Nifti1Image(np.repeat(xyz[...,None],45,axis=3),np.eye(4))
    image.header.set_xyzt_units('mm','sec'); image.header.set_zooms((1,1,1,2))
    raw=tmp_path/'raw.nii.gz'; nib.save(image,raw)
    with zipfile.ZipFile(tmp_path/'Movie.zip','w') as archive:
        archive.write(raw,'raw.nii.gz')
    nib.save(nib.Nifti1Image(np.ones((12,12,12)),np.eye(4)),source/'released_occipital.nii.gz')
    np.savetxt(source/'released_raw_to_highres_0.mat',np.eye(4))
    receipt=dict(family='s001',sources=[],raw_geometry=[dict(raw_member='raw.nii.gz')],
                 anatomical_header=dict(shape=[12,12,12],affine=np.eye(4).tolist()))
    (source/'source_receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(prep,'reference_support',lambda a:np.ones(a.shape,bool))
    def bad_registration(frame,reference,affine,mask,config):
        # Registration itself succeeds but downstream correlation is undefined.
        return np.full(frame.shape,np.nan,np.float32),np.ones(frame.shape,bool),{'fixed_to_moving_lps':np.eye(4).tolist()}
    monkeypatch.setattr(prep,'register_frame',bad_registration)
    with np.errstate(invalid='ignore'):
        result=prep.prepare_one((dict(id='Movie:s001_1',split='train',receipt='source_receipt.json'),str(source),str(tmp_path),str(tmp_path/'out')))
    assert result['status']=='excluded_insufficient_qc_observations'
    assert result['failed_registration_frames']==34
    assert len(result['registration'])==len(result['rms_step_mm'])==34
    assert result['valid_frames']==0
    assert nib.load(tmp_path/'out/Movie/s001_1/reference_frame10.nii.gz').header.get_xyzt_units()[0]=='mm'


def test_training_gate_rejects_missing_blocked_stale_and_duplicate_data(tmp_path):
    config=dict(required_source_kind=prep.SOURCE_KIND)
    manifest=tmp_path/'manifest.json'; manifest.write_text('[]')
    rows=[]
    for i in range(2):
        folder=tmp_path/str(i); folder.mkdir(); record=folder/'record.npz'; record.write_bytes(bytes([i]))
        (folder/'qc.json').write_text(json.dumps(dict(raw_uncompressed_sha256=str(i)*64)))
        rows.append(dict(id=str(i),path=str(record)))
    with pytest.raises(ValueError,match='missing'): validate_cohort_qa(manifest,rows,config)
    path=tmp_path/'qa_gate.json'
    path.write_text(json.dumps(dict(status='blocked',blockers=['anatomy'])))
    with pytest.raises(ValueError,match='blocked'): validate_cohort_qa(manifest,rows,config)
    gate=dict(status='approved_for_training',manifest_sha256=sha256(manifest),
              input_sha256={r['id']:sha256(r['path']) for r in rows},
              raw_identity_sha256={str(i):str(i)*64 for i in range(2)})
    path.write_text(json.dumps(gate)); validate_cohort_qa(manifest,rows,config)
    gate['raw_identity_sha256']['1']='0'*64; path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='unique'): validate_cohort_qa(manifest,rows,config)
    gate['raw_identity_sha256']['1']='2'*64; path.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='identity does not match'): validate_cohort_qa(manifest,rows,config)
    gate['raw_identity_sha256']['1']='1'*64; path.write_text(json.dumps(gate))
    manifest.write_text('[ ]')
    with pytest.raises(ValueError,match='manifest hash'): validate_cohort_qa(manifest,rows,config)
    manifest.write_text('[]'); (tmp_path/'0/record.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input hashes'): validate_cohort_qa(manifest,rows,config)
