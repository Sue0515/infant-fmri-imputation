import json
import zipfile

import nibabel as nib
import numpy as np
import pytest
import torch

from stsr.imputation.prepare_raw import SOURCE_KIND, checked_indices, prepare, prepare_volume, raw_geometry
from stsr.imputation.raw_spatial import SpatialConfig, image_from_array, register_frame
from stsr.imputation.runner import build_model, load_rows, restore, train, validate_source
from stsr.imputation.metrics import score
from stsr.imputation.prepare_dryad import member
from test_irregular_imputation import config, fixture, write_records


def raw_fixture():
    xyz=np.indices((20,20,20)).astype(float)
    center=np.array([9,10,8])[:,None,None,None]
    x=np.exp(-np.sum((xyz-center)**2,axis=0)/22)*1000
    x+=np.exp(-np.sum((xyz-np.array([13,6,12])[:,None,None,None])**2,axis=0)/8)*600
    image=nib.Nifti1Image(np.stack([x*(1+i*.01) for i in range(6)],axis=-1).astype(np.float32),np.diag([3.,3.,3.,1.]))
    image.header.set_xyzt_units('mm','sec')
    image.header.set_zooms((3.,3.,3.,2.))
    row=dict(analysis_start_frame=1,analysis_stop_frame=6,reference_frame=0,burn_in_frames=0,
             timing=dict(verified=True,evidence='synthetic acquisition',movie_onset_raw_seconds=2.),
             roi_evidence='synthetic fixed geometry')
    mask=x>20
    roi=np.array([[9,10,8],[13,6,12]],dtype=int)
    bad=np.zeros(6,dtype=bool)
    known=np.ones(6,dtype=bool)
    return image,row,roi,mask,bad,known


def test_raw_time_units_and_reference_must_be_outside_scored_interval():
    image,row,*_=raw_fixture()
    assert raw_geometry(image)==2
    image.header.set_xyzt_units('mm','msec')
    image.header.set_zooms((3.,3.,3.,2000.))
    assert raw_geometry(image)==2
    assert checked_indices(row,6,2.)[0].tolist()==[1,2,3,4,5]
    with pytest.raises(ValueError,match='outside'):
        checked_indices(dict(row,reference_frame=2),6,2.)
    with pytest.raises(ValueError,match='verified'):
        checked_indices(dict(row,timing=dict(verified=False)),6,2.)


def test_ras_lps_geometry_and_shear_rejection():
    pytest.importorskip('SimpleITK')
    affine=np.array([[0,-3,0,20],[3,0,0,-30],[0,0,3,10],[0,0,0,1]],dtype=float)
    image=image_from_array(np.ones((10,11,12),dtype=np.float32),affine)
    index=np.array([2,3,4])
    expected=(affine@np.r_[index,1])[:3]*[-1,-1,1]
    np.testing.assert_allclose(image.TransformIndexToPhysicalPoint(tuple(int(i) for i in index)),expected)
    affine[0,0]=1
    with pytest.raises(ValueError,match='sheared'):
        image_from_array(np.ones((10,11,12),dtype=np.float32),affine)


def test_framewise_registration_recovers_known_spatial_translation():
    pytest.importorskip('SimpleITK')
    from scipy.ndimage import shift
    image,row,roi,mask,bad,known=raw_fixture()
    reference=np.asarray(image.dataobj[...,0])
    moving=shift(reference,(1.,-.5,0.),order=1,mode='constant')
    aligned,coverage,info=register_frame(moving,reference,image.affine,mask,SpatialConfig(iterations=100,threads=1))
    region=mask&coverage
    before=np.mean((moving[region]-reference[region])**2)
    after=np.mean((aligned[region]-reference[region])**2)
    assert after < before*.2
    # Check displacement at the object's center, not the homogeneous origin:
    # small residual rotations contribute a center-dependent offset.
    matrix=np.asarray(info['fixed_to_moving_lps'])
    center=np.array([-27.,-30.,24.,1.])
    np.testing.assert_allclose((matrix@center-center)[:3],[-3,1.5,0],atol=.8)


def test_changing_one_raw_target_does_not_change_other_prepared_values():
    pytest.importorskip('SimpleITK')
    image,row,roi,mask,bad,known=raw_fixture()
    cfg=SpatialConfig(iterations=15,threads=1)
    a,_=prepare_volume(image,row,roi,mask,bad,bad,known,cfg)
    changed=image.get_fdata(dtype=np.float32)
    changed[...,3] = np.roll(changed[...,3],2,axis=0)*1.7
    other=nib.Nifti1Image(changed,image.affine,image.header)
    b,_=prepare_volume(other,row,roi,mask,bad,bad,known,cfg)
    np.testing.assert_array_equal(a['values'][:,[0,1,3,4]],b['values'][:,[0,1,3,4]])
    assert not np.array_equal(a['values'][:,2],b['values'][:,2])
    assert a['raw_frame_indices'].tolist()==[1,2,3,4,5]
    assert a['times_s'].tolist()==[2,4,6,8,10]
    assert a['movie_times_s'].tolist()==[0,2,4,6,8]


def test_raw_motion_and_gaze_never_become_targets_or_input():
    pytest.importorskip('SimpleITK')
    image,row,roi,mask,motion,known=raw_fixture()
    motion[2:4]=True
    gaze=np.zeros(6,dtype=bool)
    gaze[3]=True
    known[4]=False
    arrays,_=prepare_volume(image,row,roi,mask,motion,gaze,known,SpatialConfig(iterations=5,threads=1))
    assert np.isnan(arrays['values'][:,1:4]).all()
    assert arrays['recoverable'][:,1].all()
    assert not arrays['recoverable'][:,2:4].any()


def test_raw_plan_rejects_unverified_timing_before_creating_output(tmp_path):
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(spatial={},records=[dict(movie='Aeronaut',ppt='s0001_1_1',family='s0001',split='train',
           raw_member='Aeronaut/raw_nifti/s0001_1_1_functional01.nii.gz',timing=dict(verified=False))])))
    with pytest.raises(ValueError,match='verified'):
        prepare(plan,tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_raw_training_refuses_legacy_preprocessed_records(tmp_path):
    _,rows=write_records(tmp_path)
    with pytest.raises(ValueError,match='raw experiment refuses'):
        load_rows(rows,dict(config(),required_source_kind=SOURCE_KIND),'cpu')


@pytest.mark.parametrize('key,value',[('initialization','pretrained'),('init_checkpoint','existing.pt'),('resume','existing.pt')])
def test_training_initialization_cannot_use_existing_weights(key,value):
    with pytest.raises(ValueError,match='random initialization'):
        build_model(dict(config(),**{key:value}),'cpu')


def test_voxel_normalization_ignores_hidden_values_and_handles_query_order():
    x,valid,coords,times=fixture()
    x=x*torch.arange(1,9)[None,:,None]*50+torch.arange(8)[None,:,None]*1000
    model=build_model(dict(config(),normalization='voxel'),'cpu').eval()
    a=model.encode(x,valid,coords,times)
    b=model.encode(x.masked_fill(~valid,1e12),valid,coords,times)
    torch.testing.assert_close(a['features'],b['features'],rtol=0,atol=0)
    expected=torch.stack([x[0,v][valid[0,v]].mean() for v in range(8)])
    torch.testing.assert_close(a['mean'][0,:,0],expected)
    prediction=model.query(a,[7,1,4],[0.,8.,12.])
    torch.testing.assert_close(prediction,torch.stack([x[0,7,0],x[0,1,4],x[0,4,6]])[None],rtol=0,atol=0)
    target=torch.zeros_like(valid)
    target[:,:,[5,6]]=True
    loss,_=model.loss(x,valid&~target,target,coords,times)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_normalized_error_uses_observed_only_voxel_scale():
    truth=torch.tensor([[[10.,14.,1000.]]])
    observed=torch.tensor([[[True,True,False]]])
    prediction=torch.tensor([[[10.,14.,996.]]])
    result=score(prediction,truth,~observed,observed,torch.arange(3).float()*2)
    assert result['mse']==16 and result['normalized_mse']==4


def test_raw_receipt_and_voxel_config_round_trip_training(tmp_path):
    path,rows=write_records(tmp_path)
    receipt=dict(source_kind=SOURCE_KIND,timing_verified=True,reference_outside_analysis=True,temporal_operations=[])
    for row in rows:
        with np.load(row['path']) as data:
            arrays={key:data[key] for key in data.files}
        np.savez(row['path'],**arrays,source_kind=SOURCE_KIND,preparation_receipt=json.dumps(receipt))
    cfg=dict(config(),normalization='voxel',required_source_kind=SOURCE_KIND,initialization='random',selection_metric='normalized_mse')
    train(path,cfg,tmp_path/'run')
    model,saved=restore(tmp_path/'run'/'best.pt','cpu')
    assert model.normalization=='voxel' and saved['provenance']['initialization']=='random'
    assert saved['provenance']['pretrained_checkpoint'] is None
    assert saved['selection_metric']=='normalized_mse'
    assert np.isfinite(saved['selection_value'])
    receipt['temporal_operations']=['zscore_full_run']
    np.savez(tmp_path/'invalid.npz',source_kind=SOURCE_KIND,preparation_receipt=json.dumps(receipt))
    with pytest.raises(ValueError,match='no temporal processing'):
        validate_source(tmp_path/'invalid.npz',cfg)


def test_zip_repairs_wrong_end_boundary_without_disabling_overlap_checks(tmp_path):
    path=tmp_path/'normal.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('first.txt',b'first payload')
        archive.writestr('second.txt',b'second payload')
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo('first.txt')
        if hasattr(info,'_end_offset'):
            info._end_offset=info.header_offset
        # Simulate an offset with the wrong 32-bit wrap count as well.
        archive.getinfo('second.txt').header_offset+=2**32
        with member(archive,'first.txt') as stream:
            assert stream.read()==b'first payload'
        with member(archive,'second.txt') as stream:
            assert stream.read()==b'second payload'
        if hasattr(info,'_end_offset'):
            assert info._end_offset==archive.getinfo('second.txt').header_offset


def test_zip_rejects_payload_overlapping_next_verified_header(tmp_path):
    path=tmp_path/'overlap.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('first.txt',b'first')
        archive.writestr('second.txt',b'second')
    with zipfile.ZipFile(path) as archive:
        archive.getinfo('first.txt').compress_size+=100
        with pytest.raises(ValueError,match='overlaps verified next header'):
            member(archive,'first.txt')


def test_complete_raw_zip_preparation_preserves_time_and_provenance(tmp_path):
    pytest.importorskip('SimpleITK')
    image,row,roi,mask,motion,known=raw_fixture()
    motion[2]=True
    source=tmp_path/'raw.nii.gz'
    nib.save(image,source)
    raw_member='Aeronaut/raw_nifti/s0001_1_1_functional01.nii.gz'
    with zipfile.ZipFile(tmp_path/'Aeronaut.zip','w') as archive:
        archive.write(source,raw_member)
    row.update(movie='Aeronaut',ppt='s0001_1_1',family='s0001',split='train',
               raw_member=raw_member,annotation_evidence='synthetic raw-frame flags')
    for key,array in dict(roi_voxels=roi,registration_mask=mask,raw_motion_bad=motion,
                          raw_gaze_bad=np.zeros(6,dtype=bool),raw_gaze_known=known).items():
        row[key]=key+'.npy'
        np.save(tmp_path/row[key],array)
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps(dict(spatial=dict(iterations=5,threads=1),records=[row])))
    manifest_path=prepare(selection,tmp_path,tmp_path/'prepared')
    rows=json.loads(manifest_path.read_text())
    path=manifest_path.parent/rows[0]['path']
    validate_source(path,dict(required_source_kind=SOURCE_KIND))
    with np.load(path) as record:
        assert record['raw_frame_indices'].tolist()==[1,2,3,4,5]
        assert record['times_s'].tolist()==[2,4,6,8,10]
        assert np.isnan(record['values'][:,1]).all()
        assert record['recoverable'][:,1].all()
    qc=json.loads(path.with_suffix('.json').read_text())
    assert len(qc['raw_uncompressed_sha256'])==64
    assert len(qc['annotation_sha256'])==5
    assert qc['selection']['annotation_evidence']=='synthetic raw-frame flags'
