import json

import numpy as np
import pytest
import torch

from stsr.imputation.data import Record, check_evaluation_split, load_record, manifest
from stsr.imputation.masks import InsufficientContext, draw_holdout
from stsr.imputation.metrics import interpolate, score
from stsr.imputation.model import TemporalImputer
from stsr.imputation.runner import evaluate, infer, restore, train
from stsr.imputation.prepare_dryad import prepare, validity
from stsr.imputation.audit_masks import family_split
from stsr.imputation.profiles import describe, draw_profile, draw_stress, fit_profile
from stsr.imputation.runner import profile_from_records
from stsr.model.spacetime_attention import SpaceTimeAttentionNet


def config():
    return dict(seed=2, steps=2, lr=1e-3, eval_every=1, eval_masks=1, query_chunk=32,
                limits=dict(max_vertices=16, max_frames=32),
                mask=dict(gap_lengths=[1, 2, 3], max_gaps=2, allow_edges=True),
                model=dict(d_model=32, n_blocks=2, n_heads=4, attn_mode='factorized',
                           factorized_time_window=0, window_vertices=4, window_time=4,
                           use_routing=True, with_z_head=False, inr_k_neighbors=4))


def fixture():
    torch.manual_seed(15)
    coords = torch.cartesian_prod(torch.arange(2), torch.arange(2), torch.arange(2)).float() * 3
    times = torch.arange(16).float() * 2
    x = torch.randn(1, 8, 16)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[..., [2, 3, 10]] = False
    return x, valid, coords, times


def model():
    return TemporalImputer(SpaceTimeAttentionNet(**config()['model']), query_chunk=32)


def test_masks_keep_real_invalid_out_of_inputs_and_targets():
    _, valid, _, _ = fixture()
    seen_edges = False
    for seed in range(20):
        a, b = draw_holdout(valid[0], seed=seed, empirical=[~valid[0, 0]], **config()['mask'])
        c, d = draw_holdout(valid[0], seed=seed, empirical=[~valid[0, 0]], **config()['mask'])
        assert torch.equal(a, c) and torch.equal(b, d)
        assert not (a & b).any() and not (a & ~valid[0]).any() and not (b & ~valid[0]).any()
        assert b.any() and (a.sum(-1)[b.any(-1)] >= 2).all()
        seen_edges |= bool(b[:, [0, -1]].any())
    assert seen_edges
    with pytest.raises(InsufficientContext):
        draw_holdout(torch.zeros(8, 16, dtype=torch.bool), seed=1)


@pytest.mark.parametrize('poison', [float('nan'), 1e10])
def test_hidden_values_cannot_change_features_predictions_or_input_gradients(poison):
    x, valid, coords, times = fixture()
    m = model().eval()
    condition, target = draw_holdout(valid[0], seed=4, **config()['mask'])
    condition, target = condition[None], target[None]
    changed = x.masked_fill(~condition, poison)
    a, b = m.encode(x, condition, coords, times), m.encode(changed, condition, coords, times)
    for key in ('features', 'mean', 'scale'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    with torch.no_grad():
        pred = m(changed, condition, coords, times)
        torch.testing.assert_close(pred, m(x, condition, coords, times), rtol=0, atol=0)
    assert torch.equal(pred[condition], x[condition])
    source = changed.detach().requires_grad_(True)
    m(source, condition, coords, times)[target].sum().backward()
    assert torch.isfinite(source.grad).all() and (source.grad[~condition] == 0).all()
    assert source.grad[condition].abs().sum() > 0


def test_loss_reaches_trunk_and_inr_and_ignores_actual_invalid_values():
    x, valid, coords, times = fixture()
    m = model().train()
    condition, target = draw_holdout(valid[0], seed=4, **config()['mask'])
    x = x.masked_fill(~valid, float('nan'))
    loss, stats = m.loss(x, condition[None], target[None], coords, times)
    assert torch.isfinite(loss) and stats['target_values'] == int(target.sum())
    loss.backward()
    gradients = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in gradients.values())
    assert any('blocks' in n and g.abs().sum() > 0 for n, g in gradients.items())
    assert any('inr_head' in n and g.abs().sum() > 0 for n, g in gradients.items())
    with pytest.raises(ValueError):
        m.loss(x, valid, valid, coords, times)


def test_query_preserves_native_times_and_rejects_compressed_axis():
    x, valid, coords, times = fixture()
    m = model().eval()
    encoded = m.encode(x, valid, coords, times)
    assert torch.isfinite(m.query(encoded, [0, 1], [5., 19.])).all()
    with pytest.raises(ValueError, match='interval'):
        m.query(encoded, [0], [-1.])
    with pytest.raises(ValueError, match='integer'):
        m.query(encoded, [.5], [5.])
    with pytest.raises(ValueError, match='uniform'):
        m.encode(x[..., [0, 1, 3]], valid[..., [0, 1, 3]], coords, times[[0, 1, 3]])


def test_constant_observations_have_finite_gradient():
    x, valid, coords, times = fixture()
    x = torch.ones_like(x).requires_grad_(True)
    m = model()
    m(x, valid, coords, times)[~valid].sum().backward()
    assert torch.isfinite(x.grad).all()


def test_baselines_and_metrics_use_matching_targets_and_physical_time():
    x = torch.tensor([[[0., float('nan'), float('nan'), 6., float('nan')]]])
    observed = torch.isfinite(x)
    t = torch.arange(5).float() * 2
    linear = interpolate(x, observed, t)
    mean = interpolate(x, observed, t, 'mean_included')
    torch.testing.assert_close(linear, torch.tensor([[[0., 2., 4., 6., 6.]]]))
    torch.testing.assert_close(mean, torch.tensor([[[0., 3., 3., 6., 6.]]]))
    result = score(linear, torch.tensor([[[0., 2., 4., 6., 8.]]]), ~observed, observed, t)
    assert result['n'] == 3 and result['support']['two_sided']['n'] == 2
    assert result['support']['one_sided']['n'] == 1
    assert result['support']['bracket_4_8s']['n'] == 2


def write_records(tmp_path):
    rows = []
    x, valid, coords, times = fixture()
    for i, split in enumerate(('train', 'val', 'test')):
        path = tmp_path / (split + '.npz')
        # Motion gaps plus two ineligible trailing padding frames.
        v = valid[0].clone()
        v[:, -2:] = False
        recoverable = ~valid[0]
        np.savez(path, values=x[0].numpy() + i, valid=v.numpy(), recoverable=recoverable.numpy(),
                 coords_mm=coords.numpy(), times_s=times.numpy(), padding=np.arange(len(times)) >= 14)
        rows.append(dict(id=split, family='family_' + split, split=split, path=str(path)))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(rows))
    return path, rows


def test_family_split_leakage_and_record_contract(tmp_path):
    path, rows = write_records(tmp_path)
    r = load_record(rows[0]['path'])
    assert torch.isnan(r.values[~r.valid]).all()
    rows[1]['family'] = rows[0]['family']
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='family leakage'):
        manifest(path)
    changed_test = [dict(rows[0], split='test')]
    with pytest.raises(ValueError, match='overlaps'):
        check_evaluation_split(changed_test, rows, 'test')


def test_train_reload_evaluate_and_infer_do_not_fill_padding(tmp_path):
    path, rows = write_records(tmp_path)
    history = train(path, config(), tmp_path / 'run')
    assert len(history) == 2
    m, saved = restore(tmp_path / 'run' / 'best.pt', 'cpu')
    assert saved['provenance']['empirical_families'] == ['family_train']
    assert 'test' not in saved['provenance']['input_sha256']
    validation = load_record(rows[1]['path'])
    result = evaluate(m, [(rows[1], validation)], config())
    original = json.loads((tmp_path / 'run' / 'validation.json').read_text())
    assert result == original
    test = load_record(rows[2]['path'])
    output = infer(m, test)
    assert np.array_equal(output['values'][output['valid']], test.values.numpy()[output['valid']])
    assert np.isfinite(output['values'][output['imputed']]).all()
    assert np.isnan(output['values'][:, -2:]).all()
    assert not output['imputed'][:, -2:].any()
    with pytest.raises(FileExistsError):
        train(path, config(), tmp_path / 'run')


def test_gaze_shift_padding_and_guard_do_not_wrap_or_become_targets():
    motion = np.zeros(12, dtype=bool)
    motion[[4, 7, 10]] = True
    eye = np.zeros(12, dtype=bool)
    eye[[2, 11]] = True
    valid, recoverable, shifted, padding = validity(motion, eye, 10, eye_shift_tr=2, guard_tr=1)
    assert shifted[[0, 1, 4]].all() and not shifted[2]
    assert not recoverable[4] and recoverable[7] and not recoverable[10]
    assert not valid[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].any()
    assert padding[10:].all()


def test_prepare_streamed_roi_preserves_geometry_and_discards_stored_interpolation(tmp_path):
    import gzip
    import zipfile
    import nibabel as nib

    x = np.arange(3 * 4 * 2 * 12, dtype=np.float32).reshape(3, 4, 2, 12)
    affine = np.diag([3., 3., 3., 1.])
    image = nib.Nifti1Image(x, affine)
    image.header.set_xyzt_units('mm', 'sec')
    image.header.set_zooms((3., 3., 3., 2.))
    motion = np.zeros((12, 2), dtype=int)
    motion[4, 0] = motion[7, 1] = 1
    eye = np.zeros((12, 1), dtype=int)
    eye[7] = 1
    with zipfile.ZipFile(tmp_path / 'Movie.zip', 'w') as archive:
        archive.writestr('Movie/preprocessed_native/linear_alignment/s0001_1_1_Z.nii.gz', gzip.compress(image.to_bytes()))
        archive.writestr('Movie/infant_participants.csv', 'ppt,num_TR,prop_TR_eye\ns0001_1_1,10,0.9\n')
        archive.writestr('Movie/motion_confounds/s0001_1_1.txt', '\n'.join(' '.join(map(str, r)) for r in motion))
        archive.writestr('Movie/eye_confounds/s0001_1_1.txt', '\n'.join(str(r[0]) for r in eye))
    ijk = np.array([[0, 0, 0], [2, 3, 1]], dtype=np.int64)
    np.save(tmp_path / 'ijk.npy', ijk)
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps([dict(movie='Movie', ppt='s0001_1_1', family='s0001',
                                           split='train', voxel_ijk='ijk.npy')]))
    output = prepare(selection, tmp_path, tmp_path / 'prepared', eye_shift_tr=0)
    record = load_record(manifest(output)[0]['path'])
    np.testing.assert_allclose(record.coords.numpy(), ijk * 3)
    np.testing.assert_allclose(record.times.numpy(), np.arange(12) * 2)
    assert record.recoverable[:, 4].all() and not record.recoverable[:, 7].any()
    assert not record.valid[:, 10:].any() and not record.recoverable[:, 10:].any()
    assert np.isnan(record.values.numpy()[:, [4, 7, 10, 11]]).all()
    np.testing.assert_array_equal(record.values.numpy()[:, 0], x[ijk[:, 0], ijk[:, 1], ijk[:, 2], 0])


def template_records():
    records = []
    for i in range(3):
        missing = np.zeros(40, dtype=bool)
        missing[:2] = True
        missing[15:18] = True
        missing[-4:] = True
        records.append(dict(id=f'run{i}', family=f'family{i}', split='train', tr_seconds=2.,
                            recoverable=missing.tolist(), eligible=True))
    return records


def test_fitted_profile_keeps_gap_durations_and_boundaries_and_excludes_family():
    rows = template_records()
    profile = fit_profile(rows)
    valid = torch.ones(3, 43, dtype=torch.bool)
    valid[:, 40:] = False
    acquired = torch.arange(43) < 40
    times = torch.arange(43).float()*2
    for seed in range(10):
        a, b, info = draw_profile(valid, times, profile, seed=seed, acquired=acquired, exclude_family='family0')
        c, d, other = draw_profile(valid, times, profile, seed=seed, acquired=acquired, exclude_family='family0')
        assert torch.equal(a, c) and torch.equal(b, d) and info == other
        assert info['donor_family'] != 'family0'
        assert sorted(g['seconds'] for g in info['proposal']['gaps']) == [4., 6., 8.]
        assert info['proposal']['gap_count'] == 3
        assert b[:, :2].all() and b[:, 36:40].all() and not b[:, 40:].any()
        assert not (a & b).any() and torch.equal(a | b, valid)
    # TR conversion preserves physical duration, not frame count.
    _, _, info = draw_profile(torch.ones(1, 80, dtype=torch.bool), torch.arange(80).float(), profile, seed=9)
    assert sorted(g['seconds'] for g in info['proposal']['gaps']) == [4., 6., 8.]


def test_fitted_profile_cannot_fit_validation_and_is_order_invariant():
    rows = template_records()
    assert fit_profile(rows) == fit_profile(rows[::-1])
    with pytest.raises(ValueError, match='training'):
        fit_profile([dict(rows[0], split='val')])
    with pytest.raises(ValueError, match='training'):
        fit_profile([dict(rows[0], eligible=False)])
    families = [f's{i:04d}' for i in range(20)]
    assert family_split(families) == family_split(families[::-1]+families)
    assert sorted(family_split(families).values()).count('train') == 14


def test_fitted_mask_rejects_impossible_layout_and_never_scores_real_invalid():
    profile = fit_profile(template_records())
    with pytest.raises(InsufficientContext):
        draw_profile(torch.ones(1, 4, dtype=torch.bool), torch.arange(4).float()*2, profile, seed=1, max_attempts=4)
    with pytest.raises(InsufficientContext):
        draw_profile(torch.zeros(1, 40, dtype=torch.bool), torch.arange(40).float()*2, profile, seed=1, max_attempts=4)
    valid = torch.ones(2, 40, dtype=torch.bool)
    valid[:, [0, 4, 9, 39]] = False
    observed, target, info = draw_profile(valid, torch.arange(40).float()*2, profile, seed=1)
    assert not target[~valid].any() and not observed[~valid].any()
    assert info['realized']['missing_frames'] < info['proposal']['missing_frames']


def test_long_trailing_metrics_exclude_padding_and_stress_keeps_unknown_out():
    times = torch.arange(30).float()*2
    acquired = torch.arange(30) < 24
    valid = acquired[None].clone()
    observed, target, info = draw_stress(valid, times, acquired=acquired, seconds=32, location='trailing', seed=2)
    assert target[0, 8:24].all() and not target[:, 24:].any()
    prediction = torch.zeros(1, 1, 30)
    truth = torch.ones_like(prediction)
    result = score(prediction, truth, target[None], observed[None], times, acquired=acquired)
    assert result['support']['trailing']['n'] == 16
    assert result['support']['missing_16_32s_one_sided']['n'] == 16
    assert result['support']['missing_32_infs']['n'] == 0
    with pytest.raises(InsufficientContext):
        draw_stress(valid, times, acquired=acquired, seconds=1000, location='leading', seed=2)


def test_stitched_child_play_is_not_accepted_as_continuous_time(tmp_path):
    path, rows = write_records(tmp_path)
    rows[0]['id'] = 'Child_Play:s0001_1_1'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='Child_Play'):
        manifest(path)
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps([dict(movie='Child_Play', ppt='s0001_1_1', family='s0001',
                                           split='train', voxel_ijk='unused.npy')]))
    with pytest.raises(ValueError, match='Child_Play'):
        prepare(selection, tmp_path, tmp_path/'unused', eye_shift_tr=0)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('training_strategy', ['fitted', 'legacy'])
def test_fitted_training_restores_profile_and_reports_stress_separately(tmp_path, training_strategy):
    path, rows = write_records(tmp_path)
    second = tmp_path / 'second_train.npz'
    with np.load(rows[0]['path']) as data:
        np.savez(second, **{k: data[k] for k in data.files})
    rows.append(dict(id='second_train', family='second_family', split='train', path=str(second)))
    path.write_text(json.dumps(rows))
    cfg = dict(config(), mask_strategy=training_strategy, evaluation_mask_strategy='fitted',
               stress_evaluation=dict(seconds=[4, 1000]))
    history = train(path, cfg, tmp_path/'fitted')
    m, saved = restore(tmp_path/'fitted'/'best.pt', 'cpu')
    assert set(m.mask_profile['families']) == {'family_train', 'second_family'}
    assert len((tmp_path/'fitted'/'mask_audit.jsonl').read_text().splitlines()) == 2
    record = load_record(rows[1]['path'])
    result = evaluate(m, [(rows[1], record)], cfg)
    assert result == evaluate(m, [(rows[1], record)], dict(cfg, mask_strategy='legacy'))
    assert result == json.loads((tmp_path/'fitted'/'validation.json').read_text())
    assert result['summary']['stsr']['mse'] == saved['validation_mse']
    assert result['stress']['summary']['1000s_trailing']['methods']['stsr']['n'] == 0
    assert result['stress']['summary']['4s_trailing']['scored_records'] == 1
    profile = profile_from_records([(r, load_record(r['path'])) for r in rows if r['split'] == 'train'])
    assert profile == m.mask_profile
    with pytest.raises(ValueError, match='training'):
        profile_from_records([(rows[1], record)])


def test_legacy_random_only_ablation_does_not_sample_empirical_masks():
    valid = torch.ones(1, 20, dtype=torch.bool)
    empty = [torch.zeros(20, dtype=torch.bool)]
    _, target = draw_holdout(valid, seed=1, empirical=empty, empirical_probability=0)
    assert target.any()
    with pytest.raises(InsufficientContext):
        draw_holdout(valid, seed=1, empirical=empty, empirical_probability=1)


def test_fitted_training_fails_explicitly_with_only_one_donor_family(tmp_path):
    path, _ = write_records(tmp_path)
    with pytest.raises(ValueError, match='two nonempty donor families'):
        train(path, dict(config(), mask_strategy='fitted'), tmp_path/'unsupported')
