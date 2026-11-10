import numpy as np
import pytest
import torch

from stsr.imputation.data import Record, check_evaluation_split
from stsr.imputation.model import TemporalImputer
from stsr.imputation.runner import evaluate, infer, restore, train
from test_irregular_imputation import config, fixture, model, write_records


class ZeroModel(torch.nn.Module):
    def forward(self, values, observed, coords, times):
        return torch.where(observed, values, torch.zeros_like(values))


def evaluation_records():
    x, valid, coords, times = fixture()
    return [(dict(id=name, family=name, split='val'),
             Record(x[0] + i, valid[0], ~valid[0], coords, times))
            for i, name in enumerate(('family_a:run', 'family_b:run'))]


def indexed(results):
    return {(row['id'], row['seed']): row for row in results['records']}


def test_evaluation_masks_do_not_depend_on_manifest_order():
    records = evaluation_records()
    cfg = dict(config(), eval_masks=3)
    assert indexed(evaluate(ZeroModel(), records, cfg)) == indexed(evaluate(ZeroModel(), records[::-1], cfg))


def test_evaluation_seed_can_be_fixed_across_model_training_seeds():
    records = evaluation_records()
    cfg = dict(config(), evaluation_seed=100, eval_masks=3)
    assert indexed(evaluate(ZeroModel(), records, cfg)) == indexed(evaluate(ZeroModel(), records, dict(cfg, seed=99)))


def test_frozen_test_family_cannot_be_relabelled_as_validation():
    frozen = [dict(id='test_run', family='test_family', split='test', path='/tmp/test_run.npz')]
    changed = [dict(frozen[0], split='val')]
    with pytest.raises(ValueError):
        check_evaluation_split(changed, frozen, 'val')
    # A different run of the same reserved family must also stay out of validation.
    with pytest.raises(ValueError):
        check_evaluation_split([dict(changed[0], id='new_run', path='/tmp/new_run.npz')], frozen, 'val')


def test_excluded_tail_does_not_change_imputation_inside_analysis_interval():
    x, valid, coords, times = fixture()
    base = Record(x[0], valid[0], ~valid[0], coords, times, torch.ones(len(times), dtype=torch.bool))
    extra = 7
    padded = Record(torch.nn.functional.pad(x[0], (0, extra), value=float('nan')),
                    torch.nn.functional.pad(valid[0], (0, extra), value=False),
                    torch.nn.functional.pad(~valid[0], (0, extra), value=False),
                    coords, torch.arange(len(times) + extra).float() * 2,
                    torch.arange(len(times) + extra) < len(times))
    m = model().eval()
    a, b = infer(m, base), infer(m, padded)
    np.testing.assert_allclose(a['values'], b['values'][:, :len(times)], rtol=0, atol=0)
    assert np.isnan(b['values'][:, len(times):]).all()


def test_checkpoint_tracks_model_code_outside_imputation_package(tmp_path):
    path, _ = write_records(tmp_path)
    train(path, config(), tmp_path / 'training')
    _, saved = restore(tmp_path / 'training' / 'best.pt', 'cpu')
    hashes = saved['provenance']['source_sha256']
    assert 'stsr/model/spacetime_attention.py' in hashes
    assert 'stsr/model/inr_head.py' in hashes
    assert 'stsr/utils.py' in hashes


def test_stress_masks_are_stable_across_order_and_training_seed():
    records = evaluation_records()
    cfg = dict(config(), evaluation_seed=11, stress_evaluation=dict(seconds=[4, 8]))
    a = evaluate(ZeroModel(), records, cfg)['stress']['records']
    b = evaluate(ZeroModel(), records[::-1], dict(cfg, seed=99))['stress']['records']
    assert {(r['id'], r['case']): r for r in a} == {(r['id'], r['case']): r for r in b}


def test_train_and_validation_never_encode_excluded_tail(tmp_path, monkeypatch):
    path, _ = write_records(tmp_path)
    original = TemporalImputer.encode
    lengths = []

    def checked(self, values, observed, coords, times):
        lengths.append(len(times))
        return original(self, values, observed, coords, times)

    monkeypatch.setattr(TemporalImputer, 'encode', checked)
    train(path, config(), tmp_path / 'training')
    assert lengths and set(lengths) == {14}


def test_analysis_view_preserves_internal_gaps_and_time_origin():
    x, valid, coords, times = fixture()
    valid[:, :, -2:] = False
    recoverable = ~valid[0].clone()
    recoverable[:, -2:] = False
    record = Record(x[0], valid[0], recoverable, coords, times + 100, torch.arange(16) < 14)
    view = record.analysis_view()
    assert len(view.times) == 14 and len(record.times) == 16
    torch.testing.assert_close(view.times, times[:14] + 100)
    assert not view.valid[:, [2, 3, 10]].any()
    assert view.recoverable[:, [2, 3, 10]].all()


def test_infer_rejects_nonfinite_recovered_values():
    class InvalidModel(torch.nn.Module):
        def forward(self, values, observed, coords, times):
            return torch.full_like(values, float('nan'))

    record = evaluation_records()[0][1]
    with pytest.raises(ValueError, match='nonfinite'):
        infer(InvalidModel(), record)
