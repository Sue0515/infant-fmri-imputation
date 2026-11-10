import json
import pytest
from stsr.imputation.folds import assignments,make_folds
from stsr.imputation.data import manifest


def test_development_assignment_is_order_independent_and_preserves_test():
    rows=[dict(family='s'+str(i),split='train' if i<6 else 'test') for i in range(8)]
    a=assignments(rows,3); assert a==assignments(list(reversed(rows)),3)
    assert set(a)=={'s0','s1','s2','s3','s4','s5'}
    assert sorted(list(a.values()).count(i) for i in range(3))==[2,2,2]
    with pytest.raises(ValueError,match='overlaps'):
        assignments(rows+[dict(family='s0',split='test')],3)


def test_fold_paths_and_family_membership_round_trip(tmp_path):
    rows=[]
    for i in range(7):
        for run in range(2):
            path=tmp_path/f'r{i}_{run}.npz'; path.write_bytes(b'path validation fixture')
            rows.append(dict(id=f'M:s{i}_{run}',family=f's{i}',split='test' if i==6 else 'train',path=path.name))
    source=tmp_path/'manifest.json'; source.write_text(json.dumps(rows))
    result=make_folds(source,tmp_path/'cv',3)
    held_out=[]
    for spec in result['folds']:
        prepared=manifest(tmp_path/'cv'/spec['manifest'])
        assert {r['family'] for r in prepared if r['split']=='test'}=={'s6'}
        val={r['family'] for r in prepared if r['split']=='val'}
        train={r['family'] for r in prepared if r['split']=='train'}
        assert not val&train and len(val)==2
        held_out.extend(val)
    assert sorted(held_out)==['s0','s1','s2','s3','s4','s5']
