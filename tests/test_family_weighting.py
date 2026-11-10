import torch
import pytest
from stsr.imputation.runner import family_macro_scores,training_order

def test_family_macro_averages_masks_runs_and_families_in_order():
    rows=[]
    for f,i,errors in [('a','a1',[1.,3.]),('a','a2',[6.]),('b','b1',[10.]*4)]:
        for e in errors: rows.append(dict(id=i,family=f,metrics={'stsr':dict(normalized_mse=e,n=1000 if f=='a' else 1)}))
    macro,families=family_macro_scores(rows,'stsr')
    assert families=={'a':4.,'b':10.} and macro==7.

def test_family_sampling_does_not_weight_family_by_run_count():
    records=[({'family':'a'},None)]+[({'family':'b'},None)]*5
    g=torch.Generator().manual_seed(42); count=0
    for _ in range(1000):
        order=training_order(records,g,'family')
        assert sorted(order)==list(range(6))
        count+=order[0]==0
    assert 450<count<550
    with pytest.raises(ValueError): training_order(records,g,'invalid')
