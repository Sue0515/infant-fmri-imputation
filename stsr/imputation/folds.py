"""Development-family CV manifests; preserve every reserved test identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .data import manifest,sha256


def assignments(rows,folds=5,seed=42):
    development={r['family'] for r in rows if r['split'] in ('train','val')}
    reserved={r['family'] for r in rows if r['split']=='test'}
    if development&reserved: raise ValueError('test family overlaps development')
    if not 2<=folds<=len(development): raise ValueError('need at least one development family per fold')
    order=sorted(development,key=lambda family:hashlib.sha256(f'raw-qc-cv-v1:{seed}:{family}'.encode()).hexdigest())
    return {family:i%folds for i,family in enumerate(order)}


def make_folds(input_manifest,output,folds=5,seed=42):
    rows=manifest(input_manifest); groups=assignments(rows,folds,seed)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for fold in range(folds):
        folder=output/('fold_'+str(fold)); folder.mkdir()
        entries=[]
        for row in rows:
            split='test' if row['split']=='test' else ('val' if groups[row['family']]==fold else 'train')
            entries.append(dict(row,split=split,path=os.path.relpath(row['path'],folder)))
        path=folder/'manifest.json'; path.write_text(json.dumps(entries,indent=2)+'\n')
        manifest(path)
        summaries.append(dict(fold=fold,manifest=str(path.relative_to(output)),
             records={s:sum(r['split']==s for r in entries) for s in ('train','val','test')},
             families={s:sorted({r['family'] for r in entries if r['split']==s}) for s in ('train','val','test')}))
    receipt=dict(version=1,seed=seed,source_manifest_sha256=sha256(input_manifest),assignments=groups,
        folds=summaries,scope='Input manifests only; CV models/results and family-level aggregation are not produced.')
    (output/'folds.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--out',required=True)
    p.add_argument('--folds',type=int,default=5); p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(); print(json.dumps(make_folds(args.manifest,args.out,args.folds,args.seed),indent=2))

if __name__=='__main__': main()
