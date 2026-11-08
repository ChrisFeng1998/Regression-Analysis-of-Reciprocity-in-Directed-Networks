"""Evaluate R2, p1 and p1.5 using two ten-fold dyad partitions.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from data_analysis.raw_data import load_application
from estimators.assessment_models import (RIDGE_GRID, CONDITIONAL_RIDGE_GRID, conditional_design,
    fit_conditional, fit_nodes, predict_nodes, fit_p15, predict_p15, scores)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('R2', 'p1', 'p15')
METRICS = ('joint_log_loss', 'four_state_brier', 'mutual_brier', 'mutual_auc')


def auc_rank(y, score):
    y = np.asarray(y, bool); score = np.asarray(score, float)
    n1 = int(y.sum()); n0 = len(y)-n1
    assert n1 > 0 and n0 > 0
    return float((rankdata(score, method='average')[y].sum()-n1*(n1+1)/2)/(n1*n0))


def fit_fold(data, train, inner):
    validation, test = train & ~inner, ~train
    result = {}
    candidates = []; initial = None
    for ridge in RIDGE_GRID[::-1]:
        fit = fit_p15(data, inner, ridge, initial)
        initial = fit['parameters']
        loss = scores(data, predict_p15(data, fit, validation), validation)['nll']
        candidates.append((loss, ridge))
    ridge = min(candidates, key=lambda pair:pair[0])[1]
    result['p15'] = np.exp(predict_p15(data, fit_p15(data, train, ridge), test))
    inner_design = conditional_design(data, inner)
    outer_design = conditional_design(data, train)
    for model in ('R2', 'p1'):
        candidates = []
        for cridge in CONDITIONAL_RIDGE_GRID:
            try:
                conditional = fit_conditional(inner_design, p1=model=='p1', ridge=cridge)
            except RuntimeError:
                continue  # Same original eligibility check for unstable fits.
            initial = None
            for nridge in RIDGE_GRID[::-1]:
                fit = fit_nodes(data, inner, conditional['theta'], nridge, initial)
                initial = np.r_[fit['mu'], fit['alpha'], fit['beta']]
                loss = scores(data, predict_nodes(data, fit, validation), validation)['nll']
                candidates.append((loss, float(cridge), float(nridge)))
        failed_outer = set()
        for _, cridge, nridge in sorted(candidates):
            if cridge in failed_outer:
                continue
            try:
                conditional = fit_conditional(outer_design, p1=model=='p1', ridge=cridge)
            except RuntimeError:
                failed_outer.add(cridge)
                continue
            fit = fit_nodes(data, train, conditional['theta'], nridge)
            result[model] = np.exp(predict_nodes(data, fit, test))
            break
        else:
            raise RuntimeError(f'No stable training-only candidate for {model}')
    return result


def fit_predictions(data, name):
    # Every potential dyad, including 00, is predicted exactly once per repeat.
    probabilities = np.full((2, 3, len(data['i']), 4), np.nan)
    for task, (train, inner) in enumerate(zip(data['train_masks'], data['inner_masks'])):
        repeat, fold = divmod(task, 10)
        fitted = fit_fold(data, train, inner)
        for m, model in enumerate(MODELS):
            probabilities[repeat, m, ~train] = fitted[model]
        print(f'{name}: partition {repeat+1}/2, fold {fold+1}/10 complete', flush=True)
    assert np.isfinite(probabilities).all()
    return probabilities


def summarize(data, probabilities, name):
    all_dyads = np.ones(len(data['i']), bool)
    mutual = (data['A'][data['i'], data['j']]+2*data['A'][data['j'], data['i']]) == 3
    records = []
    for repeat in range(2):
        for m, model in enumerate(MODELS):
            p = probabilities[repeat, m]
            s = scores(data, np.log(p), all_dyads)
            records.append(dict(network=name, model=model, partition=repeat+1,
                joint_log_loss=s['nll'], four_state_brier=s['brier'],
                mutual_brier=s['mutual_brier'], mutual_auc=auc_rank(mutual, p[:, 3])))
    means = []
    for model in MODELS:
        group = [r for r in records if r['model']==model]
        means.append(dict(network=name, model=model,
            **{key:float(np.mean([r[key] for r in group])) for key in METRICS}))
    return records, means


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)





def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=ROOT,help='Directory containing the downloaded raw dataset files; see README')
    parser.add_argument('--dataset',choices=['lawyer','trade'],help='Default: both datasets')
    parser.add_argument('--seed',type=int,default=20261108)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/assessment')
    args=parser.parse_args()
    names=[args.dataset] if args.dataset else ['lawyer','trade']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[];means=[]
    for name in names:
        data=load_application(name,args.raw_dir)
        if 'train_masks' not in data:
            # New deterministic partitions; all potential unordered dyads are eligible.
            rng=np.random.default_rng(args.seed);N=len(data['i']);train=[];inner=[]
            for repeat in range(2):
                assignment=np.empty(N,int);assignment[rng.permutation(N)]=np.arange(N)%10
                for fold in range(10):
                    mask=assignment!=fold;within=mask.copy();indices=np.flatnonzero(mask)
                    within[rng.choice(indices,max(1,len(indices)//10),replace=False)]=False
                    train.append(mask);inner.append(within)
            data['train_masks']=np.array(train);data['inner_masks']=np.array(inner)
        p=fit_predictions(data,name)
        per_partition,average=summarize(data,p,name)
        records.extend(per_partition);means.extend(average)
    write_csv(args.output_dir/'metrics.csv',means)
    write_csv(args.output_dir/'partition_metrics.csv',records)
    for row in means:print(row['network'],row['model'],*(f'{row[k]:.5f}' for k in METRICS))


if __name__=='__main__':
    main()
