
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from estimators import mcmc

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ('rho','gamma1','gamma2')
original_start = mcmc.homogeneous_start
recovery_records = []


def numerical_start(A, V):
    # The same optional identical-likelihood recovery used in the completed run.
    try:
        return original_start(A,V)
    except RuntimeError as error:
        if str(error) != 'Homogeneous initializer did not converge':
            raise
        theta, record = mcmc.recover_homogeneous_start(A,V)
        recovery_records.append(record)
        return theta


mcmc.homogeneous_start = numerical_start


def generate_input(row, rscript):
    n = int(row['n'])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'network.bin'
        subprocess.run([rscript, '--vanilla', str(Path(__file__).with_name('mcmc_inputs.R')), str(n),
            str(row['a']), str(row['b']), str(row['data_seed']), str(path)], check=True)
        A = np.fromfile(path, dtype=np.uint8, count=n*n).reshape(n,n).astype(np.int8)
        raw = np.fromfile(path, dtype='<f8', offset=n*n, count=2*n*n)
        V = [raw[:n*n].reshape(n,n).copy(), raw[n*n:].reshape(n,n).copy()]
    return A,V


def summarize(rows,output):
    summaries=[]
    for regime in (1,2,3):
        part=[r for r in rows if int(r['regime'])==regime]
        if not part:continue
        a,b,n=float(part[0]['a']),float(part[0]['b']),int(part[0]['n'])
        truth=[.5+b*np.log(n),1.,1.]
        row=dict(regime=regime,a=a,b=b,N=len(part))
        for j,name in enumerate(PARAMETERS):
            error=np.abs(np.array([r['estimate_'+name] for r in part])-truth[j])
            row['MAE_'+name]=float(error.mean())
        row['mean_seconds']=float(np.mean([r['fit_seconds'] for r in part]))
        for key in ('max_rhat','max_mcse_se','max_residual_se','extension_change_se'):
            row[key]=max(float(r[key]) for r in part)
        summaries.append(row)
    with (output/'MCMC_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)


def refit(row, output, rscript):
    path = output/(row['id']+'.json')
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['passed'] and int(saved['mcmc_seed'])==int(row['mcmc_seed']):
            print('Reusing',row['id'],flush=True)
            return saved
    A,V = generate_input(row,rscript)  # Input generation is outside fit timing.
    recovery_records.clear()
    print('Fitting',row['id'],flush=True)
    fit = mcmc.fit_cmle(A,V,int(row['mcmc_seed']),backend='numba',config=mcmc.FORMAL,
                      progress=lambda text:print(row['id'],text,flush=True))
    result = {key:row[key] for key in ('id','regime','a','b','n','data_seed','mcmc_seed')}
    result.update(passed=bool(fit['pass_diagnostics'] and fit['extension_stable']),
        fit_seconds=fit['wall_seconds'],phase_seconds=fit['phase_seconds'],
        initializer_recovery=list(recovery_records),
        extension_change_se=fit['max_extension_shift_se'],settings=fit['config'])
    if fit['theta'] is not None:
        result.update({'estimate_'+name:float(value) for name,value in zip(PARAMETERS,fit['theta'])})
        sample = fit['extended']['verification']
        construction = fit['extended']['construction']
        result.update(max_rhat=sample['max_rhat'],
            max_mcse_se=float(np.max(sample['projected']['mcse']/sample['se'])),
            max_residual_se=float(np.max(abs(sample['correction']/sample['se']))),
            construction_max_rhat=construction['max_rhat'],
            construction_max_mcse_se=float(np.max(construction['projected']['mcse']/sample['se'])),
            draws_per_chain=sample['draws'],conditional_SE=sample['se'].tolist(),
            raw_ESS=sample['raw']['ess_ar'].tolist(),
            projected_ESS=sample['projected']['ess_ar'].tolist(),
            projected_MCSE=sample['projected']['mcse'].tolist(),
            proposal_counts=sample['counts'].tolist())
    path.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise RuntimeError(f"{row['id']}: diagnostics/stability failed; result saved, graph not replaced")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regime',type=int,choices=[1,2,3])
    parser.add_argument('--replications',type=int,default=1)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--seed',type=int,default=2026102100)
    parser.add_argument('--rscript',default=shutil.which('Rscript'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/mcmc')
    args=parser.parse_args()
    if args.replications<1 or args.start<0:parser.error('Require positive replications and nonnegative start')
    if not args.rscript:parser.error('Rscript is required for the network generator')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    warmup_seconds=mcmc.warmup()
    (args.output_dir/'jit_seconds.txt').write_text(f'{warmup_seconds:.9f}\n')
    fitted=[];regimes=[(.3,.5),(.2,.1),(.2,-.1)]
    for regime in ([args.regime] if args.regime else [1,2,3]):
        a,b=regimes[regime-1]
        for rep in range(args.start,args.start+args.replications):
            index=3*rep+regime
            row=dict(id=f'n200_r{regime}_rep{rep+1:03d}',regime=regime,n=200,a=a,b=b,
                     data_seed=args.seed+index,mcmc_seed=1100000000+10000*index)
            fitted.append(refit(row,args.output_dir,args.rscript))
            summarize(fitted,args.output_dir)


if __name__=='__main__':
    main()
