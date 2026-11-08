"""Transitivity sensitivity simulation and degree-conditional closure checks.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from data_analysis.raw_data import load_application
from estimators.transitivity_core import (design, independent_draw, sample, fit_tetrad,
    network_stats, diagnostics, fiber_chain)

ROOT=Path(__file__).resolve().parents[1]
# Fixed experimental choices; no fitted parameters or Monte Carlo outputs.
SETTINGS = {'seed': 20261021,
 'transitivity_cases': [{'n': 100, 'delta': 0.0, 'mu_shift': 0.0, 'burn_sweeps': 256},
                        {'n': 100, 'delta': 0.5, 'mu_shift': -0.44140625, 'burn_sweeps': 256},
                        {'n': 100, 'delta': 1.0, 'mu_shift': -0.8896484375, 'burn_sweeps': 256},
                        {'n': 200, 'delta': 0.0, 'mu_shift': 0.0, 'burn_sweeps': 256},
                        {'n': 200, 'delta': 0.5, 'mu_shift': -0.39990234375, 'burn_sweeps': 256},
                        {'n': 200, 'delta': 1.0, 'mu_shift': -0.8055419921875, 'burn_sweeps': 256}],
 'applications': {'lawyer': {'draws_per_chain': 4096, 'mutual_proposal': False},
                  'trade': {'draws_per_chain': 16384, 'mutual_proposal': True}}}
PARAMETERS=["rho","gamma1","gamma2"]


def sensitivity_summary(frame, out):
    records=[]
    for (n,delta),group in frame.groupby(["n","delta"],sort=True):
        ok=group.loc[group.status=="ok"]
        if len(ok)==0:
            raise RuntimeError(f"No successful fits for n={n}, delta={delta}")
        for p in PARAMETERS:
            error=ok[p]-ok[p+"_true"]
            covered=(ok[p+"_lower"]<=ok[p+"_true"]) & (ok[p+"_upper"]>=ok[p+"_true"])
            records.append(dict(parameter=p,n=int(n),delta=delta,N=len(ok),attempted=len(group),
                failed=len(group)-len(ok),bias=error.mean(),bias_mcse=error.std(ddof=1)/np.sqrt(len(ok)),
                SD=ok[p].std(ddof=1),RMSE=np.sqrt(np.mean(error**2)),coverage_percent=100*covered.mean()))
    table=pd.DataFrame(records)
    table.to_csv(out/"transitivity_summary.csv",index=False)
    return table


def closure_summary(draws, out, observed, stem="closure_summary"):
    records=[]
    for name,values in draws.items():
        n,observed_closure=observed[name]
        lo,hi=np.quantile(values,[.025,.975])
        records.append(dict(network=name,n=n,observed=observed_closure,
            simulated_mean=values.mean(),lower_025=lo,upper_975=hi,draws=values.size))
    table=pd.DataFrame(records)
    table.to_csv(out/(stem+".csv"),index=False)
    return table


def simulate(args, settings):
    if args.n is None or args.delta is None:
        raise ValueError("simulate requires --n and --delta")
    if args.replications<1 or args.start<0:
        raise ValueError("Require a positive count and a nonnegative zero-based --start")
    case=next(c for c in settings["transitivity_cases"] if c["n"]==args.n and c["delta"]==args.delta)
    path=args.out/"transitivity_estimates.csv"
    if path.exists():raise FileExistsError("Choose an empty --out to preserve existing results")
    for rep in range(args.start,args.start+args.replications):
        d=design(args.n,settings["seed"]+1000000+args.n*10000+rep)
        seed=settings["seed"]+3000000+args.n*10000+int(args.delta*100)*1000+rep
        row=dict(n=args.n,delta=args.delta,replication=rep,mu_shift=case["mu_shift"],
                 design_seed=d["seed"],sampler_seed=seed)
        row.update({p+"_true":v for p,v in zip(PARAMETERS,d["theta"])})
        began=time.perf_counter()
        try:
            if args.delta==0:
                A=independent_draw(d,seed)[0]
            else:
                A,_=sample(d,args.delta,case["mu_shift"],seed,draws=32,
                           burn=case["burn_sweeps"],interval=2)
            row["generation_seconds"]=time.perf_counter()-began
            fit=fit_tetrad(A,d["V"],ci=True)
            row.update(status="ok",fit_seconds=fit["seconds"],error="")
            for j,p in enumerate(PARAMETERS):
                row.update({p:fit["theta"][j],p+"_se":fit["se"][j],
                            p+"_lower":fit["lower"][j],p+"_upper":fit["upper"][j]})
        except Exception as exc:
            row.update(status="failed",fit_seconds=np.nan,error=repr(exc))
            row.setdefault("generation_seconds",time.perf_counter()-began)
            for p in PARAMETERS:
                row.update({p:np.nan,p+"_se":np.nan,p+"_lower":np.nan,p+"_upper":np.nan})
        # Fixed column order across success/failure; no network replacement.
        fields=["n","delta","replication","mu_shift","design_seed","sampler_seed",
                "generation_seconds","fit_seconds","status","error"]
        for p in PARAMETERS:fields += [p+"_true",p,p+"_se",p+"_lower",p+"_upper"]
        pd.DataFrame([row],columns=fields).to_csv(path,mode="a",header=not path.exists(),index=False)
        print(f"n={args.n}, delta={args.delta:g}, network={rep}: {row['status']}",flush=True)
    sensitivity_summary(pd.read_csv(path),args.out)


def application(args, settings):
    selected=[args.dataset] if args.dataset else ["lawyer","trade"]
    closure={};observed={}
    for name in selected:
        data=load_application(name,args.raw_dir)
        A,V=data['A'],data['V']
        # Refit full-network reciprocity in memory; node effects cancel on the fiber.
        theta=fit_tetrad(A,V,ci=False)['theta']
        observed[name]=(len(A),float(network_stats(A)[-1]))
        config=settings['applications'][name]
        N=len(A)*(len(A)-1)//2;enhanced=config["mutual_proposal"]
        initial=[]
        for chain,multiplier in enumerate([0.,.5,1.5,2.]):
            initial.append(fiber_chain(A,V,theta*multiplier,1,30*N,1,8300+chain,False,enhanced)[0])
        count=config["draws_per_chain"]
        while count<=65536:
            traces=[]
            for chain in range(4):
                _,trace,_,_=fiber_chain(initial[chain],V,theta,count,
                    50*N,max(512,N),9400+chain,False,enhanced)
                traces.append(trace)
            traces=np.stack(traces)
            names=["mutual"]+[f"weighted_mutual_{j+1}" for j in range(V.shape[2])]+["T_normalized","closure_rate"]
            diag=diagnostics(traces,names)
            passed=all(r["rhat"] is not None and r["rhat"]<1.01 and
                       r["bulk_ess"]>=400 and r["tail_ess"]>=200 for r in diag)
            (args.out/f"closure_diagnostics_{name}_{count}.json").write_text(json.dumps(diag,indent=2)+"\n")
            print(f"{name}: {count} draws/chain, diagnostic gate passed={passed}",flush=True)
            if passed:break
            count*=2
        if not passed:raise RuntimeError(f"{name}: diagnostics failed; no accepted closure table generated")
        closure[name]=traces[:,:,-1]
    closure_summary(closure,args.out,observed)




if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["summary","simulate","application"],nargs="?")
    parser.add_argument("--n",type=int,choices=[100,200])
    parser.add_argument("--delta",type=float,choices=[0.,.5,1.])
    parser.add_argument("--start",type=int,default=0,help="zero-based network index")
    parser.add_argument("--replications",type=int,default=1)
    parser.add_argument("--dataset",choices=["lawyer","trade"])
    parser.add_argument("--raw-dir",type=Path,default=ROOT,help="Directory containing downloaded raw datasets for application checks")
    parser.add_argument("--input",type=Path,help="resummarize a newly generated transitivity_estimates.csv")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit()
    args.out=args.out or ROOT/"outputs"/("transitivity_summary" if args.command=="summary" else "transitivity_"+args.command)
    args.out.mkdir(parents=True,exist_ok=True)
    settings=SETTINGS
    if args.command=="summary":
        if args.input is None:parser.error('summary requires --input')
        sensitivity_summary(pd.read_csv(args.input),args.out)
    elif args.command=="simulate":simulate(args,settings)
    elif args.command=="application":application(args,settings)
