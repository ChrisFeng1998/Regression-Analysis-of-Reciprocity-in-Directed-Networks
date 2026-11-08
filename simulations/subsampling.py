"""Subsampling-estimator simulations
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from estimators.CLL import (organize_data_tetrad_logit, neg_log_vec,
                 logit_score_vec, logit_Hessian_vec)

ROOT = Path(__file__).resolve().parents[1]
REGIMES = [(.3, .5), (.2, .1), (.2, -.1)]
K_VALUES = [2, 5, 10]
PARAMETERS = ["rho", "gamma1", "gamma2"]

def generate_network(n, a, b):
    mu = -a*np.log(n) + 0.5
    rho = b*np.log(n) + 0.5
    gamma_1 = 1
    gamma_2 = 1
    beta = 1 * np.random.uniform(0, 1, n)
    beta = beta - np.mean(beta)
    alpha = 1 * np.random.uniform(0, 1, n)
    alpha = alpha - np.mean(alpha)
    V_1 = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            V_1[i, j] = np.random.uniform(-1, 1)
            V_1[j, i] = V_1[i, j]

    V_2 = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            V_2[i, j] = np.random.normal(0,1)
            V_2[j, i] = V_2[i, j]

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Fill adjacency matrix based on given probabilities
    for i in range(n - 1):
        for j in range(i + 1, n):
            k_0 = 1 + np.exp(mu + alpha[i] + beta[j]) + np.exp(mu + alpha[j] + beta[i]) + np.exp(
                2 * mu + alpha[i] + alpha[j] + beta[i] + beta[j] + rho + V_1[i, j] * gamma_1 + V_2[i, j] * gamma_2)
            p_00 = 1 / k_0
            p_10 = np.exp(mu + alpha[i] + beta[j]) / k_0
            p_01 = np.exp(mu + alpha[j] + beta[i]) / k_0
            p_11 = np.exp(2 * mu + alpha[i] + alpha[j] + beta[i] + beta[j] + rho + V_1[i, j] * gamma_1 + V_2[i, j] * gamma_2) / k_0

            dyad_ij = np.random.choice(['00', '10', '01', '11'], p=[p_00, p_10, p_01, p_11])

            if dyad_ij == "00":
                adjacency_matrix[i, j] = 0
                adjacency_matrix[j, i] = 0
            elif dyad_ij == "10":
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 0
            elif dyad_ij == "01":
                adjacency_matrix[i, j] = 0
                adjacency_matrix[j, i] = 1
            elif dyad_ij == "11":
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix, [V_1, V_2], np.array([rho, gamma_1, gamma_2])

def tetrad_con_logit(D, W):
    tetrad_data = organize_data_tetrad_logit(D, W)
    S, W_tilde = tetrad_data[0], tetrad_data[1]
    g = S.nonzero()

    Y_trim = pd.Series(np.ravel(0 * (S[g[0], :] == 1) + 1 * (S[g[0], :] == 2) - 1 * (S[g[0], :] == 3), order='F'),
                       name='S')
    W_trim = W_tilde[g[0], :]

    theta_init = np.zeros(len(W) + 1)

    theta_solver = minimize(neg_log_vec, theta_init, args=(Y_trim, W_trim), method='Newton-CG', \
                            jac=logit_score_vec, hess=logit_Hessian_vec, \
                            options={'xtol': 1e-6, 'maxiter': 1000, 'disp': False})

    theta_ml = theta_solver.x

    return theta_ml


def simulate(args):
    n = args.n
    m = n//2
    if args.replications < 1 or args.start < 0:
        raise ValueError("replications must be positive; start is a zero-based network index")
    a, b = REGIMES[args.regime-1]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "subsampling_estimates.csv"
    subset_path = out / "subset_fits.csv"
    if path.exists() or subset_path.exists():
        raise FileExistsError("Choose an empty --out directory to avoid overwriting results")
    fields = ["n", "m", "regime", "a", "b", "replication", "seed", "K", *PARAMETERS, "fit_seconds"]
    subset_fields = ["n", "m", "regime", "replication", "seed", "subset", "nodes", *PARAMETERS, "fit_seconds"]
    with path.open("w", newline="") as file, subset_path.open("w", newline="") as subsets:
        writer, subwriter = csv.DictWriter(file, fields), csv.DictWriter(subsets, subset_fields)
        writer.writeheader(); subwriter.writeheader()
        for rep in range(args.start, args.start+args.replications):
            seed = 2026102200+100000*args.regime+100*n+rep
            np.random.seed(seed)
            A, W, truth = generate_network(n, a, b)
            estimates, seconds = [], []
            for draw in range(10):
                indices = np.random.choice(n, m, replace=False)
                D = A[np.ix_(indices, indices)]
                V = [v[np.ix_(indices, indices)] for v in W]
                began = time.perf_counter()
                theta = tetrad_con_logit(D, V)
                elapsed = time.perf_counter()-began
                if not np.isfinite(theta).all():
                    raise RuntimeError(f"Nonfinite estimate: network {rep}, subset {draw+1}")
                estimates.append(theta); seconds.append(elapsed)
                subwriter.writerow(dict(n=n,m=m,regime=args.regime,replication=rep+1,
                    seed=seed,subset=draw+1,nodes=json.dumps(indices.tolist()),
                    **dict(zip(PARAMETERS,theta)),fit_seconds=elapsed))
                subsets.flush()
            for k in K_VALUES:
                estimate = np.mean(estimates[:k], axis=0)
                writer.writerow(dict(n=n,m=m,regime=args.regime,a=a,b=b,replication=rep+1,
                    seed=seed,K=k,**dict(zip(PARAMETERS,estimate)),fit_seconds=sum(seconds[:k])))
            if args.include_full:
                began=time.perf_counter(); theta=tetrad_con_logit(A,W)
                writer.writerow(dict(n=n,m=n,regime=args.regime,a=a,b=b,replication=rep+1,
                    seed=seed,K=0,**dict(zip(PARAMETERS,theta)),fit_seconds=time.perf_counter()-began))
            file.flush()
            print(f"n={n}, regime={args.regime}, network={rep+1}: ten subsets finished", flush=True)
    return path


def plot_errors(errors, n, regime, upper, out):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()
    import matplotlib.pyplot as plt
    n_values = K_VALUES
    a,b=REGIMES[regime-1]
    for j, (parameter,label) in enumerate(zip(PARAMETERS,
            [r"$\widehat{\rho}_n$",r"$\widehat{\gamma}_1$",r"$\widehat{\gamma}_2$"])):
        errors_method1=[errors[0][:,j]]*3
        errors_method2=[errors[k][:,j] for k in K_VALUES]
        positions1 = np.arange(len(n_values)) * 2.0       
        positions2 = positions1 + 0.6                       

        fig, ax = plt.subplots(figsize=(10, 6))

        # Boxplot for method 1
        bp1 = ax.boxplot(errors_method1, positions=positions1, widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', color='blue'),
                         medianprops=dict(color='darkblue'))

        # Boxplot for method 2 (MLE)
        bp2 = ax.boxplot(errors_method2, positions=positions2, widths=0.5, patch_artist=True,
                         boxprops=dict(facecolor='lightgreen', color='green'),
                         medianprops=dict(color='darkgreen'))

        
        ax.set_xticks(positions1 + 0.3)
        ax.set_xticklabels(n_values, fontsize=20)
        ax.set_xlabel('Number of subsamples, K', fontsize=22)
        ax.set_ylabel('Absolute error', fontsize=22)
        ax.tick_params(axis='y', labelsize=20)

        
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Full conditional likelihood', 'Subsampling average'], fontsize=15, loc='upper right')
        ax.set_title(label+rf'; $n={n},\ a={a},\ b={b}$', fontsize=16, pad=12)
        ax.set_ylim(-.02,upper[j])
        fig.tight_layout()
        fig.savefig(out/f"subsampling_n{n}_a{a}_b{b}_{parameter}.pdf")
        plt.close(fig)


def summary(args):
    if args.input is None:raise ValueError('summary requires --input with your generated estimates CSV')
    frame=pd.read_csv(args.input);out=args.out;out.mkdir(parents=True,exist_ok=True)
    records=[]
    for (n,regime),group in frame.groupby(['n','regime']):
        n,regime=int(n),int(regime);a,b=REGIMES[regime-1]
        truth=np.array([.5+b*np.log(n),1.,1.]);errors={}
        for k,values in group.groupby('K'):
            assert not values.replication.duplicated().any()
            error=np.abs(values[PARAMETERS].to_numpy(float)-truth);errors[int(k)]=error
            for j,p in enumerate(PARAMETERS):
                records.append(dict(n=n,regime=regime,K=int(k),parameter=p,N=len(values),
                                    MAE=error[:,j].mean(),total_seconds=values.fit_seconds.sum()))
        if not args.no_plots and all(k in errors for k in [0,*K_VALUES]):
            upper=[math.ceil(max(e[:,j].max() for e in errors.values())*1.12/.05)*.05 for j in range(3)]
            plot_errors(errors,n,regime,upper,out)
    pd.DataFrame(records).to_csv(out/'subsampling_summary.csv',index=False)
    print(f'Summary written to {out}; full-versus-subsample plots require K=0,2,5,10 in your input.')


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["summary","simulate"],nargs="?")
    parser.add_argument("--n",type=int,choices=[100,200])
    parser.add_argument("--regime",type=int,choices=[1,2,3])
    parser.add_argument("--replications",type=int,default=1)
    parser.add_argument("--start",type=int,default=0,help="zero-based network index; official n=200 indices 0..999")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--input",type=Path,help="summarize a newly generated subsampling_estimates.csv")
    parser.add_argument("--no-plots",action="store_true")
    parser.add_argument("--include-full",action="store_true",help="also refit full-network point CL; expensive at n=200")
    args=parser.parse_args()
    if args.command is None:
        parser.print_help()
        parser.exit()
    args.out=args.out or ROOT/"outputs"/("subsampling_summary" if args.command=="summary" else "subsampling_"+args.command)
    if args.command=="summary" and args.input is None:parser.error("summary requires --input")
    if args.command=="summary":summary(args)
    else:
        if args.command=="simulate" and (args.n is None or args.regime is None):
            parser.error("simulate requires --n and --regime")
        simulate(args)
