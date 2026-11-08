"""
Parameter order: mu, rho, gamma_X, gamma_Y, delta1[, delta2]. The delta
coefficients are the reciprocity-covariate coefficients. 
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def parameter_names(q):
    return ['mu', 'rho', 'gamma_X', 'gamma_Y']+[f'delta{j+1}' for j in range(q)]


def generate_network(n=50, a=.2, b=-.1, q=2, rng=None):
    if n < 3 or q not in [1, 2]:
        raise ValueError('n >= 3 and q in {1,2} are required')
    rng = np.random.default_rng() if rng is None else rng
    X, U = rng.uniform(-1, 1, n), rng.normal(size=n)
    Y, W = rng.uniform(-1, 1, n), rng.normal(size=n)
    i, j = np.triu_indices(n, 1)
    V = rng.uniform(-1, 1, (len(i), 1))
    if q == 2:
        V = np.column_stack([V[:, 0], rng.normal(size=len(i))])
    truth = np.r_[.5-a*math.log(n), .5+b*math.log(n), 1., 1., np.ones(q)]
    mu, rho, gamma_X, gamma_Y = truth[:4]
    # Effective branch in the R file: latent terms are added after multiplication
    # by gamma_X/gamma_Y; the earlier overwritten probability block is omitted.
    e10 = mu+X[i]*gamma_X+Y[j]*gamma_Y+U[i]+W[j]
    e01 = mu+X[j]*gamma_X+Y[i]*gamma_Y+U[j]+W[i]
    logits = np.column_stack([np.zeros(len(i)), e10, e01,
                             e10+e01+rho+V@truth[4:]])
    log_K = logsumexp(logits, axis=1)
    probs = np.exp(logits-log_K[:, None])
    states = (rng.random(len(i))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    raw = np.column_stack([np.isin(states, [1, 3]), np.isin(states, [2, 3]),
        X[i], X[j], Y[i], Y[j], V, probs[:, 1:], np.exp(log_K)])
    return dict(n=n, X=X, U=U, Y=Y, W=W, V=V, raw_data=raw,
                truth=truth, probabilities=probs)


def design_from_raw(raw, q):
    raw = np.asarray(raw, float)
    if q not in [1, 2] or raw.ndim != 2 or raw.shape[1] != 10+q:
        raise ValueError('Expected original R raw-data columns (11 or 12 columns)')
    m = len(raw)
    features = np.zeros((m, 4, 4+q))
    features[:, 1, :4] = np.column_stack([np.ones(m), np.zeros(m), raw[:, 2], raw[:, 5]])
    features[:, 2, :4] = np.column_stack([np.ones(m), np.zeros(m), raw[:, 3], raw[:, 4]])
    features[:, 3, :4] = np.column_stack([2*np.ones(m), np.ones(m), raw[:, 2]+raw[:, 3], raw[:, 4]+raw[:, 5]])
    features[:, 3, 4:] = raw[:, 6:6+q]
    state = raw[:, 0].astype(int)+2*raw[:, 1].astype(int)
    if not np.isin(raw[:, :2], [0, 1]).all():
        raise ValueError('The first two columns must be binary dyad states')
    return features, state


def likelihood(theta, features, state):
    """Sum negative multinomial log likelihood, its gradient and Hessian."""
    logits = features@np.asarray(theta, float)
    log_K = logsumexp(logits, axis=1)
    prob = np.exp(logits-log_K[:, None])
    value = float(np.sum(log_K-logits[np.arange(len(state)), state]))
    expected = np.einsum('ms,msk->mk', prob, features)
    gradient = np.sum(expected-features[np.arange(len(state)), state], axis=0)
    centered = features-expected[:, None, :]
    H = np.einsum('ms,msk,msl->kl', prob, centered, centered)
    return value, gradient, H, prob


def fit(raw, q=2, iterlim=2000, tol=1e-6, ftol=1e-8, steptol=1e-10):
    if iterlim < 1 or min(tol, ftol, steptol) <= 0:
        raise ValueError('iterlim and stopping tolerances must be positive')
    features, state = design_from_raw(raw, q)
    theta = np.zeros(4+q)
    value, gradient, H, prob = likelihood(theta, features, state)
    evaluated_theta = theta.copy()
    chi2, iteration, code = 1e10, 0, 4
    while True:
        old_value, old_gradient = value, gradient
        # mlogit's default method is nr. A singular solve is an error, not ridge.
        direction = -np.linalg.solve(H, gradient)
        iteration += 1
        if iteration > iterlim:
            code = 4
            break
        step = 1.
        while step >= steptol:
            trial = theta+step*direction
            new_value, new_gradient, new_H, new_prob = likelihood(trial, features, state)
            if np.isfinite(new_value) and new_value <= old_value:
                break
            step *= .5
        if step < steptol:
            code = 3
            break
        value, gradient, H, prob = new_value, new_gradient, new_H, new_prob
        evaluated_theta = trial.copy()
        if abs(value-old_value) < ftol:
            # This precedes the coefficient update in mlogit.optim 1.1-1.
            code = 2
            break
        theta = trial
        chi2 = float(-direction@old_gradient)
        if abs(chi2) < tol:
            code = 1
            break
    return dict(coefficients=theta, evaluated_coefficients=evaluated_theta,
                negative_loglikelihood=float(value),
                coefficient_negative_loglikelihood=likelihood(theta, features, state)[0],
                gradient=gradient, hessian=H, probabilities=prob,
                iterations=iteration, code=code, converged=code in [1, 2],
                decrement=chi2, parameter_names=parameter_names(q))


def _eigenvalue_ratio(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    numerator, denominator = abs(eigenvalues.max()), abs(eigenvalues.min())
    return float(numerator/denominator) if denominator else np.inf


def regularized_inverse_1(matrix, lam=1e-3, threshold=1e3):
    """Unused trailing R helper retained for completeness; not used in fitting."""
    from simulations.p1_MLE import mass_ginv
    matrix = np.asarray(matrix, float)
    if _eigenvalue_ratio(matrix) > threshold:
        matrix = matrix+lam*np.eye(len(matrix))
    return mass_ginv(matrix, tol=np.sqrt(np.finfo(float).eps))


def regularized_inverse_2(matrix, lam=1e-3, threshold=1e3):
    """Unused trailing R helper; the original eigenvalue ratio is preserved."""
    matrix = np.asarray(matrix, float)
    if _eigenvalue_ratio(matrix) > threshold:
        matrix = matrix+lam*np.eye(len(matrix))
    return np.linalg.inv(matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--a', type=float)
    parser.add_argument('--b', type=float)
    parser.add_argument('--replications', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/p1_5_misspecification'))
    args = parser.parse_args()
    if args.replications is not None and args.replications < 1:
        parser.error('replications must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    runs = []
    for setting in ([1, 2] if args.setting == 'both' else [int(args.setting)]):
        q = setting
        a = args.a if args.a is not None else (.3 if q == 1 else .2)
        b = args.b if args.b is not None else (.5 if q == 1 else -.1)
        count = args.replications if args.replications is not None else (20 if q == 1 else 1000)
        filename = (f'p1.5_misspecification_{a:g}_{b:g}_{args.n}.csv' if q == 2 else
                    f'p1.5_misspecification_setting1_{a:g}_{b:g}_{args.n}.csv')
        output = args.output_dir/filename
        detailed = args.output_dir/(Path(filename).stem+'_estimates.csv')
        output_exists, detailed_exists = output.exists(), detailed.exists()
        with output.open('a', newline='') as stream, detailed.open('a', newline='') as ds:
            writer, dw = csv.writer(stream), csv.writer(ds)
            if not output_exists:
                writer.writerow([f'V{i+1}' for i in range(4+q)])
            if not detailed_exists:
                dw.writerow(['replication', *parameter_names(q), 'converged', 'code', 'iterations'])
            all_errors = []
            for rep in range(1, count+1):
                data = generate_network(args.n, a, b, q, rng)
                result = fit(data['raw_data'], q)
                error = np.abs(result['coefficients']-data['truth'])
                writer.writerow(error)
                dw.writerow([rep, *result['coefficients'], result['converged'], result['code'], result['iterations']])
                stream.flush(); ds.flush()
                all_errors.append(error)
                print(f'setting {setting}, {rep}/{count}: code {result["code"]}', flush=True)
            print('MAE:', dict(zip(parameter_names(q), np.mean(all_errors, axis=0))))
        runs.append(dict(setting=setting, n=args.n, a=a, b=b, replications=count,
            error_columns=parameter_names(q), output=filename))
    (args.output_dir/'last_run_config.json').write_text(json.dumps(dict(seed=args.seed,
        runs=runs, optimizer='mlogit 1.1-1 default NR port', numpy=np.__version__,
        setting1_additional_CSV='The R setting1 printed errors but did not save its own CSV.'), indent=2))


if __name__ == '__main__':
    main()
