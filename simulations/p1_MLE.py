"""Full MLE with unrestricted node effects.

This is the covariate-dependent reciprocity model, not the constant-reciprocity p1 model.
"""
import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def generate_network(n=50, a=0.2, b=-0.1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if n < 3:
        raise ValueError('n must be at least 3')
    beta = rng.uniform(0, 1, n)
    beta -= beta[-1]
    alpha = rng.uniform(0, 1, n)
    alpha -= alpha[-1]
    ii, jj = np.triu_indices(n, 1)
    V1, V2 = np.zeros((n, n)), np.zeros((n, n))
    V1[ii, jj] = rng.uniform(-1, 1, len(ii))
    V2[ii, jj] = rng.normal(size=len(ii))
    V1[jj, ii], V2[jj, ii] = V1[ii, jj], V2[ii, jj]
    truth = np.array([.5-a*math.log(n), .5+b*math.log(n), 1., 1.])
    free = np.r_[alpha[:-1], beta[:-1], truth]
    prob = probabilities(free, V1, V2)
    states = (rng.random(len(ii))[:, None] > np.cumsum(prob, axis=1)).sum(axis=1)
    A = np.zeros((n, n), dtype=np.int64)
    A[ii, jj] = np.isin(states, [1, 3])
    A[jj, ii] = np.isin(states, [2, 3])
    return dict(A=A, V1=V1, V2=V2, alpha=alpha, beta=beta,
                truth=truth, probabilities=prob)


def unpack(theta, n):
    theta = np.asarray(theta, float)
    if theta.shape != (2*n+2,):
        raise ValueError(f'Expected {2*n+2} free parameters')
    return np.r_[theta[:n-1], 0.], np.r_[theta[n-1:2*n-2], 0.], theta[-4:]


def raw_weights(theta, V1, V2):
    n = len(V1)
    alpha, beta, (mu, rho, gamma1, gamma2) = unpack(theta, n)
    ij = mu + alpha[:, None] + beta[None, :]
    with np.errstate(over='ignore', invalid='ignore'):
        w10 = np.exp(ij)
        w01 = np.exp(ij.T)
        w11 = np.exp(ij+ij.T+rho+gamma1*V1+gamma2*V2)
    for array in (w10, w01, w11):
        np.fill_diagonal(array, 0.)
    denominator = 1.+w10+w01+w11
    return w10, w01, w11, denominator


def probabilities(theta, V1, V2):
    n = len(V1)
    alpha, beta, (mu, rho, gamma1, gamma2) = unpack(theta, n)
    i, j = np.triu_indices(n, 1)
    e10 = mu+alpha[i]+beta[j]
    e01 = mu+alpha[j]+beta[i]
    logits = np.column_stack([np.zeros(len(i)), e10, e01,
        e10+e01+rho+gamma1*V1[i, j]+gamma2*V2[i, j]])
    return np.exp(logits-logsumexp(logits, axis=1)[:, None])


def negative_loglikelihood(theta, A, V1, V2):
    n = len(A)
    alpha, beta, (mu, rho, gamma1, gamma2) = unpack(theta, n)
    i, j = np.triu_indices(n, 1)
    e10 = mu+alpha[i]+beta[j]
    e01 = mu+alpha[j]+beta[i]
    logits = np.column_stack([np.zeros(len(i)), e10, e01,
        e10+e01+rho+gamma1*V1[i, j]+gamma2*V2[i, j]])
    state = A[i, j].astype(int)+2*A[j, i].astype(int)
    return float(np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(i)), state]))


def score_and_iteration_hessian(theta, A, V1, V2):
    """Gradient and exact Hessian of the mean negative dyadic log likelihood."""
    A, V1, V2 = np.asarray(A), np.asarray(V1), np.asarray(V2)
    n = len(A)
    m = n*(n-1)/2
    p10, p01, p11, K = raw_weights(theta, V1, V2)
    K2 = K**2
    mutual = np.triu(A*A.T, 1)
    score = np.r_[(-A.sum(1)+((p10+p11)/K).sum(1))[:-1],
                  (-A.sum(0)+((p01+p11)/K).sum(1))[:-1],
                  -A.sum()+.5*((p01+p10+2*p11)/K).sum(),
                  -mutual.sum()+.5*(p11/K).sum(),
                  -(mutual*V1).sum()+.5*(V1*p11/K).sum(),
                  -(mutual*V2).sum()+.5*(V2*p11/K).sum()]/m
    aa = (p11-p01*p10)/K2
    bb = aa.copy()
    ab = (p10+p11)*(1+p01)/K2
    aa_diagonal, bb_diagonal, ab_diagonal = ab.sum(1), ab.sum(0), aa.sum(1)
    np.fill_diagonal(aa, aa_diagonal)
    np.fill_diagonal(bb, bb_diagonal)
    np.fill_diagonal(ab, ab_diagonal)

    a_mu = ((p10+2*p11+p11*p01)/K2).sum(1)
    b_mu = ((p01+2*p11+p11*p10)/K2).sum(1)
    a_rho = p11*(1+p01)/K2
    b_rho = p11*(1+p10)/K2
    ag = np.column_stack([a_mu, a_rho.sum(1), (V1*a_rho).sum(1), (V2*a_rho).sum(1)])
    bg = np.column_stack([b_mu, b_rho.sum(1), (V1*b_rho).sum(1), (V2*b_rho).sum(1)])
    H = np.zeros((2*n+2, 2*n+2))
    s, t = slice(0, n-1), slice(n-1, 2*n-2)
    H[s, s], H[t, t], H[s, t], H[t, s] = aa[:-1, :-1], bb[:-1, :-1], ab[:-1, :-1], ab.T[:-1, :-1]
    H[s, -4:], H[t, -4:] = ag[:-1], bg[:-1]
    H[-4:, s], H[-4:, t] = ag[:-1].T, bg[:-1].T
    Z = np.stack([np.ones_like(V1), V1, V2], axis=-1)

    H[-4, -4] = .5*(((4+p01+p10)*p11+p01+p10)/K2).sum()
    cross = .5*np.einsum('ij,ijk->k', (2+p01+p10)*p11/K2, Z)
    H[-4, -3:], H[-3:, -4] = cross, cross
    H[-3:, -3:] = .5*np.einsum('ij,ijk,ijl->kl', (1+p01+p10)*p11/K2, Z, Z)
    return score, H/m


def true_hessian(theta, A, V1, V2):
    """Compatibility accessor for the corrected Hessian used in default fitting."""
    return score_and_iteration_hessian(theta, A, V1, V2)[1]


def mass_ginv(matrix, tol=1e-5):
    """MASS::ginv cutoff: singular value > max(tol * largest, 0)."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    positive = s > max(tol*s[0], 0.)
    if not positive.any():
        return np.zeros(matrix.T.shape)
    return (vt[positive].T/s[positive]) @ u[:, positive].T


def fit(A, V1, V2, tol=1e-6, max_iter=1000):
    if max_iter < 1 or tol <= 0:
        raise ValueError('max_iter and tol must be positive')
    A, V1, V2 = np.asarray(A, int), np.asarray(V1, float), np.asarray(V2, float)
    n = len(A)
    if A.shape != (n, n) or V1.shape != A.shape or V2.shape != A.shape:
        raise ValueError('A, V1 and V2 must be equally sized square matrices')
    if np.any(np.diag(A)) or not np.isin(A, [0, 1]).all():
        raise ValueError('A must be binary with a zero diagonal')
    theta = np.zeros(2*n+2)
    converged, status, scaled_update = False, 'maximum_iterations', None
    for iteration in range(1, max_iter+1):
        try:
            score, H = score_and_iteration_hessian(theta, A, V1, V2)
            update = mass_ginv(H, tol=1e-5) @ score
        except np.linalg.LinAlgError:
            theta[:] = 0.
            status = 'svd_failure_zero_return_as_in_R'
            break
        scaled_update = float(np.max(np.abs(2*update/n**2)))
        if scaled_update < tol:
            converged, status = True, 'original_R_stopping_rule'
            break  # R returns the current coefficients before applying this update.
        theta -= .1*update
    if not converged:
        warnings.warn(status, RuntimeWarning)
    alpha, beta, common = unpack(theta, n)
    score, _ = score_and_iteration_hessian(theta, A, V1, V2)
    return dict(theta_free=theta, alpha=alpha, beta=beta, common=common,
        full_parameters=np.r_[alpha, beta, common], converged=converged, status=status,
        iterations=iteration, scaled_update=scaled_update,
        score_max=float(np.max(np.abs(score))),
        negative_loglikelihood=negative_loglikelihood(theta, A, V1, V2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--a', type=float, default=.2)
    parser.add_argument('--b', type=float, default=-.1)
    parser.add_argument('--replications', type=int, default=700)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/full_mle'))
    args = parser.parse_args()
    if args.replications < 1:
        parser.error('replications must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir/f'MLE_{args.n}_{args.a:g}_{args.b:g}.csv'
    diagnostic = output.with_name(output.stem+'_diagnostics.csv')
    rng = np.random.default_rng(args.seed)
    diagnostic_exists = diagnostic.exists()
    with output.open('a', newline='') as stream, diagnostic.open('a', newline='') as ds:
        writer, dw = csv.writer(stream), csv.writer(ds)
        if not diagnostic_exists:
            dw.writerow(['replication', 'converged', 'iterations', 'score_max', 'scaled_update', 'status'])
        for rep in range(1, args.replications+1):
            data = generate_network(args.n, args.a, args.b, rng)
            result = fit(data['A'], data['V1'], data['V2'])
            writer.writerow(np.r_[result['common'][1:], data['truth'][1:]])
            dw.writerow([rep, result['converged'], result['iterations'], result['score_max'], result['scaled_update'], result['status']])
            stream.flush(); ds.flush()
            print(f'{rep}/{args.replications}: {result["status"]}', flush=True)
    (args.output_dir/'last_run_config.json').write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         'output_columns': ['est_rho', 'est_gamma1', 'est_gamma2', 'true_rho', 'true_gamma1', 'true_gamma2'],
         'hessian': 'corrected_global_block_and_node_mu_cross_terms',
         'output_appends_as_in_R': True, 'numpy': np.__version__}, indent=2))


if __name__ == '__main__':
    main()
