
import time
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp

RIDGE_GRID = np.logspace(-5, 0, 6)
CONDITIONAL_RIDGE_GRID = np.array([0.,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])


def observation_mask(data, train):
    n = len(data["A"])
    obs = np.zeros((n, n), bool)
    i, j = data["i"][train], data["j"][train]
    obs[i, j] = obs[j, i] = True
    return obs


@njit(cache=True)
def _configuration(a, i, j):
    if a[i, j] == 1:
        return 2 if a[j, i] == 1 else 1
    return 3 if a[j, i] == 0 else 0


@njit(cache=True)
def _perimeter_rows(A, O, V, fill, x1, x2, y):
    """Exactly the original 24-permutation sum, without materializing n choose 4."""
    n, q = A.shape[0], V.shape[2]
    count = 0
    for i in range(n):
        for j in range(n):
            if i == j or not O[i, j]:
                continue
            state = _configuration(A, i, j)
            if state == 0:
                continue
            next_state = 1 if state == 1 else (3 if state == 2 else 2)
            for k in range(n):
                if k == i or k == j or not O[j, k]:
                    continue
                if _configuration(A, j, k) != next_state:
                    continue
                for l in range(n):
                    if l == i or l == j or l == k or not O[k, l] or not O[l, i]:
                        continue
                    if _configuration(A, k, l) != state or _configuration(A, l, i) != next_state:
                        continue
                    if fill:
                        x1[count, 0] = 2.0
                        x2[count, 0] = 2.0
                        for v in range(q):
                            x1[count, v+1] = V[i, j, v] + V[k, l, v]
                            x2[count, v+1] = V[i, l, v] + V[j, k, v]
                        y[count] = state - 1
                    count += 1
    return count


def conditional_design(data, train):
    start = time.monotonic()
    obs = observation_mask(data, train)
    A = data["A"].copy()
    A[~obs] = -1
    q = data["V"].shape[2] + 1
    dummy = np.empty((0, q))
    count = _perimeter_rows(A, obs, data["V"], False, dummy, dummy, np.empty(0, np.int8))
    if count == 0:
        raise ValueError("No identifying observed-perimeter tetrad contributions.")
    x1, x2 = np.empty((count, q)), np.empty((count, q))
    y = np.empty(count, np.int8)
    assert _perimeter_rows(A, obs, data["V"], True, x1, x2, y) == count
    unique, weights = np.unique(np.column_stack([x1, x2, y]), axis=0, return_counts=True)
    return {"x1": unique[:, :q], "x2": unique[:, q:2*q], "y": unique[:, -1].astype(int),
            "weights": weights.astype(float), "original_contributions": count,
            "configuration_counts": np.bincount(y, minlength=3).tolist(),
            "design_seconds": time.monotonic()-start}


def conditional_objective(theta, design, p1=False, hessian=False):
    k = 1 if p1 else len(theta)
    x1, x2 = design["x1"][:, :k], design["x2"][:, :k]
    logits = np.column_stack([np.zeros(len(x1)), x1 @ theta, x2 @ theta])
    lp = logits - logsumexp(logits, axis=1)[:, None]
    p = np.exp(lp)
    w = design["weights"] / design["weights"].sum()
    target = design["y"]
    v1 = p[:, 1] - (target == 1)
    v2 = p[:, 2] - (target == 2)
    grad = x1.T @ (w*v1) + x2.T @ (w*v2)
    value = -np.sum(w * lp[np.arange(len(target)), target])
    if not hessian:
        return value, grad
    H = (x1.T * (w*p[:, 1]*(1-p[:, 1]))) @ x1
    H += (x2.T * (w*p[:, 2]*(1-p[:, 2]))) @ x2
    cross = (x1.T * (w*p[:, 1]*p[:, 2])) @ x2
    H -= cross + cross.T
    return H


def conditional_penalized_objective(theta,design,p1=False,ridge=0.,hessian=False):
    if hessian:
        return conditional_objective(theta,design,p1,True)+ridge*np.eye(len(theta))
    value,gradient=conditional_objective(theta,design,p1)
    return value+ridge*np.dot(theta,theta)/2,gradient+ridge*theta


def fit_conditional(design, p1=False, ridge=0.):
    k = 1 if p1 else design["x1"].shape[1]
    result = minimize(conditional_penalized_objective, np.zeros(k), args=(design, p1, ridge),
                      jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
    H0 = conditional_objective(result.x, design, p1, True)
    H = H0+ridge*np.eye(k)
    eig = np.linalg.eigvalsh(H)
    grad = float(np.max(np.abs(result.jac)))
    if grad > 1e-6 or eig.min() < 1e-10 or not np.isfinite(result.x).all():
        raise RuntimeError(f"Unstable conditional fit: gradient={grad}, curvature={eig.min()}, message={result.message}")
    return {"theta": result.x, "gradient_max": grad, "hessian_min_eigenvalue": float(eig.min()),
            "unpenalized_hessian_min_eigenvalue":float(np.linalg.eigvalsh(H0).min()),
            "conditional_ridge":float(ridge),
            "iterations": int(result.nit), "objective": float(result.fun),
            "configuration_counts": design["configuration_counts"],
            "original_contributions": design["original_contributions"],
            "design_seconds": design["design_seconds"]}


def state_log_prob(eta1, eta2, r):
    logits = np.column_stack([np.zeros(len(eta1)), eta1, eta2, eta1+eta2+r])
    return logits - logsumexp(logits, axis=1)[:, None]


def node_objective(parameters, data, train, theta, ridge):
    n = len(data["A"])
    i, j = data["i"][train], data["j"][train]
    alpha = parameters[1:n+1] - parameters[1:n+1].mean()
    beta = parameters[n+1:] - parameters[n+1:].mean()
    r = theta[0] + data["V"][i, j, :theta.size-1] @ theta[1:]
    lp = state_log_prob(parameters[0]+alpha[i]+beta[j], parameters[0]+alpha[j]+beta[i], r)
    p = np.exp(lp)
    a, b = data["A"][i, j], data["A"][j, i]
    if not (np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()):
        raise ValueError("Training objective accessed an unobserved outcome.")
    e1, e2 = p[:, 1]+p[:, 3]-a, p[:, 2]+p[:, 3]-b
    m = len(i)
    ga = (np.bincount(i, e1, minlength=n)+np.bincount(j, e2, minlength=n))/m + ridge*alpha/n
    gb = (np.bincount(j, e1, minlength=n)+np.bincount(i, e2, minlength=n))/m + ridge*beta/n
    value = -lp[np.arange(m), a+2*b].mean() + ridge*(alpha@alpha+beta@beta)/(2*n)
    gradient = np.r_[np.mean(e1+e2), ga-ga.mean(), gb-gb.mean()]
    return value, gradient


def fit_nodes(data, train, theta, ridge, initial=None):
    start = time.monotonic()
    n = len(data["A"])
    if initial is None:
        initial = np.zeros(2*n+1)
        density = data["A"][observation_mask(data, train)].mean()
        initial[0] = np.log(density/(1-density))
    result = minimize(node_objective, initial, args=(data, train, theta, ridge),
                      method="L-BFGS-B", jac=True,
                      options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 3000, "maxls": 40})
    if np.max(np.abs(result.jac)) > 2e-6:
        raise RuntimeError(f"Node fit failed: {result.message}, gradient {np.max(np.abs(result.jac))}")
    alpha = result.x[1:n+1]-result.x[1:n+1].mean()
    beta = result.x[n+1:]-result.x[n+1:].mean()
    return {"mu": float(result.x[0]), "alpha": alpha, "beta": beta, "theta": theta.copy(),
            "ridge": float(ridge), "gradient_max": float(np.max(np.abs(result.jac))),
            "iterations": int(result.nit), "seconds": time.monotonic()-start}


def predict_nodes(data, fit, selection=None):
    i, j = data["i"], data["j"]
    if selection is not None:
        i, j = i[selection], j[selection]
    theta = np.asarray(fit["theta"])
    r = theta[0] + data["V"][i, j, :theta.size-1] @ theta[1:]
    a, b = np.asarray(fit["alpha"]), np.asarray(fit["beta"])
    return state_log_prob(fit["mu"]+a[i]+b[j], fit["mu"]+a[j]+b[i], r)


def p15_objective(parameters, data, train, ridge):
    i, j = data["i"][train], data["j"][train]
    X, V = data["X"], data["V"][i, j]
    k = X.shape[1]
    out, inc = parameters[1:1+k], parameters[1+k:1+2*k]
    rho, gamma = parameters[1+2*k], parameters[2+2*k:]
    lp = state_log_prob(parameters[0]+X[i]@out+X[j]@inc,
                        parameters[0]+X[j]@out+X[i]@inc, rho+V@gamma)
    p = np.exp(lp)
    a, b = data["A"][i, j], data["A"][j, i]
    e1, e2, er = p[:, 1]+p[:, 3]-a, p[:, 2]+p[:, 3]-b, p[:, 3]-a*b
    m = len(i)
    grad = np.r_[np.mean(e1+e2), (X[i].T@e1+X[j].T@e2)/m,
                 (X[j].T@e1+X[i].T@e2)/m, er.mean(), V.T@er/m]
    penalty = np.ones(len(parameters))
    penalty[[0, 1+2*k]] = 0
    value = -lp[np.arange(m), a+2*b].mean() + ridge*np.sum(penalty*parameters**2)/2
    return value, grad+ridge*penalty*parameters


def fit_p15(data, train, ridge, initial=None):
    start = time.monotonic()
    k, q = data["X"].shape[1], data["V"].shape[2]
    if initial is None:
        initial = np.zeros(2+2*k+q)
        density = data["A"][observation_mask(data, train)].mean()
        initial[0] = np.log(density/(1-density))
    result = minimize(p15_objective, initial, args=(data, train, ridge),
                      method="L-BFGS-B", jac=True,
                      options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 3000, "maxls": 40})
    if np.max(np.abs(result.jac)) > 2e-6:
        raise RuntimeError(f"p1.5 fit failed: {result.message}")
    return {"parameters": result.x, "ridge": float(ridge), "iterations": int(result.nit),
            "gradient_max": float(np.max(np.abs(result.jac))), "seconds": time.monotonic()-start}


def predict_p15(data, fit, selection=None):
    i, j = data["i"], data["j"]
    if selection is not None:
        i, j = i[selection], j[selection]
    p = np.asarray(fit["parameters"])
    k = data["X"].shape[1]
    out, inc = p[1:1+k], p[1+k:1+2*k]
    return state_log_prob(p[0]+data["X"][i]@out+data["X"][j]@inc,
                         p[0]+data["X"][j]@out+data["X"][i]@inc,
                         p[1+2*k]+data["V"][i, j]@p[2+2*k:])


def scores(data, log_prob, selection):
    i, j = data["i"][selection], data["j"][selection]
    y = data["A"][i, j]+2*data["A"][j, i]
    p = np.exp(log_prob)
    assert np.isfinite(log_prob).all() and np.allclose(p.sum(axis=1), 1, atol=1e-12)
    target = np.eye(4)[y]
    return {"nll": float(-log_prob[np.arange(len(y)), y].mean()),
            "brier": float(np.mean(np.sum((p-target)**2, axis=1))),
            "mutual_brier": float(np.mean((p[:, 3]-(y==3))**2)),
            "dyads": len(y), "mutual_dyads": int((y==3).sum())}
