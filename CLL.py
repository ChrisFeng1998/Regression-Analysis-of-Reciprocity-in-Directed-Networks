import numpy as np
from scipy.optimize import minimize
import scipy as sp
import pandas as pd
import itertools as it
import numexpr as ne

## Create a dictionary mapping each dyad to the indices of the tetrads it belongs to.
def generate_dyad_to_tetrads_dict(tetrad_list):
    dyad_to_tetrads_dict = {}
    for tetrad_index, [i, j, k, l] in enumerate(tetrad_list):
        for dyad in [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]:
            dyad_to_tetrads_dict.setdefault(dyad, set()).add(tetrad_index)

    return dyad_to_tetrads_dict


## Create the np.array that maps each dyad to its indices within each dyad.
def generate_tetrad_indices_directed(N, full_set=False):
    tetrad_list = np.asarray(list(it.combinations(range(0, N), 4)), dtype='int')

    ij = np.ravel_multi_index([tetrad_list[:, 0], tetrad_list[:, 1]], (N, N))
    ji = np.ravel_multi_index([tetrad_list[:, 1], tetrad_list[:, 0]], (N, N))

    ik = np.ravel_multi_index([tetrad_list[:, 0], tetrad_list[:, 2]], (N, N))
    ki = np.ravel_multi_index([tetrad_list[:, 2], tetrad_list[:, 0]], (N, N))

    il = np.ravel_multi_index([tetrad_list[:, 0], tetrad_list[:, 3]], (N, N))
    li = np.ravel_multi_index([tetrad_list[:, 3], tetrad_list[:, 0]], (N, N))

    jk = np.ravel_multi_index([tetrad_list[:, 1], tetrad_list[:, 2]], (N, N))
    kj = np.ravel_multi_index([tetrad_list[:, 2], tetrad_list[:, 1]], (N, N))

    jl = np.ravel_multi_index([tetrad_list[:, 1], tetrad_list[:, 3]], (N, N))
    lj = np.ravel_multi_index([tetrad_list[:, 3], tetrad_list[:, 1]], (N, N))

    kl = np.ravel_multi_index([tetrad_list[:, 2], tetrad_list[:, 3]], (N, N))
    lk = np.ravel_multi_index([tetrad_list[:, 3], tetrad_list[:, 2]], (N, N))

    tetrad_to_dyads_indices = np.column_stack((ij, ji, ik, ki, il, li, jk, kj, jl, lj, kl, lk))

    if full_set:
        dyad_to_tetrads_dict = generate_dyad_to_tetrads_dict(tetrad_list)
    else:
        dyad_to_tetrads_dict = None

    return [tetrad_to_dyads_indices, dyad_to_tetrads_dict]


## Prepare the data for the conditional likelihood function, identifying the relevant tetrads for identification.
def organize_data_tetrad_logit(D, W, dtcon=None):
    # Calculate three types of configurations defined in the paper
    def calc_S(D_ij, D_ji, D_jk, D_kj, D_kl, D_lk, D_li, D_il):

        Configuration1 = 'D_ij*(1-D_ji)*D_jk*(1-D_kj)*D_kl*(1-D_lk)*D_li*(1-D_il)'
        Configuration2 = '2*D_ij*(D_ji)*(1-D_jk)*(1-D_kj)*D_kl*(D_lk)*(1-D_li)*(1-D_il)'
        Configuration3 = '3*(1-D_ij)*(1-D_ji)*D_jk*(D_kj)*(1-D_kl)*(1-D_lk)*D_li*(D_il)'

        return ne.evaluate(Configuration1) + ne.evaluate(Configuration2) + ne.evaluate(Configuration3)

    def calc_W_tilde(W_ij,W_kl):

        return ne.evaluate('W_ij+ W_kl')

    N = np.shape(D)[0]  # Number of nodes in the network
    K = len(W)
    Nchoose4 = N * (N - 1) * (N - 2) * (N - 3) // 24  # Number of tetrads in network
    D = D.reshape((-1,))  # Reshape adjacency matrix into a 1-d array (vectorize)


    if dtcon is None:
        dtcon = generate_tetrad_indices_directed(N, full_set=True)

    string = 'ijkl'

    # Generate all permutations
    permutations = list(it.permutations(string))

    # Convert tuples to strings and print
    permutations = [''.join(p) for p in permutations]

    dic = {'ij': 0, 'ji': 1, 'ik': 2, 'ki': 3, 'il': 4, 'li': 5, 'jk': 6, 'kj': 7, 'jl': 8, 'lj': 9, 'kl': 10, 'lk': 11}

    S = np.empty((Nchoose4, 0))

    # Consider all permutations of each tetrad
    for i in range(24):
        index1 = dic[permutations[i][0] + permutations[i][1]]
        index2 = dic[permutations[i][1] + permutations[i][0]]
        index3 = dic[permutations[i][1] + permutations[i][2]]
        index4 = dic[permutations[i][2] + permutations[i][1]]
        index5 = dic[permutations[i][2] + permutations[i][3]]
        index6 = dic[permutations[i][3] + permutations[i][2]]
        index7 = dic[permutations[i][3] + permutations[i][0]]
        index8 = dic[permutations[i][0] + permutations[i][3]]
        S1 = calc_S(D[dtcon[0][:, index1]], D[dtcon[0][:, index2]], D[dtcon[0][:, index3]], D[dtcon[0][:, index4]],
                    D[dtcon[0][:, index5]], D[dtcon[0][:, index6]], D[dtcon[0][:, index7]], D[dtcon[0][:, index8]])
        S = np.column_stack((S, S1))

    tetrads_to_keep = set(S.nonzero()[0])
    tetrad_frac_TL = len(tetrads_to_keep) / Nchoose4
    S = S.reshape((-1, 1), order='F')

    proj_tetrads_dict = {dyad: np.asarray(list(tetrads & tetrads_to_keep), dtype='int') \
                         for dyad, tetrads in dtcon[1].items()}

    proj_tetrads_dict = {dyad: np.ravel([tetrads + k * Nchoose4 for k in range(24)]) \
                         for dyad, tetrads in proj_tetrads_dict.items()}

    W_vec = np.zeros((N ** 2, K))

    for k in range(0, K):
        # Vectorize kth N x N dyad-specific regressor matrix; turn into 1d numpy array
        W_vec[:, k] = W[k].reshape((-1,), order="F")

    W_tilde = np.empty((0, 2*(K+1)))
    for i in range(24):
        index_1 = dic[permutations[i][0] + permutations[i][1]]
        index_2 = dic[permutations[i][2] + permutations[i][3]]
        index_3 = dic[permutations[i][0] + permutations[i][3]]
        index_4 = dic[permutations[i][1] + permutations[i][2]]
        W1 = calc_W_tilde(W_vec[dtcon[0][:, index_1]],W_vec[dtcon[0][:, index_2]])
        W2 = calc_W_tilde(W_vec[dtcon[0][:, index_3]],W_vec[dtcon[0][:, index_4]])
        W_add = np.column_stack((np.ones(len(W1))*2, W1,np.ones(len(W1))*2, W2))
        W_tilde = np.vstack((W_tilde, W_add))

    return [S, W_tilde, tetrads_to_keep, tetrad_frac_TL, proj_tetrads_dict]


# Negative log-likelihood function
def neg_log_vec(theta, D, X):
    n = len(D)
    K = X.shape[1] // 2
    r_1 = np.exp(X[:, :K] @ theta)
    r_2 = np.exp(X[:, K:] @ theta)
    r = r_1 + r_2

    log_likelihood = np.log(1 + r) - [1 if i == 1 else 0 for i in D] * (X[:, :K] @ theta) - [1 if i == -1 else 0 for i
                                                                                             in D] * (X[:, K:] @ theta)
    return np.mean(log_likelihood)

## Score of the negative log-likelihood function
def logit_score_vec(theta, D, X):
    n = len(D)
    K = X.shape[1] // 2
    r_1 = np.exp(X[:, :K] @ theta)
    r_2 = np.exp(X[:, K:] @ theta)
    r_sum = 1 + r_1 + r_2

    weighted_sum = X[:, :K].T @ (r_1 / r_sum) + X[:, K:].T @ (r_2 / r_sum)
    correction = X[:, :K].T @ [1 if i == 1 else 0 for i in D] + X[:, K:].T @ [1 if i == -1 else 0 for i in D]
    score = 1 / n * (weighted_sum - correction)

    return score

## Hessian matrix
def logit_Hessian_vec(theta, D, X):
    n = len(D)
    K = X.shape[1] // 2

    r_1 = np.exp(X[:, :K] @ theta)
    r_2 = np.exp(X[:, K:] @ theta)
    r_sum = (1 + r_1 + r_2) ** 2

    Z = X[:, :K] - X[:, K:]

    grad_2 = ((r_1 / r_sum)[:, None] * X[:, :K]).T @ X[:, :K] + ((r_2 / r_sum)[:, None] * X[:, K:]).T @ X[:, K:]

    second_term = ((r_1 * r_2 / r_sum)[:, None] * Z).T @ Z

    hessian = (1 / n) * (grad_2 + second_term)

    return hessian

## Compute the project of score function
def tetrad_logit_score_proj(dyad_score_components):

    def multiply_AB(A, B):

        return ne.evaluate('A * B')

    if dyad_score_components[0].size:
        # Compute dyad ij's projection contribution by summing scores over all contributing
        # tetrads to which ij belongs.
        proj = multiply_AB(dyad_score_components[0], dyad_score_components[1]).sum(axis=0)
    else:
        K = np.shape(dyad_score_components[1])[1]
        proj = np.zeros((K,))

    return proj


def tetrad_con_logit(D, W):
    tetrad_data = organize_data_tetrad_logit(D, W)
    S, W_tilde = tetrad_data[0], tetrad_data[1]
    g = S.nonzero()

    Y_trim = pd.Series(np.ravel(0 * (S[g[0], :] == 1) + 1 * (S[g[0], :] == 2) - 1 * (S[g[0], :] == 3), order='F'),
                       name='S')
    W_trim = W_tilde[g[0], :]

    theta_init = np.zeros(len(W) + 1)

    ## Solve the point estimator
    theta_solver = minimize(neg_log_vec, theta_init, args=(Y_trim, W_trim), method='Newton-CG', \
                            jac=logit_score_vec, hess=logit_Hessian_vec, \
                            options={'xtol': 1e-6, 'maxiter': 1000, 'disp': False})

    theta_ml = theta_solver.x

    n = len(D)
    N = n * (n - 1) // 2
    K = len(W) + 1
    Nchoose4 = n * (n - 1) * (n - 2) * (n - 3) / 24


    ## Calculate asymptotic variance of the estimator according to Theorem 3 in our paper
    value_c1 = np.exp(W_trim[:, 0:K] @ theta_ml) / (
                1 + np.exp(W_trim[:, 0:K] @ theta_ml) + np.exp(W_trim[:, K:] @ theta_ml))
    value_c2 = np.exp(W_trim[:, K:] @ theta_ml) / (
                1 + np.exp(W_trim[:, 0:K] @ theta_ml) + np.exp(W_trim[:, K:] @ theta_ml))

    score_c1 = sp.sparse.coo_matrix((np.ravel(value_c1),\
                                     (g[0], g[1])), shape=(24 * int(Nchoose4), 1), dtype='float64').tocsr() - \
               sp.sparse.coo_matrix((np.ravel([1 if i == 1 else 0 for i in Y_trim]), \
                                     (g[0], g[1])), shape=(24 * int(Nchoose4), 1), dtype='float64').tocsr()

    score_c2 = sp.sparse.coo_matrix((np.ravel(value_c2),\
                                     (g[0], g[1])), shape=(24 * int(Nchoose4), 1), dtype='float64').tocsr() - \
               sp.sparse.coo_matrix((np.ravel([1 if i == -1 else 0 for i in Y_trim]), \
                                     (g[0], g[1])), shape=(24 * int(Nchoose4), 1), dtype='float64').tocsr()

    proj_tetrads_dict = tetrad_data[4]
    proj_score = np.array([tetrad_logit_score_proj([score_c1[tetrads, :].toarray(), W_tilde[:, 0:K][tetrads, :]]) \
                           for dyad, tetrads in proj_tetrads_dict.items()]) / (N - 2 * (n - 1) + 1) + \
                 np.array([tetrad_logit_score_proj([score_c2[tetrads, :].toarray(), W_tilde[:, K:][tetrads, :]]) \
                           for dyad, tetrads in proj_tetrads_dict.items()]) / (N - 2 * (n - 1) + 1)

    OMEGA_hat = np.cov(proj_score, rowvar=False) * ((N - 1) / (N - K))

    hessian_log = logit_Hessian_vec(theta_ml, Y_trim, W_trim)

    scale_hessian = hessian_log * len(Y_trim) / Nchoose4

    iGAMMA_hat = np.linalg.inv(scale_hessian)

    vcov_beta_TL = 36 * np.dot(np.dot(iGAMMA_hat, OMEGA_hat), iGAMMA_hat) / N

    left_CI = []
    right_CI = []
    for i in range(len(theta_ml)):
        left_CI.append(theta_ml[i] - np.sqrt(vcov_beta_TL[i][i]) * 1.96)
        right_CI.append(theta_ml[i] + np.sqrt(vcov_beta_TL[i][i]) * 1.96)

    print(f"Point estimator (theta_ml): {theta_ml}")
    print(f"95% Confidence Interval lower bound (left_CI): {left_CI}")
    print(f"95% Confidence Interval upper bound (right_CI): {right_CI}")

    return [theta_ml, left_CI, right_CI]








