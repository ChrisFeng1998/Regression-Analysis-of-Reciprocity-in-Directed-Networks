import numpy as np
import pandas as pd
import CLL
import os

n = 100
a = 0.2
b = 0.1
mu = -a*np.log(n) + 0.5
rho = b*np.log(n) + 0.5
gamma_1 = 1
gamma_2 = 1
theta_0 = [rho, gamma_1,gamma_2]
theta_hat_list = []
coverage_list = []
left_CI_list = []
right_CI_list = []
error_normalize_list = []

N = 1000  # Number of simulations

for _ in range(N):
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

    D = adjacency_matrix
    W = []
    W.append(V_1)
    W.append(V_2)

    result = CLL.tetrad_con_logit(D,W)

    theta_hat = result[0]

    left_CI = result[1]

    right_CI = result[2]

    coverage = [1 if left_CI[i] < theta_0[i] < right_CI[i] else 0 for i in range(len(theta_0))]

    theta_hat_list.append(theta_hat)
    coverage_list.append(coverage)
    left_CI_list.append(left_CI)
    right_CI_list.append(right_CI)
    error_normalize = (theta_hat - theta_0) * 2 * 1.96 / (np.array(right_CI) - np.array(left_CI))
    error_normalize_list.append(error_normalize)

theta_hat_array = np.array(theta_hat_list)
coverage_array = np.array(coverage_list)
left_CI_array = np.array(left_CI_list)
right_CI_array = np.array(right_CI_list)
error_normalize_array = np.array(error_normalize_list)

print(np.mean(coverage_array[:,0]))
print(np.mean(right_CI_array[:,0]) - np.mean(left_CI_array[:,0]))

print(np.mean(coverage_array[:,1]))
print(np.mean(right_CI_array[:,1]) - np.mean(left_CI_array[:,1]))

print(np.mean(coverage_array[:,2]))
print(np.mean(right_CI_array[:,2]) - np.mean(left_CI_array[:,2]))

theta_hat_df = pd.DataFrame(theta_hat_array)
left_CI_df = pd.DataFrame(left_CI_array)
right_CI_df = pd.DataFrame(right_CI_array)
coverage_df = pd.DataFrame(coverage_array)
error_normalize_df = pd.DataFrame(error_normalize_array)

data_concat = pd.concat([theta_hat_df,left_CI_df, right_CI_df,coverage_df,error_normalize_df], axis=1)

SGE_ID = os.getenv('SLURM_JOB_ID')
data_concat.to_csv(f"{SGE_ID}_{n}_{a}_{b}_inference.csv", index=False)