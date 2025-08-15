import pandas as pd
import numpy as np
import CLL

####### Lawyer's advice network

attributes = pd.read_csv("LazegaLawyers/ELattr.dat", header=None, sep=r"\s+")

# Rename columns
attributes.columns = ["seniority", "status", "gender", "office", "years_with_firm",
                      "age", "practice", "law_school"]

# Remove unnecessary columns
attributes = attributes.drop(columns=["seniority"])

attributes['years_with_firm'] = (attributes['years_with_firm'] -np.mean(attributes['years_with_firm'])) / np.std(attributes['years_with_firm'])
#
attributes['age'] = (attributes['age'] -np.mean(attributes['age'])) / np.std(attributes['age'])

# Get the number of rows (n)
n = attributes.shape[0]
# Initialize V matrices
V_1 = np.zeros((n, n), dtype=float)
V_2 = np.zeros((n, n), dtype=float)
V_3 = np.zeros((n, n), dtype=float)
V_4 = np.zeros((n, n), dtype=float)
V_5 = np.zeros((n, n), dtype=float)
V_6 = np.zeros((n, n), dtype=float)
V_7 = np.zeros((n, n), dtype=float)

# Fill V matrices
for i in range(n - 1):
    for j in range(i + 1, n):
        V_1[i, j] = int(attributes.iloc[i, 0] == attributes.iloc[j, 0])  # status
        V_1[j, i] = V_1[i, j]

        V_2[i, j] = int(attributes.iloc[i, 1] == attributes.iloc[j, 1])  # gender
        V_2[j, i] = V_2[i, j]

        V_3[i, j] = int(attributes.iloc[i, 2] == attributes.iloc[j, 2])  # office
        V_3[j, i] = V_3[i, j]

        V_4[i, j] = abs(attributes.iloc[i, 3] - attributes.iloc[j, 3])  # year_with_firm
        V_4[j, i] = V_4[i, j]

        V_5[i, j] = abs(attributes.iloc[i, 4] - attributes.iloc[j, 4])  # age
        V_5[j, i] = V_5[i, j]

        V_6[i, j] = int(attributes.iloc[i, 5] == attributes.iloc[j, 5])  # practice
        V_6[j, i] = V_6[i, j]

        V_7[i, j] = int(attributes.iloc[i, 6] == attributes.iloc[j, 6])  # law_school
        V_7[j, i] = V_7[i, j]

adj_adv = pd.read_csv("LazegaLawyers/ELadv.dat", header=None, sep=" ").values
adj_adv = adj_adv[:, 1:]

D = np.array(adj_adv)
W = []
result_0 = CLL.tetrad_con_logit(D,W)

W = [V_1,V_2,V_3, V_6,V_7]
result = CLL.tetrad_con_logit(D,W)



####### Trade network

Log_of_Gravity = pd.read_excel("trade_network/Log of Gravity.xls")
countrycodes = pd.read_excel("trade_network/countrycodes.xls")
countrycodes.rename(columns={countrycodes.columns[0]: "code"}, inplace=True)

# Calculate GDP per country
gdps = (Log_of_Gravity.groupby("s1_im")["lypim"].mean()
        .reset_index().rename(columns={"s1_im": "code", "lypim": "log_gdp"}))
gdps["importer_GDP"] = np.exp(gdps["log_gdp"])

gdps = countrycodes.merge(gdps, on="code", how="right").iloc[:, [0,1,5]]

# Compute total imports and exports
Total_import = Log_of_Gravity.groupby("s1_im")["trade"].sum().reset_index().rename(columns={"trade": "Total_import"})
Total_export = Log_of_Gravity.groupby("s2_ex")["trade"].sum().reset_index().rename(columns={"trade": "Total_export"})
Total_import_export = (Total_import.merge(Total_export, left_on="s1_im", right_on="s2_ex", how="right")
                       .merge(countrycodes, left_on="s1_im", right_on="code", how="right"))
Total_import_export.rename(columns={"s1_im": "code"}, inplace=True)
Total_import_export["total_volume"] = Total_import_export["Total_import"] + Total_import_export["Total_export"]
Total_import_export["code"] = Total_import_export["code"].astype(str)
Total_import_export = Total_import_export.iloc[:, [0,5,1,3,8]]

# Construct network data
Network_data = (Log_of_Gravity[['s1_im', 's2_ex', 'trade']]
                .merge(countrycodes[['code', 'country']], left_on="s1_im", right_on="code", how="right")
                .rename(columns={"country": "importer_country"})
                .merge(countrycodes[['code', 'country']], left_on="s2_ex", right_on="code", how="right")
                .rename(columns={"country": "exporter_country"})
                .merge(Total_import, on="s1_im", how="right")
                .merge(Total_export, on="s2_ex", how="right"))
Network_data= Network_data.iloc[:, [0,1,4,6,2,7,8]]

Network_data["ex_to_im"] = (Network_data["trade"] >= 0.01 * Network_data["Total_import"]).astype(int)
Network_data["im_to_ex"] = (Network_data["trade"] >= 0.01 * Network_data["Total_export"]).astype(int)

adj_matrix = Network_data.pivot_table(index='s1_im', columns='s2_ex', values='im_to_ex', fill_value=0)
adj_matrix_export = adj_matrix.to_numpy()
np.sum(adj_matrix_export)

covariates = pd.DataFrame(columns=["s1_im", "s2_ex", "ldist", "border", "comlang", "colony", "comfrt"])
n = len(countrycodes)
for i in range(n - 1):
    if i % 50 == 0:
        print(i)
    for j in range(i + 1, n):
        country_i, country_j = countrycodes.loc[i, "code"], countrycodes.loc[j, "code"]
        subset = Log_of_Gravity[(Log_of_Gravity["s1_im"] == country_i) & (Log_of_Gravity["s2_ex"] == country_j)]
        covariates = pd.concat([covariates, subset], ignore_index=True)
covariates = covariates.iloc[:, :7]

covariates["ldist"] = (covariates["ldist"] - covariates["ldist"].mean()) / covariates["ldist"].std()

D = np.array(adj_matrix_export)
n = len(D[0])

V_1 = np.zeros((n, n), dtype=float)
V_2 = np.zeros((n, n), dtype=float)
V_3 = np.zeros((n, n), dtype=float)
V_4 = np.zeros((n, n), dtype=float)
V_5 = np.zeros((n, n), dtype=float)
V_6 = np.zeros((n, n), dtype=float)

for i in range(n - 1):
    for j in range(i + 1, n):
        edge_index = int((j - i - 1) + (i) * (2 * n - i -1) / 2)
        V_1[i, j] = covariates.iloc[edge_index,2]   # log_dist
        V_1[j, i] = V_1[i, j]

        V_2[i, j] = covariates.iloc[edge_index,3]  # border
        V_2[j, i] = V_2[i, j]

        V_3[i, j] = covariates.iloc[edge_index,4]  # com_lang
        V_3[j, i] = V_3[i, j]

        V_4[i, j] = covariates.iloc[edge_index,5]  # colony
        V_4[j, i] = V_4[i, j]

        V_5[i, j] = covariates.iloc[edge_index,6] # comfrt
        V_5[j, i] = V_5[i, j]

W = []
result_0 = CLL.tetrad_con_logit(D,W)

W = [V_1, V_2, V_3, V_4, V_5]
result = CLL.tetrad_con_logit(D, W)

















