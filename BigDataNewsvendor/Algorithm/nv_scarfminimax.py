import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import time
import os

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

df = pd.read_csv(data_file)
Demand = df['Demand'].values
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features_Raw = np.hstack((DayC, Time))
lF = Features_Raw.shape[1]

with open(config_file, 'r') as f: config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']
TOTAL_LEN = len(Demand)

print(f"已加载数据. Features: {lF}")

# ==========================================
# 2. 参数设置
# ==========================================
b = 2.5 / 3.5
h = 1 / 3.5



# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环
# ==========================================
out_of_sample_cost = np.zeros(lnte)
decision_q = np.zeros(lnte)
muD = np.zeros(lnte)
sigmaD = np.zeros(lnte)
ResOpt = np.zeros(lnte)
coefs = np.zeros((lnte, 1 + lF))  # Intercept + Features

print("Processing Scarf Rule...")
start_time = time.time()

for k in range(lnte):
    start_idx = lntr + lnva
    t = start_idx + k
    if k % 100 == 0: print(f"  Step {k}/{lnte}")

    # A. 准备数据
    X_train_raw = Features_Raw[t - lntr: t, :]
    scale_factor = np.max(np.abs(X_train_raw), axis=0)
    scale_factor[scale_factor == 0] = 1.0
    X_train = X_train_raw / scale_factor
    y_train = Demand[t - lntr: t]

    # B. 均值回归
    model_mean = LinearRegression(fit_intercept=True)
    model_mean.fit(X_train, y_train)
    residuals = y_train - model_mean.predict(X_train)
    ResOpt[k] = np.sum(residuals ** 2)

    # C. 方差回归
    y_train_var = np.log(residuals ** 2 + 1e-8)
    model_var = LinearRegression(fit_intercept=True)
    model_var.fit(X_train, y_train_var)

    # D. 预测
    X_current = Features_Raw[t, :] / scale_factor
    X_current = X_current.reshape(1, -1)

    mu_val = model_mean.predict(X_current)[0]
    sigma_val = np.exp(model_var.predict(X_current)[0] / 2)

    muD[k] = mu_val
    sigmaD[k] = sigma_val

    # Scarf Rule
    scarf_term = (np.sqrt(b) - 1.0 / np.sqrt(b))
    q_star = mu_val + (sigma_val / 2.0) * scarf_term
    decision_q[k] = q_star

    # 记录系数
    coefs[k, 0] = model_mean.intercept_
    coefs[k, 1:] = model_mean.coef_

    # E. 成本
    if t < TOTAL_LEN:
        actual = Demand[t]
        out_of_sample_cost[k] = nv_cost(q_star, actual, b, h)

print(f"Finished in {time.time() - start_time:.2f}s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_scarf.csv'

data_dict = {
    'Decision_Q': decision_q,
    'Decision_d':Demand[start_idx:start_idx + lnte],
    'Cost': out_of_sample_cost,
    'Mu_Pred': muD,
    'Sigma_Pred': sigmaD,
    'Residual_Sum': ResOpt
}
# 展开系数
for i in range(coefs.shape[1]):
    col = 'Coef_Intercept' if i == 0 else f'Coef_Feat_{i}'
    data_dict[col] = coefs[:, i]

df_out = pd.DataFrame(data_dict)
df_out.to_csv(output_filename, index=False)
print(f"Saved {output_filename}")