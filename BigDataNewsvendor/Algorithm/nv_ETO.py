import numpy as np
import pandas as pd
import json
from scipy.stats import norm
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

# 提取特征
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features = np.hstack((DayC, Time))

with open(config_file, 'r') as f:
    config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']

TOTAL_LEN = len(Demand)
print(f"已加载数据. Features shape: {Features.shape}")

# ==========================================
# 2. 参数设置
# ==========================================
b = 2.5 / 3.5
h = 1 / 3.5
r = b / (b + h)


# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环逻辑
# ==========================================
# 初始化
out_of_sample_cost = np.zeros(lnte)
muD = np.zeros(lnte)
sigmaD = np.zeros(lnte)
Q_pred = np.zeros(lnte)

# 记录系数 (Intercept + 2个特征)
coef_history = np.zeros((lnte, 1 + Features.shape[1]))

start_time = time.time()

start_idx = lntr + lnva

for i in range(lnte):
    t = start_idx + i

    if i % 100 == 0: print(f"Step {i}/{lnte}")

    # A. 准备数据
    X_train_raw = Features[t - lntr: t, :]
    norm_val = np.linalg.norm(X_train_raw, ord=np.inf)
    if norm_val == 0: norm_val = 1.0
    X_train = X_train_raw / norm_val

    y_train = Demand[t - lntr: t]

    # B. 均值回归
    model_mean = LinearRegression(fit_intercept=True)
    model_mean.fit(X_train, y_train)

    # C. 方差回归
    residuals = y_train - model_mean.predict(X_train)
    y_train_var = np.log(residuals ** 2 + 1e-8)
    model_var = LinearRegression(fit_intercept=True)
    model_var.fit(X_train, y_train_var)

    # D. 预测
    X_current = Features[t, :].reshape(1, -1) / norm_val
    mu_pred = model_mean.predict(X_current)[0]
    sigma_pred = np.exp(model_var.predict(X_current)[0] / 2)

    muD[i] = mu_pred
    sigmaD[i] = sigma_pred

    # E. 优化
    z_score = norm.ppf(r)
    optimal_Q = mu_pred + sigma_pred * z_score
    Q_pred[i] = optimal_Q

    actual_demand = Demand[t]
    out_of_sample_cost[i] = nv_cost(optimal_Q, actual_demand, b, h)

    # 记录系数
    coef_history[i, 0] = model_mean.intercept_
    coef_history[i, 1:] = model_mean.coef_

print(f"Loop finished in {time.time() - start_time:.2f} s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_ETO.csv'

# 基础结果
data_dict = {
    'Decision_Q': Q_pred,
    'Decision_D': Demand[start_idx:start_idx+lnte],
    'Cost': out_of_sample_cost,
    'Mu_Pred': muD,
    'Sigma_Pred': sigmaD

}
# 添加系数列
for idx in range(coef_history.shape[1]):
    col_name = 'Beta_Intercept' if idx == 0 else f'Beta_Feat_{idx}'
    data_dict[col_name] = coef_history[:, idx]

df_out = pd.DataFrame(data_dict)
df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")