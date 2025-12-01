import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
import time
import os

# ==========================================
# 1. 数据加载
# ==========================================
data_file = 'newsvendor_simple_data.csv'
config_file = 'data_config.json'

if not os.path.exists(data_file): raise FileNotFoundError("找不到数据文件！")

df = pd.read_csv(data_file)
Demand = df['Demand'].values
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features_Raw = np.hstack((DayC, Time))
lF = Features_Raw.shape[1]

with open(config_file, 'r') as f: config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']
TOTAL_LEN = len(Demand)

print(f"✅ 已加载数据. Features: {lF}")

# ==========================================
# 2. 参数设置
# ==========================================
delay = 3
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
Valfac = np.zeros(lnva)
QfacD = np.zeros(lnva)
muD = np.zeros(lnva)
sigmaD = np.zeros(lnva)
ResOpt = np.zeros(lnva)
Qfac_coefs = np.zeros((lnva, 1 + lF))  # Intercept + Features

print("Processing Scarf Rule...")
start_time = time.time()

for k in range(lnva):
    t = lntr + k
    if k % 100 == 0: print(f"  Step {k}/{lnva}")

    # A. 准备数据
    X_train_raw = Features_Raw[t - lntr: t, :]
    scale_factor = np.max(np.abs(X_train_raw), axis=0)
    scale_factor[scale_factor == 0] = 1.0
    X_train = X_train_raw / scale_factor

    end_idx = t + delay
    if end_idx > TOTAL_LEN: break
    y_train = Demand[t - lntr + delay: end_idx]

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
    QfacD[k] = q_star

    # 记录系数
    Qfac_coefs[k, 0] = model_mean.intercept_
    Qfac_coefs[k, 1:] = model_mean.coef_

    # E. 成本
    if t + delay < TOTAL_LEN:
        actual = Demand[t + delay]
        Valfac[k] = nv_cost(q_star, actual, b, h)

print(f"Finished in {time.time() - start_time:.2f}s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'nv_emerg_scarf_de_{delay}_simple_python.csv'

data_dict = {
    'Decision_Q': QfacD,
    'Cost': Valfac,
    'Mu_Pred': muD,
    'Sigma_Pred': sigmaD,
    'Residual_Sum': ResOpt
}
# 展开系数
for i in range(Qfac_coefs.shape[1]):
    col = 'Coef_Intercept' if i == 0 else f'Coef_Feat_{i}'
    data_dict[col] = Qfac_coefs[:, i]

df_out = pd.DataFrame(data_dict)
df_out.to_csv(output_filename, index=False)
print(f"Saved {output_filename}")