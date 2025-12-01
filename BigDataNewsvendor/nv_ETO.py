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
data_file = 'newsvendor_simple_data.csv'
config_file = 'data_config.json'

if not os.path.exists(data_file):
    raise FileNotFoundError("找不到数据文件！")

df = pd.read_csv(data_file)
Demand = df['Demand'].values

# 提取特征 (仅使用 DayC 和 Time)
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features = np.hstack((DayC, Time))

with open(config_file, 'r') as f:
    config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']

TOTAL_LEN = len(Demand)
print(f"✅ 已加载数据. Features shape: {Features.shape}")

# ==========================================
# 2. 参数设置
# ==========================================
delay = 3
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
Valfac = np.zeros(lnva)
muD = np.zeros(lnva)
sigmaD = np.zeros(lnva)
QfacD = np.zeros(lnva)

# 记录系数 (Intercept + 2个特征)
coef_history = np.zeros((lnva, 1 + Features.shape[1]))

start_time = time.time()

for i in range(lnva):
    t = lntr + i

    if i % 100 == 0: print(f"Step {i}/{lnva}")

    # A. 准备数据
    X_train_raw = Features[t - lntr: t, :]
    norm_val = np.linalg.norm(X_train_raw, ord=np.inf)
    if norm_val == 0: norm_val = 1.0
    X_train = X_train_raw / norm_val

    end_idx = t + delay
    if end_idx > TOTAL_LEN: break
    y_train = Demand[t - lntr + delay: end_idx]

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
    QfacD[i] = optimal_Q

    # F. 验证
    if t + delay < TOTAL_LEN:
        actual_demand = Demand[t + delay]
        Valfac[i] = nv_cost(optimal_Q, actual_demand, b, h)

    # 记录系数
    coef_history[i, 0] = model_mean.intercept_
    coef_history[i, 1:] = model_mean.coef_

print(f"Loop finished in {time.time() - start_time:.2f} s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'nv_emerg_estopt_os_{delay}_simple_python.csv'

# 基础结果
data_dict = {
    'Decision_Q': QfacD,
    'Cost': Valfac,
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