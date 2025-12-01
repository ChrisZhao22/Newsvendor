import numpy as np
import pandas as pd
import json
import cvxpy as cp
import time
import os

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../newsvendor_simple_data.csv'
config_file = '../data_config.json'

if not os.path.exists(data_file):
    raise FileNotFoundError("找不到数据文件！")

df = pd.read_csv(data_file)
Demand = df['Demand'].values
# 特征构建
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features_Raw = np.hstack((DayC, Time))
lF = Features_Raw.shape[1]

with open(config_file, 'r') as f:
    config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']
TOTAL_LEN = len(Demand)

print(f"✅ 已加载数据. Features: {lF}")

# ==========================================
# 2. 参数设置
# ==========================================
delay = 3
lambda_val = 0.1
is_lasso = True
b = 2.5 / 3.5
h = 1 / 3.5

# ==========================================
# 4. 主循环逻辑
# ==========================================
Qfac = np.zeros((lnte, 1 + lF))  # 存储系数
QfacD = np.zeros(lnte)  # 存储决策
Valfac = np.zeros(lnte)  # 存储实际成本
Costfac = np.zeros(lnte)  # 存储优化目标值

print(f"Start CVXPY Optimization (Lasso={is_lasso})...")
start_time = time.time()

start_idx = lntr + lnva
for k in range(lnte):
    t = start_idx + k
    if k % 50 == 0: print(f"Step {k}/{lnte}")

    # A. 数据准备
    X_train_raw = Features_Raw[t - lntr: t, :]
    scale_factor = np.max(np.abs(X_train_raw))
    if scale_factor == 0: scale_factor = 1.0
    X_train = X_train_raw / scale_factor

    end_idx = t + delay
    if end_idx > TOTAL_LEN: break
    y_train = Demand[t - lntr + delay: end_idx]

    # B. 建模
    q0 = cp.Variable(1)
    q = cp.Variable(lF)
    predictions = q0 + X_train @ q

    underage = b * cp.maximum(y_train - predictions, 0)
    overage = h * cp.maximum(predictions - y_train, 0)
    empirical_risk = cp.sum(underage + overage) / lntr

    regularization = lambda_val * cp.norm(q, 1) if is_lasso else lambda_val * cp.norm(q, 2) ** 2

    prob = cp.Problem(cp.Minimize(empirical_risk + regularization), [q0 >= 0, q >= 0])
    try:
        prob.solve(solver=cp.SCS, eps=1e-3)
    except:
        prob.solve()

    # C. 记录
    Costfac[k] = prob.value
    if q0.value is not None:
        Qfac[k, 0] = q0.value[0]
        Qfac[k, 1:] = q.value

        # D. 测试
        X_current = Features_Raw[t, :] / scale_factor
        decision_q = max(0, q0.value[0] + X_current @ q.value)
        QfacD[k] = decision_q

        if t + delay < TOTAL_LEN:
            actual = Demand[t + delay]
            Valfac[k] = b * (actual - decision_q) if actual > decision_q else h * (decision_q - actual)

print(f"Total time: {time.time() - start_time:.2f} s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_emerg_reg_L1_{lambda_val}_simple_python.csv'

data_dict = {
    'Decision_Q': QfacD,
    'Realized_Cost': Valfac,
    'Optimization_Obj': Costfac
}
# 展开系数矩阵保存
for i in range(Qfac.shape[1]):
    col_name = 'Coef_Intercept' if i == 0 else f'Coef_Feat_{i}'
    data_dict[col_name] = Qfac[:, i]

df_out = pd.DataFrame(data_dict)
df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")