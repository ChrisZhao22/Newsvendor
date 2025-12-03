import numpy as np
import pandas as pd
import json
import cvxpy as cp
import time

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

df = pd.read_csv(data_file)
Demand = df['Demand'].values
# 特征构建
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features_Raw = np.hstack((DayC, Time))
feat_dim = Features_Raw.shape[1]

with open(config_file, 'r') as f:
    config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']
TOTAL_LEN = len(Demand)

print(f"已加载数据. Features: {feat_dim}")

# ==========================================
# 2. 参数设置
# ==========================================
delay = 0
lambda_val = 1e-7  # 可调参数（正则化程度）
is_lasso = True
b = 2.5 / 3.5
h = 1 / 3.5

# ==========================================
# 4. 主循环逻辑
# ==========================================
coef = np.zeros((lnte, 1 + feat_dim))  # 存储系数
D_pred = np.zeros(lnte)  # 存储决策
out_of_sample_cost = np.zeros(lnte)  # 存储实际成本
Costfac = np.zeros(lnte)  # 存储优化目标值

print(f"Start CVXPY Optimization (Lasso={is_lasso})...")
start_time = time.time()

start_idx = lntr + lnva

for k in range(lnte):
    t = start_idx + k
    if k % 50 == 0: print(f"Step {k}/{lnte}")

    # A. 数据准备
    X_train_raw = Features_Raw[t - lntr: t, :]
    scale_factor = np.max(np.sum(np.abs(X_train_raw), axis=1))
    X_train = X_train_raw / scale_factor

    y_train = Demand[t - lntr + delay: t + delay]

    # B. 建模
    beta_0 = cp.Variable(1)
    beta = cp.Variable(feat_dim)
    predictions = beta_0 + X_train @ beta

    underage = b * cp.maximum(y_train - predictions, 0)
    overage = h * cp.maximum(predictions - y_train, 0)
    empirical_risk = cp.sum(underage + overage) / lntr
    # 加正则化：lasso（当is_lasso参数取False时，为ridge regression）
    regularization = lambda_val * cp.norm(beta, 1) if is_lasso else lambda_val * cp.norm(beta, 2) ** 2

    objective = cp.Minimize(empirical_risk + regularization)
    # 非负约束（GYB代码中存在，但其合理性存疑）
    constraints = [
        beta_0 >= 0,
        beta >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)  # 调用求解器进行一次求解

    # C. 记录
    Costfac[k] = prob.value
    if beta_0.value is not None:
        coef[k, 0] = beta_0.value[0]
        coef[k, 1:] = beta.value

        # D. 计算样本外成本
        X_current = Features_Raw[t, :] / scale_factor
        decision_q = beta_0.value[0] + X_current @ beta.value
        D_pred[k] = decision_q

        if t < TOTAL_LEN:
            actual = Demand[t + delay]
            out_of_sample_cost[k] = np.maximum(actual - decision_q, 0) * b + np.maximum(decision_q - actual, 0) * h
print(f"Total time: {time.time() - start_time:.2f} s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_ERM.csv'

data_dict = {
    'Decision_Q': D_pred,
    'Demand_D': Demand[start_idx + delay:start_idx + lnte + delay],
    'Realized_Cost': out_of_sample_cost,
    'Optimization_Obj': Costfac
}
# 展开系数矩阵保存
for i in range(coef.shape[1]):
    col_name = 'Coef_Intercept' if i == 0 else f'Coef_Feat_{i}'
    data_dict[col_name] = coef[:, i]

df_out = pd.DataFrame(data_dict)
df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")
