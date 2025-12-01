import numpy as np
import pandas as pd
import json
import time
import os
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../newsvendor_simple_data.csv'
config_file = '../data_config.json'

if not os.path.exists(data_file) or not os.path.exists(config_file):
    raise FileNotFoundError("找不到数据文件，请先运行 data_generator.py！")

# 读取 CSV
df = pd.read_csv(data_file)
Demand = df['Demand'].values

# 提取特征 (DayC, Time)
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features_Raw = np.hstack((DayC, Time))

# 读取配置
with open(config_file, 'r') as f:
    config = json.load(f)

lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']
TOTAL_LEN = len(Demand)

print(f"✅ 已加载数据: {data_file}. Total Samples: {TOTAL_LEN}")
print(f"   Features Shape: {Features_Raw.shape}")

# ==========================================
# 2. 参数设置 (Based on PDF)
# ==========================================
# 报童参数
b = 2.5 / 3.5
h = 1 / 3.5
tau = b / (b + h)  # Target Quantile (PDF notation: tau) [cite: 51-52]

# 模型参数
# Regularization parameter C = 1 / (lambda * m) [cite: 178]
# 我们这里直接设定 lambda，然后在循环中计算 C
lambda_reg = 0.01

# Kernel parameter sigma (for RBF kernel)
# gamma = 1 / (2 * sigma^2)
sigma = 1.0
gamma = 1.0 / (2 * sigma ** 2)


# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


def solve_kernel_quantile_dual(K, Y, C, tau):
    """
    求解对偶 QP 问题
    minimize 0.5 * alpha^T * K * alpha - alpha^T * Y
    s.t. C(tau-1) <= alpha_i <= C*tau
         sum(alpha) = 0
    """
    m = len(Y)
    alpha = cp.Variable(m)

    # 目标函数
    # 0.5 * quad_form(alpha, K) - alpha.T @ Y
    # 注意: cp.quad_form 要求 K 是半正定 (PSD)。RBF 核矩阵通常是 PSD 的，
    # 但数值误差可能导致微小的负特征值，使用 psd_wrap 保证求解器接受。
    objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(K)) - alpha.T @ Y)

    # 约束条件
    constraints = [
        alpha >= C * (tau - 1),
        alpha <= C * tau,
        cp.sum(alpha) == 0
    ]

    # 求解
    prob = cp.Problem(objective, constraints)
    # 使用 OSQP 或 SCS 求解器
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
    except:
        prob.solve(solver=cp.SCS, eps=1e-4)

    return alpha.value


# ==========================================
# 4. 主循环 (Kernel Quantile Regression)
# ==========================================
Q_pred = np.zeros(lnte)
Cost_realized = np.zeros(lnte)
Cost_in_sample = np.zeros(lnte)

print(f"Start Kernel Quantile Regression (lambda={lambda_reg}, sigma={sigma})...")
start_time = time.time()

# 全局归一化特征 (避免数据泄露，只用前 lntr+lnva 数据计算统计量)
train_feat_ref = Features_Raw[:lntr + lnva]
feat_min = np.min(train_feat_ref, axis=0)
feat_max = np.max(train_feat_ref, axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0


def normalize(X):
    return (X - feat_min) / feat_range


start_idx = lntr + lnva

# 为了提高效率，我们可以不每一步都重训练，而是每隔一定步数重训练一次
# 或者使用滑动窗口。这里为了严格对比，使用滑动窗口。
# 注意：QP 求解较慢，如果 lntr 很大 (如 >1000)，每步求解会非常耗时。
# 考虑到效率，这里我们设置一个 step_size，每隔 step_size 步更新一次模型。
update_step = 50

for k in range(lnte):
    t = start_idx + k

    if k % update_step == 0:
        if k % 100 == 0:
            print(f"Step {k}/{lnte} (Retraining Model...)")

        # A. 准备训练数据
        X_train_raw = Features_Raw[t - lntr: t]
        Y_train = Demand[t - lntr: t]
        X_train = normalize(X_train_raw)

        m = len(Y_train)
        C = 1.0 / (lambda_reg * m)  # [cite: 178]

        # B. 计算核矩阵 K
        # K_ij = k(x_i, x_j) [cite: 48]
        K_train = rbf_kernel(X_train, gamma=gamma)

        # C. 求解对偶变量 alpha
        alpha_val = solve_kernel_quantile_dual(K_train, Y_train, C, tau)

        # D. 计算偏置 b (Offset)
        # b 可以通过支持向量计算
        # 支持向量: C(tau-1) < alpha_i < C*tau
        # f(x_i) = y_i  =>  sum(alpha_j * K_ij) + b = y_i  =>  b = y_i - sum(...)
        if alpha_val is not None:
            support_indices = np.where(
                (alpha_val > C * (tau - 1) + 1e-5) &
                (alpha_val < C * tau - 1e-5)
            )[0]

            if len(support_indices) > 0:
                # 使用所有支持向量计算 b 的平均值以提高稳定性
                b_values = []
                for idx in support_indices:
                    pred_no_b = np.dot(K_train[idx], alpha_val)
                    b_values.append(Y_train[idx] - pred_no_b)
                b_val = np.mean(b_values)
            else:
                # 如果没有严格内部的支持向量（罕见），使用中位数近似或边界点
                b_val = 0.0
        else:
            # 求解失败 fallback
            alpha_val = np.zeros(m)
            b_val = np.quantile(Y_train, tau)

        # 缓存当前模型
        current_alpha = alpha_val
        current_b = b_val
        current_X_train = X_train

    # E. 预测 (Prediction)
    # f(x) = sum(alpha_i * k(x_i, x)) + b
    x_test_raw = Features_Raw[t, :].reshape(1, -1)
    x_test = normalize(x_test_raw)

    # 计算测试点与所有训练点的核函数值向量 k_vector
    # k_vector[i] = k(x_i, x_test)
    k_vector = rbf_kernel(current_X_train, x_test, gamma=gamma).flatten()

    q_pred = np.dot(current_alpha, k_vector) + current_b
    q_pred = max(0, q_pred)

    # F. 记录
    Q_pred[k] = q_pred

    # 估算样本内成本 (基于当前训练集)
    # 注意：这里的 Cost_in_sample 是基于上一次训练时刻的训练集
    # 为了速度，不再每一步都重新计算全量样本内成本

    # 样本外测试
    if t < TOTAL_LEN:
        actual_demand = Demand[t]
        Cost_realized[k] = nv_cost(q_pred, actual_demand, b, h)

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_kernel_quantile_lambda{lambda_reg}_python.csv'

df_out = pd.DataFrame({
    'Decision_Q': Q_pred,
    'Realized_Cost': Cost_realized
})

df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")