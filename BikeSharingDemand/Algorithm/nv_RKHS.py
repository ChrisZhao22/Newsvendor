import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载与预处理
# ==========================================
train_file = '../data/input_data/train_set.csv'
test_file = '../data/input_data/test_set.csv'

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

feature_cols = ['temp', 'humidity', 'windspeed', 'season', 'weather']
target_col = 'demand'

X_train_raw = train_df[feature_cols].values
y_train = train_df[target_col].values

X_test_raw = test_df[feature_cols].values
y_test = test_df[target_col].values

# 归一化 (Min-Max) - 核方法必须归一化
feat_min = np.min(X_train_raw, axis=0)
feat_max = np.max(X_train_raw, axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0


def normalize(X):
    return (X - feat_min) / feat_range


X_train = normalize(X_train_raw)
X_test = normalize(X_test_raw)

n_train = len(X_train)
n_test = len(X_test)

print(f"数据加载完成. 训练集: {n_train}, 测试集: {n_test}")

# ==========================================
# 2. 参数设置
# ==========================================
b = 30  # 缺货成本
h = 10  # 持有成本
tau = b / (b + h)  # Target Quantile (0.75)

# --- 模型超参数 ---
# Lambda: 正则化强度. 越小模型越复杂(易过拟合), 越大模型越平滑.
lambda_reg = 0.0000009

# Sigma: RBF核的宽度. 越小越敏感(只看极近邻), 越大越平滑(看全局).
sigma = 1.0
gamma = 1.0 / (2 * sigma ** 2)

# 计算 SVM 形式的惩罚系数 C
# C = 1 / (lambda * n)
C_val = 1.0 / (lambda_reg * n_train)

print(f"参数: b={b}, h={h} => Tau={tau:.2f}")
print(f"超参数: Lambda={lambda_reg}, Sigma={sigma} => C={C_val:.4f}, Gamma={gamma:.4f}")


# ==========================================
# 3. 辅助函数
# ==========================================
def solve_kernel_quantile_dual(K, Y, C, tau):
    """
    求解 KQR 对偶 QP 问题
    """
    m = len(Y)
    alpha = cp.Variable(m)

    # 目标函数: 0.5 * alpha^T * K * alpha - alpha^T * Y
    # psd_wrap 告诉 cvxpy K 矩阵是半正定的 (Positive Semi-Definite)
    objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(K)) - alpha.T @ Y)

    # 约束条件
    constraints = [
        alpha >= C * (tau - 1),
        alpha <= C * tau,
        cp.sum(alpha) == 0
    ]

    # 求解
    prob = cp.Problem(objective, constraints)

    # 尝试求解
    try:
        # OSQP 是处理这种 QP 问题的强力求解器
        prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, verbose=False)
    except:
        prob.solve(verbose=False)

    return alpha.value


# ==========================================
# 4. 模型训练 (Static Training)
# ==========================================
print(f"\n开始训练 Kernel Quantile Regression (KQR)...")
start_time = time.time()

# A. 计算核矩阵 K_train (n_train x n_train)
# 这步可能比较耗内存，如果 n > 5000 需谨慎
K_train = rbf_kernel(X_train, gamma=gamma)

# B. 求解对偶变量 alpha
alpha_val = solve_kernel_quantile_dual(K_train, y_train, C_val, tau)

if alpha_val is None:
    print("错误: 求解器失败，回退到全局分位数预测。")
    # Fallback plan
    b_val = np.quantile(y_train, tau)
    alpha_val = np.zeros(n_train)
else:
    # C. 计算截距 b (Offset)
    # 理论上 b = y_i - sum(alpha_j * K_ij) 对于任意支持向量 i 成立
    # 支持向量满足: C(tau-1) < alpha_i < C*tau

    # 这里的阈值 1e-5 是为了避开数值误差边界
    lower_bound = C_val * (tau - 1) + 1e-5
    upper_bound = C_val * tau - 1e-5

    support_indices = np.where((alpha_val > lower_bound) & (alpha_val < upper_bound))[0]

    if len(support_indices) > 0:
        # 取所有支持向量计算出的 b 的平均值
        b_list = []
        # 预计算预测值 (不含 b) -> K @ alpha
        pred_no_b_all = K_train @ alpha_val

        for idx in support_indices:
            b_list.append(y_train[idx] - pred_no_b_all[idx])
        b_val = np.mean(b_list)
        print(f"找到 {len(support_indices)} 个支持向量来计算截距 b.")
    else:
        print("警告: 没有找到严格内部的支持向量 (可能是 C 太大或太小). 使用中位数近似.")
        # 近似方法：让预测值的均值 对齐 目标值的均值 (偏差修正)
        pred_no_b_all = K_train @ alpha_val
        b_val = np.mean(y_train - pred_no_b_all)

train_time = time.time() - start_time
print(f"训练完成. 耗时: {train_time:.4f} s")

# ==========================================
# 5. 测试集预测与评估
# ==========================================
# 计算测试集核矩阵 K_test (n_test x n_train)
# K_test[i, j] = kernel(x_test[i], x_train[j])
K_test = rbf_kernel(X_test, X_train, gamma=gamma)

# 预测: f(x) = sum(alpha_i * K(x_i, x)) + b
# 矩阵形式: pred = K_test @ alpha + b
Q_pred = K_test @ alpha_val + b_val

# 修正：非负需求
Q_pred = np.maximum(Q_pred, 0)

# 计算成本
out_of_sample_cost = np.zeros(n_test)
for i in range(n_test):
    d = y_test[i]
    q = Q_pred[i]
    if q >= d:
        out_of_sample_cost[i] = (q - d) * h
    else:
        out_of_sample_cost[i] = (d - q) * b

avg_cost = np.mean(out_of_sample_cost)
print(f"\n测试集平均成本: {avg_cost:.2f}")

# ==========================================
# 6. 保存结果
# ==========================================
output_filename = '../data/output_data/nv_RKHS.csv'

df_out = pd.DataFrame({
    'Actual_Demand': y_test,
    'Predicted_Order': Q_pred,
    'Realized_Cost': out_of_sample_cost
})
df_out.to_csv(output_filename, index=False)
print(f"结果已保存至 {output_filename}")

# ==========================================
# 7. 可视化
# ==========================================
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(14, 6))

t = np.arange(n_test)

# 1. 真实需求
plt.plot(t, y_test, label='Actual Demand', color='#2c3e50', alpha=0.8, marker='.', linestyle='-')

# 2. 预测订货量
plt.plot(t, Q_pred, label='Optimal Order (KQR/RKHS)', color='#e74c3c', linewidth=2.5)

# 3. 填充成本区域
plt.fill_between(t, y_test, Q_pred, where=(Q_pred >= y_test),
                 interpolate=True, color='#27ae60', alpha=0.2, label='Over-stocking')
plt.fill_between(t, y_test, Q_pred, where=(Q_pred < y_test),
                 interpolate=True, color='#c0392b', alpha=0.2, label='Under-stocking')

plt.title(f'NV-RKHS (Kernel Quantile Regression)\nAvg Cost: {avg_cost:.2f} | Sigma={sigma}, Lambda={lambda_reg}',
          fontsize=16)
plt.ylabel('Quantity')
plt.xlabel('Test Sample Index')
plt.legend()
plt.tight_layout()
plt.show()