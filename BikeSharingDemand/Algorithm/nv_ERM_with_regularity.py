import numpy as np
import pandas as pd
import cvxpy as cp
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载与预处理
# ==========================================
train_file = '../data/input_data/train_set.csv'
test_file = '../data/input_data/test_set.csv'

# 读取数据
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# 指定特征
feature_cols = ['temp', 'humidity', 'windspeed', 'season', 'weather']
target_col = 'demand'

# 提取矩阵
X_train_raw = train_df[feature_cols].values
y_train = train_df[target_col].values

X_test_raw = test_df[feature_cols].values
y_test = test_df[target_col].values

# 归一化 (Min-Max)
# 重要：必须用训练集的统计量来缩放测试集
feat_min = np.min(X_train_raw, axis=0)
feat_max = np.max(X_train_raw, axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0  # 防止除以0

def normalize(X):
    return (X - feat_min) / feat_range

X_train = normalize(X_train_raw)
X_test = normalize(X_test_raw)

n_train, feat_dim = X_train.shape
n_test = X_test.shape[0]

print(f"数据加载完成.")
print(f"训练集: {n_train}, 测试集: {n_test}")
print(f"特征维度: {feat_dim} {feature_cols}")

# ==========================================
# 2. 参数设置
# ==========================================
b = 30  # 缺货成本
h = 10  # 持有成本
# 目标分位数 (Target Quantile) = 30 / 40 = 0.75
# 在线性分位数回归中，这相当于倾斜的绝对值损失函数
tau = b / (b + h)

# 正则化参数
lambda_val = 0.1  # 正则化强度 (可调)
is_lasso = True   # True=L1正则, False=L2正则

# ==========================================
# 3. 模型训练 (ERM - Empirical Risk Minimization)
# ==========================================
print(f"\n开始训练 ERM 模型 (Lasso={is_lasso})...")
start_time = time.time()

# --- CVXPY 建模 ---
beta_0 = cp.Variable(1)          # 截距项
beta = cp.Variable(feat_dim)     # 特征系数

# 预测值向量
predictions = beta_0 + X_train @ beta

# 定义损失函数: Newsvendor Loss (Pinball Loss / Check Loss)
# Loss = sum( b * max(y - y_pred, 0) + h * max(y_pred - y, 0) )
# 注意：这里我们直接优化报童损失，这等价于分位数回归
underage = b * cp.maximum(y_train - predictions, 0)
overage = h * cp.maximum(predictions - y_train, 0)
empirical_risk = cp.sum(underage + overage) / n_train

# 正则化项
if is_lasso:
    regularization = lambda_val * cp.norm(beta, 1)
else:
    regularization = lambda_val * cp.norm(beta, 2)**2

# 目标函数
objective = cp.Minimize(empirical_risk + regularization)

# 约束条件
# 注意：移除了 beta >= 0 的约束，因为风速/湿度可能与需求呈负相关
# 但我们通常希望最终的订货量是非负的
constraints = []

prob = cp.Problem(objective, constraints)

# 求解
try:
    prob.solve(solver=cp.GUROBI, verbose=False)
except:
    prob.solve(solver=cp.ECOS, verbose=False)

train_time = time.time() - start_time
print(f"训练完成. 耗时: {train_time:.4f} s. Status: {prob.status}")

# 提取系数
intercept_val = beta_0.value[0]
coef_vals = beta.value

print("\n--- 训练结果 (Coefficients) ---")
print(f"Intercept (截距): {intercept_val:.4f}")
for name, val in zip(feature_cols, coef_vals):
    print(f"{name}: {val:.4f}")

# ==========================================
# 4. 测试集预测与评估
# ==========================================
# 计算预测值 Q = beta_0 + X_test * beta
Q_pred = intercept_val + X_test @ coef_vals
# 修正：订货量不能为负数
Q_pred = np.maximum(Q_pred, 0)

# 计算样本外成本
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
# 5. 保存结果 (CSV)
# ==========================================
output_filename = '../data/output_data/nv_ERM_with_regularity.csv'

df_out = pd.DataFrame({
    'Actual_Demand': y_test,
    'Predicted_Order': Q_pred,
    'Realized_Cost': out_of_sample_cost
})

# 把系数也存进去方便后续分析（重复存储在每一行或另存文件）
# 这里简单起见，不存系数到这个结果文件里，只存预测流

df_out.to_csv(output_filename, index=False)
print(f"预测流结果已保存至 {output_filename}")

# ==========================================
# 6. 可视化
# ==========================================
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(14, 6))

t = np.arange(n_test)

# 1. 真实需求
plt.plot(t, y_test, label='Actual Demand (Test Set)', color='#2c3e50', alpha=0.8, marker='.', linestyle='-')

# 2. 预测订货量
plt.plot(t, Q_pred, label='Optimal Order (ERM)', color='#e74c3c', linewidth=2.5)

# 3. 填充成本区域
plt.fill_between(t, y_test, Q_pred, where=(Q_pred >= y_test),
                 interpolate=True, color='#27ae60', alpha=0.2, label='Over-stocking')
plt.fill_between(t, y_test, Q_pred, where=(Q_pred < y_test),
                 interpolate=True, color='#c0392b', alpha=0.2, label='Under-stocking')

plt.title(f'NV-ERM (Linear Quantile Regression)\nAvg Cost: {avg_cost:.2f} | Features: {len(feature_cols)}', fontsize=16)
plt.ylabel('Quantity')
plt.xlabel('Test Sample Index')
plt.legend()
plt.tight_layout()
plt.show()