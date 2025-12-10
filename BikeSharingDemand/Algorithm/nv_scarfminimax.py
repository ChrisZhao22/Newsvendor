import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

# 指定特征
feature_cols = ['temp', 'humidity', 'windspeed', 'season', 'weather']
target_col = 'demand'

# 提取矩阵
X_train_raw = train_df[feature_cols].values
y_train = train_df[target_col].values

X_test_raw = test_df[feature_cols].values
y_test = test_df[target_col].values

# 归一化 (使用训练集统计量)
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
b_cost = 30.0  # 缺货成本 (Backorder Cost)
h_cost = 10.0  # 持有成本 (Holding Cost)

# Scarf's Rule 的核心项
# Q = mu + sigma * 0.5 * (sqrt(b/h) - sqrt(h/b))
cost_ratio = b_cost / h_cost
scarf_z_score = 0.5 * (np.sqrt(cost_ratio) - 1.0 / np.sqrt(cost_ratio))

print(f"成本参数: b={b_cost}, h={h_cost}")
print(f"Scarf's Rule Z-score (Robust Factor): {scarf_z_score:.4f}")

# ==========================================
# 3. 模型训练 (Mean & Variance Regression)
# ==========================================
print(f"\n开始训练 Scarf's Rule 模型 (Mean + Variance Estimation)...")
start_time = time.time()

# --- Step 1: 均值回归 (Mean Model) ---
model_mean = LinearRegression(fit_intercept=True)
model_mean.fit(X_train, y_train)

# 获取训练集残差
mu_train_pred = model_mean.predict(X_train)
residuals = y_train - mu_train_pred
res_sum_sq = np.sum(residuals**2)

# --- Step 2: 方差回归 (Variance Model) ---
# log(residuals^2) ~ X * beta
y_train_var = np.log(residuals**2 + 1e-10)
model_var = LinearRegression(fit_intercept=True)
model_var.fit(X_train, y_train_var)

train_time = time.time() - start_time
print(f"训练完成. 耗时: {train_time:.4f} s")
print(f"训练集残差平方和 (RSS): {res_sum_sq:.2f}")

# ==========================================
# 4. 测试集预测与评估
# ==========================================
# A. 预测均值 mu
mu_test = model_mean.predict(X_test)

# B. 预测方差 sigma
log_sigma2_test = model_var.predict(X_test)
sigma_test = np.sqrt(np.exp(log_sigma2_test))

# C. 计算 Scarf's Rule 订货量
# Q* = mu + sigma * scarf_factor
Q_pred = mu_test + sigma_test * scarf_z_score

# 修正：需求非负
Q_pred = np.maximum(Q_pred, 0)

# D. 计算成本
out_of_sample_cost = np.zeros(n_test)
for i in range(n_test):
    d = y_test[i]
    q = Q_pred[i]
    if q >= d:
        out_of_sample_cost[i] = (q - d) * h_cost
    else:
        out_of_sample_cost[i] = (d - q) * b_cost

avg_cost = np.mean(out_of_sample_cost)
print(f"\n测试集平均成本: {avg_cost:.2f}")

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = '../data/output_data/nv_scarf.csv'

df_out = pd.DataFrame({
    'Actual_Demand': y_test,
    'Predicted_Order': Q_pred,
    'Predicted_Mu': mu_test,
    'Predicted_Sigma': sigma_test,
    'Realized_Cost': out_of_sample_cost
})
df_out.to_csv(output_filename, index=False)
print(f"结果已保存至 {output_filename}")

# ==========================================
# 6. 可视化
# ==========================================
sns.set_theme(style="whitegrid", context="talk")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

t = np.arange(n_test)

# --- 子图 1: 决策与需求对比 ---
ax1.plot(t, y_test, label='Actual Demand', color='#2c3e50', alpha=0.7, marker='.', linestyle='-')
ax1.plot(t, Q_pred, label="Optimal Order (Scarf's Rule)", color='#e74c3c', linewidth=2)

# 填充成本
ax1.fill_between(t, y_test, Q_pred, where=(Q_pred >= y_test),
                 interpolate=True, color='#27ae60', alpha=0.2, label='Over-stocking')
ax1.fill_between(t, y_test, Q_pred, where=(Q_pred < y_test),
                 interpolate=True, color='#c0392b', alpha=0.2, label='Under-stocking')

ax1.set_title(f"NV-Scarf (Min-Max Robust Rule)\nAvg Cost: {avg_cost:.2f}", fontsize=16)
ax1.set_ylabel('Quantity')
ax1.legend(loc='upper right', fontsize=12)

# --- 子图 2: 均值与 Scarf 调整量的分解 ---
ax2.plot(t, mu_test, label='Predicted Mean ($\mu$)', color='blue', linestyle='--')
# 叠加 sigma 带来的缓冲库存 (Safety Stock)
# Scarf Rule 实际上是在均值基础上加了一个安全库存: sigma * z_scarf
ax2.fill_between(t, mu_test, Q_pred, color='orange', alpha=0.3, label="Robust Safety Stock ($z_{scarf} \cdot \sigma$)")

ax2.set_title("Decision Decomposition: Mean vs. Robust Buffer", fontsize=14)
ax2.set_ylabel('Quantity')
ax2.set_xlabel('Test Sample Index')
ax2.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()