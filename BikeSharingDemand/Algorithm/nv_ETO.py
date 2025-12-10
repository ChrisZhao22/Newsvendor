import numpy as np
import pandas as pd
from scipy.stats import norm
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
print(f"特征: {feature_cols}")

# ==========================================
# 2. 参数设置
# ==========================================
b = 30  # 缺货成本
h = 10  # 持有成本
ratio = b / (b + h) # 临界分位数 (0.75)

# 标准正态分布的 z-score
z_score = norm.ppf(ratio)

print(f"成本参数: b={b}, h={h} => Target Ratio={ratio:.2f} => Z-score={z_score:.2f}")

# ==========================================
# 3. 模型训练 (Two-Stage Regression)
# ==========================================
print(f"\n开始训练 ETO 模型 (Mean + Variance Regression)...")
start_time = time.time()

# --- Step 1: 均值回归 (Mean Model) ---
# E[D|X] = X * beta_mu
model_mu = LinearRegression(fit_intercept=True)
model_mu.fit(X_train, y_train)

# 获取训练集上的预测均值
mu_train_pred = model_mu.predict(X_train)

# --- Step 2: 方差回归 (Variance Model) ---
# 既然 ETO 假设正态分布，我们需要估计条件方差 Var[D|X]
# 方法：对残差的平方取对数进行回归 -> log(residuals^2) ~ X * beta_sigma
residuals = y_train - mu_train_pred
# 加一个极小值防止 log(0)
log_res_sq = np.log(residuals**2 + 1e-10)

model_sigma = LinearRegression(fit_intercept=True)
model_sigma.fit(X_train, log_res_sq)

train_time = time.time() - start_time
print(f"训练完成. 耗时: {train_time:.4f} s")

# 打印系数
print("\n--- Mean Model Coefficients ---")
print(f"Intercept: {model_mu.intercept_:.4f}")
for name, val in zip(feature_cols, model_mu.coef_):
    print(f"{name}: {val:.4f}")

# ==========================================
# 4. 测试集预测与评估
# ==========================================
# A. 预测均值 mu
mu_test = model_mu.predict(X_test)

# B. 预测方差 sigma (注意我们要还原对数)
# log(sigma^2) = prediction
# sigma = sqrt(exp(prediction))
log_sigma2_test = model_sigma.predict(X_test)
sigma_test = np.sqrt(np.exp(log_sigma2_test))

# C. 计算最优订货量 Q* = mu + z * sigma
# 假设条件分布为正态分布 N(mu, sigma^2)
Q_pred = mu_test + z_score * sigma_test

# 修正：需求非负
Q_pred = np.maximum(Q_pred, 0)

# D. 计算成本
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
# 5. 保存结果
# ==========================================
output_filename = '../data/output_data/nv_ETO.csv'

df_out = pd.DataFrame({
    'Actual_Demand': y_test,
    'Predicted_Order': Q_pred,
    'Predicted_Mu': mu_test,
    'Predicted_Sigma': sigma_test,
    'Realized_Cost': out_of_sample_cost
})

df_out.to_csv(output_filename, index=False)
print(f"预测结果已保存至 {output_filename}")

# ==========================================
# 6. 可视化
# ==========================================
sns.set_theme(style="whitegrid", context="talk")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

t = np.arange(n_test)

# --- 子图 1: 订货量 vs 真实需求 ---
ax1.plot(t, y_test, label='Actual Demand', color='#2c3e50', marker='.', linestyle='-', alpha=0.7)
ax1.plot(t, Q_pred, label='Optimal Order (ETO)', color='#e74c3c', linewidth=2)
# 填充区域
ax1.fill_between(t, y_test, Q_pred, where=(Q_pred >= y_test),
                 interpolate=True, color='#27ae60', alpha=0.2, label='Over-stocking')
ax1.fill_between(t, y_test, Q_pred, where=(Q_pred < y_test),
                 interpolate=True, color='#c0392b', alpha=0.2, label='Under-stocking')

ax1.set_title(f'NV-ETO (Parametric Normal): Decisions\nAvg Cost: {avg_cost:.2f}', fontsize=16)
ax1.set_ylabel('Quantity')
ax1.legend(loc='upper right', fontsize=12)

# --- 子图 2: 预测的均值与置信区间 (95%) ---
# 展示模型对不确定性的估计能力
ax2.plot(t, mu_test, label='Predicted Mean ($\mu$)', color='blue', linestyle='--')
# 画出 +/- 2 sigma 的区间 (约95%置信区间)
upper_bound = mu_test + 1.96 * sigma_test
lower_bound = mu_test - 1.96 * sigma_test
ax2.fill_between(t, lower_bound, upper_bound, color='blue', alpha=0.1, label='95% Confidence Interval ($\mu \pm 1.96\sigma$)')
ax2.scatter(t, y_test, color='black', s=10, alpha=0.5, label='Actual Demand') # 点图

ax2.set_title('Uncertainty Estimation: Mean & Confidence Interval', fontsize=14)
ax2.set_ylabel('Quantity')
ax2.set_xlabel('Test Sample Index')
ax2.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()