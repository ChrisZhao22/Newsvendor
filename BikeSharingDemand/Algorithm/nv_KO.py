import numpy as np
import pandas as pd
from scipy.stats import norm
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

# 归一化 (Min-Max) - 核方法对距离非常敏感，归一化至关重要
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
ratio = b / (b + h)  # 目标分位数 (0.75)

# 带宽列表 (Bandwidth)
# 带宽决定了"邻域"的大小。太小=过拟合(只看最近的一个点)，太大=欠拟合(退化为全局分位数)
# 对于归一化后(0-1)的5维空间，0.1~0.5 是常见范围
bandvec = [0.1, 0.3, 0.5]


def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 3. 主循环逻辑 (Kernel Optimization)
# ==========================================
results_dict = {'Actual_Demand': y_test}
best_avg_cost = float('inf')
best_bw = None
best_Q_pred = None

print(f"\n开始 Kernel Optimization (KO) 计算...")
start_time = time.time()

for bandwidth in bandvec:
    print(f"  Processing bandwidth h={bandwidth} ...")

    Q_pred = np.zeros(n_test)
    Cost_list = np.zeros(n_test)

    # 遍历每一个测试样本
    for i in range(n_test):
        # 当前测试点的特征向量
        x_curr = X_test[i]

        # 1. 计算距离 (Euclidean Distance)
        # 计算 x_curr 与所有 X_train 的距离
        dists = np.linalg.norm(X_train - x_curr, axis=1)

        # 2. 计算核权重 (Gaussian Kernel)
        # weights ~ exp( - dist^2 / (2*h^2) ) / (sqrt(2pi)*h)
        # 实际上 norm.pdf(x / h) 就可以
        weights = norm.pdf(dists / bandwidth)

        # 归一化权重 (防止全为0)
        w_sum = np.sum(weights)
        if w_sum > 1e-10:
            weights_norm = weights / w_sum
        else:
            # 如果所有点都太远(权重近似0)，则退化为平均权重
            weights_norm = np.ones(n_train) / n_train

        # 3. 求解加权分位数 (Weighted Quantile)
        # 目标：找到 q，使得 sum(weights * I(y <= q)) >= ratio

        # 将训练集需求和权重绑定排序
        sort_idx = np.argsort(y_train)
        sDemand = y_train[sort_idx]
        sWeights = weights_norm[sort_idx]

        # 计算加权累积分布 (Weighted CDF)
        cumulative_weights = np.cumsum(sWeights)

        # 找到第一个累积权重超过 ratio 的位置
        idx_opt = np.searchsorted(cumulative_weights, ratio)
        # 边界保护
        idx_opt = min(idx_opt, n_train - 1)

        q_opt = sDemand[idx_opt]

        # 4. 记录
        Q_pred[i] = q_opt
        actual = y_test[i]
        Cost_list[i] = nv_cost(q_opt, actual, b, h)

    # 统计当前带宽的表现
    avg_cost = np.mean(Cost_list)
    print(f"    -> Avg Cost: {avg_cost:.2f}")

    # 保存结果
    results_dict[f'Pred_Q_bw_{bandwidth}'] = Q_pred
    results_dict[f'Cost_bw_{bandwidth}'] = Cost_list

    # 记录最佳模型
    if avg_cost < best_avg_cost:
        best_avg_cost = avg_cost
        best_bw = bandwidth
        best_Q_pred = Q_pred

print(f"计算完成. 总耗时: {time.time() - start_time:.4f} s")
print(f"最佳带宽: {best_bw}, 最低平均成本: {best_avg_cost:.2f}")

# ==========================================
# 4. 保存结果
# ==========================================
output_filename = '../data/output_data/nv_KO.csv'
df_out = pd.DataFrame(results_dict)
df_out.to_csv(output_filename, index=False)
print(f"结果已保存至 {output_filename}")

# ==========================================
# 5. 可视化 (展示最佳带宽的结果)
# ==========================================
sns.set_theme(style="whitegrid", context="talk")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

t = np.arange(n_test)

# --- 子图 1: 全局趋势 ---
ax1.plot(t, y_test, label='Actual Demand', color='#2c3e50', alpha=0.7, marker='.', linestyle='-')
ax1.plot(t, best_Q_pred, label=f'Optimal Order (KO, h={best_bw})', color='#e74c3c', linewidth=2)

# 填充成本区域
ax1.fill_between(t, y_test, best_Q_pred, where=(best_Q_pred >= y_test),
                 interpolate=True, color='#27ae60', alpha=0.2, label='Over-stocking')
ax1.fill_between(t, y_test, best_Q_pred, where=(best_Q_pred < y_test),
                 interpolate=True, color='#c0392b', alpha=0.2, label='Under-stocking')

ax1.set_title(f'NV-KO (Kernel Optimization): Best Bandwidth h={best_bw}\nAvg Cost: {best_avg_cost:.2f}', fontsize=16)
ax1.set_ylabel('Quantity')
ax1.legend(loc='upper right', fontsize=12)

# --- 子图 2: 不同带宽的对比 (前50个样本) ---
ax2.plot(t[:50], y_test[:50], color='black', alpha=0.3, linestyle='--', label='Demand')
colors = sns.color_palette("husl", len(bandvec))

for i, bw in enumerate(bandvec):
    pred = results_dict[f'Pred_Q_bw_{bw}']
    ax2.plot(t[:50], pred[:50], label=f'h={bw}', color=colors[i], linewidth=1.5)

ax2.set_title('Bandwidth Sensitivity Analysis (Zoomed In)', fontsize=14)
ax2.set_ylabel('Quantity')
ax2.set_xlabel('Test Sample Index')
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()