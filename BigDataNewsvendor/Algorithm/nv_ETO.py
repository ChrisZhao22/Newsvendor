import numpy as np
import pandas as pd
import json
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

df = pd.read_csv(data_file)
Demand = df['Demand'].values

# 提取特征
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features = np.hstack((DayC, Time))

with open(config_file, 'r') as f:
    config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']

TOTAL_LEN = len(Demand)
print(f"已加载数据. Features shape: {Features.shape}")

# ==========================================
# 2. 参数设置
# ==========================================
delay = 0
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
out_of_sample_cost = np.zeros(lnte)
muD = np.zeros(lnte)
sigmaD = np.zeros(lnte)
Q_pred = np.zeros(lnte)

# 记录系数 (Intercept + 2个特征)
coef_history = np.zeros((lnte, 1 + Features.shape[1]))

start_time = time.time()

start_idx = lntr + lnva

for i in range(lnte):
    t = start_idx + i

    if i % 100 == 0: print(f"Step {i}/{lnte}")

    # A. 准备数据
    X_train_raw = Features[t - lntr: t, :]
    scale_factor = np.max(np.sum(np.abs(X_train_raw), axis=1))
    X_train = X_train_raw / scale_factor
    y_train = Demand[t - lntr + delay: t + delay]

    # B. 均值回归
    model_mu = LinearRegression(fit_intercept=True)
    model_mu.fit(X_train, y_train)

    # C. 方差回归
    residuals = y_train - model_mu.predict(X_train)
    y_train_var = np.log(residuals ** 2 + 1e-8)
    model_sigma = LinearRegression(fit_intercept=True)
    model_sigma.fit(X_train, y_train_var)

    # D. 预测
    X_current = Features[t, :].reshape(1, -1) / scale_factor
    mu_pred = model_mu.predict(X_current)[0]
    sigma_pred = np.exp(model_sigma.predict(X_current)[0] / 2)

    muD[i] = mu_pred
    sigmaD[i] = sigma_pred

    # E. 优化
    z_score = norm.ppf(r)
    optimal_Q = mu_pred + sigma_pred * z_score
    Q_pred[i] = max(optimal_Q, 0)

    actual_demand = Demand[t + delay]
    out_of_sample_cost[i] = nv_cost(optimal_Q, actual_demand, b, h)

    # 记录系数
    coef_history[i, 0] = model_mu.intercept_
    coef_history[i, 1:] = model_mu.coef_

print(f"Loop finished in {time.time() - start_time:.2f} s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_ETO.csv'

# 基础结果
data_dict = {
    'Decision_Q': Q_pred,
    'Decision_D': Demand[start_idx + delay:start_idx + lnte + delay],
    'Cost': out_of_sample_cost,
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

# ==========================================
# 6. 可视化：AI美化版 (Professional Style)
# ==========================================
sns.set_theme(style="whitegrid", context="talk")

# 创建画布，包含两个子图 (上下排列)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1]})

# 定义时间轴和数据片段
t = np.arange(lnte)
y_demand = Demand[start_idx:start_idx + lnte]
y_pred = Q_pred

# ====================
# 子图 1: 全局趋势 (Global View)
# ====================
# 1. 画线
ax1.plot(t, y_demand, label='Actual Demand', color='#2c3e50', linewidth=1.5, alpha=0.8)  # 深蓝灰色
ax1.plot(t, y_pred, label='Optimal Order (Q)', color='#e74c3c', linewidth=2.0, linestyle='-')  # 亮红色

# 2. 填充成本区域 (高亮差异)
# 当 订货 > 需求 (库存积压/Holding Cost) -> 用浅黄色/绿色填充
ax1.fill_between(t, y_demand, y_pred, where=(y_pred >= y_demand),
                 interpolate=True, color='#27ae60', alpha=0.15, label='Inventory (Overage)')
# 当 订货 < 需求 (缺货损失/Stockout Cost) -> 用浅红色填充
ax1.fill_between(t, y_demand, y_pred, where=(y_pred < y_demand),
                 interpolate=True, color='#c0392b', alpha=0.15, label='Stockout (Underage)')

# 3. 装饰
ax1.set_title('NV-ETO: Newsvendor Decision Analysis: Global Overview', fontsize=18, fontweight='bold', pad=15)
ax1.set_ylabel('Quantity', fontsize=14)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax1.set_xlim(0, lnte)
ax1.margins(x=0)

# ====================
# 子图 2: 细节放大 (Zoomed View)
# ====================
zoom_len = 300
t_zoom = t[:zoom_len]
y_demand_zoom = y_demand[:zoom_len]
y_pred_zoom = y_pred[:zoom_len]

# 1. 画线 (带数据点 Marker，方便看具体点的位置)
ax1.plot(t_zoom, y_demand_zoom, color='#2c3e50', alpha=0, linewidth=0)  # 仅用于统一颜色逻辑，实际画在下面
ax2.plot(t_zoom, y_demand_zoom, label='Actual Demand', color='#2c3e50',
         linewidth=1.8, linestyle='-', marker='.', markersize=4, alpha=0.7)
ax2.plot(t_zoom, y_pred_zoom, label='Optimal Order (Q)', color='#e74c3c',
         linewidth=2.5, linestyle='-', alpha=0.9)  # 预测线通常平滑，不加marker防止太乱

# 2. 填充区域 (同样高亮成本)
ax2.fill_between(t_zoom, y_demand_zoom, y_pred_zoom, where=(y_pred_zoom >= y_demand_zoom),
                 interpolate=True, color='#27ae60', alpha=0.2)
ax2.fill_between(t_zoom, y_demand_zoom, y_pred_zoom, where=(y_pred_zoom < y_demand_zoom),
                 interpolate=True, color='#c0392b', alpha=0.2)

# 3. 装饰
ax2.set_title(f'NV-ETO: Zoomed Detail (First {zoom_len} Samples)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Time Step / Index', fontsize=14)
ax2.set_ylabel('Quantity', fontsize=14)
ax2.set_xlim(0, zoom_len)
ax2.grid(True, linestyle='--', alpha=0.6)  # 网格线虚线化

# 调整整体布局防止重叠
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # 调整两个图之间的间距

plt.show()
