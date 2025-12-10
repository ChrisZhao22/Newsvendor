import numpy as np
import pandas as pd
import json
import cvxpy as cp
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
b = 3 / 4
h = 1 / 4

# ==========================================
# 4. 主循环逻辑
# ==========================================
coef = np.zeros((lnte, 1 + feat_dim))  # 存储系数
Q_pred = np.zeros(lnte)  # 存储决策
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
        Q_pred[k] = decision_q

        if t < TOTAL_LEN:
            actual = Demand[t + delay]
            out_of_sample_cost[k] = np.maximum(actual - decision_q, 0) * b + np.maximum(decision_q - actual, 0) * h
print(f"Total time: {time.time() - start_time:.2f} s")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_ERM.csv'

data_dict = {
    'Decision_Q': Q_pred,
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
ax1.set_title('NV-ERM: Newsvendor Decision Analysis: Global Overview', fontsize=18, fontweight='bold', pad=15)
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
ax2.set_title(f'NV-ERM: Zoomed Detail (First {zoom_len} Samples)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Time Step / Index', fontsize=14)
ax2.set_ylabel('Quantity', fontsize=14)
ax2.set_xlim(0, zoom_len)
ax2.grid(True, linestyle='--', alpha=0.6)  # 网格线虚线化

# 调整整体布局防止重叠
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # 调整两个图之间的间距

plt.show()
