import numpy as np
import pandas as pd
import json
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

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

print(f"已加载数据: {data_file}. Total Samples: {TOTAL_LEN}")
print(f"   Features Shape: {Features_Raw.shape}")
# ==========================================
# 2. 参数设置
# ==========================================
# 报童参数
delay = 0
b = 3 / 4
h = 1 / 4
alpha = b / (b + h)  # Target Quantile alpha

# 多项式阶数 k. PDF 定义 s(A) 为局部多项式的系数个数

smoothness_parameter = 2
poly_degree = np.floor(smoothness_parameter).astype(int)
# 训练样本数量
n = lntr
# 特征维数
p = Features_Raw.shape[1]
# 分箱指数及每一个维度分箱数
r = 1 / (2 * smoothness_parameter + p)
print('计算所得的分箱阶数：', (n / np.log(n)) ** r)
J_grid = round((n / np.log(n)) ** r)


# ==========================================
# 3. 辅助函数：分位数回归 LP 求解器
# ==========================================
def solve_local_quantile_regression(X_local, Y_local, x_test, x_center, delta_n, alpha, degree):
    """
    求解局部加权分位数回归
    Objective: min sum rho_alpha(Y_i - P_n(theta, X_i, x_center))
    """
    n_samples, n_features = X_local.shape

    # 1. 基础中心化/归一化特征 U_i = (X_i - x_{n,r}) / delta_n
    X_base = (X_local - x_center) / delta_n

    # 2. 构造多项式设计矩阵 (Polynomial Design Matrix)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_design = poly.fit_transform(X_base)  # 构造出多项式基函数
    num_params = X_design.shape[1]  # s(A): 系数总个数

    # 3. CVXPY线性规划形式建模+GUROBI求解
    # min sum(alpha * u_plus + (1-alpha) * u_minus)
    # s.t. Y - X*theta = u_plus - u_minus

    # 决策变量
    theta = cp.Variable(num_params)
    u_plus = cp.Variable(n_samples)
    u_minus = cp.Variable(n_samples)

    # 目标函数
    pinball_loss = cp.sum(alpha * u_plus + (1 - alpha) * u_minus)  # 标准分位数回归目标函数形式
    objective = cp.Minimize(pinball_loss / lntr)

    # 约束条件
    constraints = [
        Y_local == X_design @ theta + u_plus - u_minus,
        u_plus >= 0,
        u_minus >= 0,
    ]

    # 问题构建
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.GUROBI, verbose=False)
    except cp.SolverError:
        # 如果 Gurobi 失败，尝试默认求解器
        prob.solve(verbose=False)

    # 4. 预测 (Estimation)
    if prob.status == cp.OPTIMAL:
        theta_val = theta.value
        # a. 局部中心化/缩放测试点
        x_test_base = (x_test - x_center) / delta_n
        # b. 构造多项式设计向量
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_design_test = poly.fit_transform(x_test_base.reshape(1, -1))
        # c. 预测: y_pred = theta^T * X_design_test
        y_pred = np.dot(X_design_test, theta_val)[0]  # [0] 提取标量结果
        return y_pred
    else:
        return np.quantile(Y_local, alpha)


def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环 (Local Polynomial Estimation)
# ==========================================
Q_pred = np.zeros(lnte)
Cost_out_of_sample = np.zeros(lnte)
Cost_in_sample = np.zeros(lnte)

print(f"Start Local Polynomial Quantile Regression (Binning J={J_grid}, Degree={poly_degree})...")
start_time = time.time()

# 全局归一化 (参考范围仅限训练集)
global_feat_raw = Features_Raw
feat_min = np.min(global_feat_raw, axis=0)
feat_max = np.max(global_feat_raw, axis=0)
feat_middle = (feat_min + feat_max) / 2
feat_range = feat_max - feat_min


def normalize(X):
    return (X - feat_middle) / feat_range


Features_Norm = normalize(Features_Raw)
# Side length delta_n
delta_n = 1.0 / J_grid

start_idx = lntr + lnva

# 求解主循环
for k in range(lnte):
    t = start_idx + k

    if k % 50 == 0:
        print(f"Step {k}/{lnte}")

    # A. 准备数据
    X_train = Features_Norm[t - lntr: t]
    Y_train = Demand[t - lntr + delay: t]

    x_test = Features_Norm[t, :].reshape(1, -1)

    # B. 确定test样本所在分箱 (Binning)
    test_bin_indices = np.floor((x_test + 0.5) / delta_n).astype(int)
    test_bin_indices = np.clip(test_bin_indices, 0, J_grid - 1)

    # test样本所在分箱 Bin的中心点
    x_test_bin_center = (test_bin_indices + 0.5) * delta_n

    # C. 从当前滑动训练样本中筛选 Bin 内样本
    train_bin_indices = np.floor((X_train + 0.5) / delta_n).astype(int)
    train_bin_indices = np.clip(train_bin_indices, 0, J_grid - 1)
    in_bin_mask = np.all(train_bin_indices == test_bin_indices, axis=1)

    X_local_train = X_train[in_bin_mask]
    Y_local_train = Y_train[in_bin_mask]

    # D. 求解局部估计

    order_pred = 0
    poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
    min_samples_required = poly.fit_transform(x_test).shape[1]
    if len(Y_local_train) >= min_samples_required:
        order_pred = solve_local_quantile_regression(X_local_train, Y_local_train, x_test, x_test_bin_center, delta_n,
                                                     alpha, poly_degree)
    Q_pred[k] = order_pred
    Cost_in_sample[k] = np.mean(nv_cost(order_pred, Y_local_train, b, h))

    if t + delay < TOTAL_LEN:
        actual_demand = Demand[t + delay]
        Cost_out_of_sample[k] = nv_cost(order_pred, actual_demand, b, h)
end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_bin_smoother.csv'

df_out = pd.DataFrame({
    'Decision_Q': Q_pred,
    'Decision_D': Demand[start_idx:start_idx + lnte],
    'Out_of_sample_cost': Cost_out_of_sample,
    'Local_InSample_Cost': Cost_in_sample

})

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
ax1.set_title('NV-BinSmoother: Newsvendor Decision Analysis: Global Overview', fontsize=18, fontweight='bold', pad=15)
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
ax2.set_title(f'NV-BinSmoother: Zoomed Detail (First {zoom_len} Samples)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Time Step / Index', fontsize=14)
ax2.set_ylabel('Quantity', fontsize=14)
ax2.set_xlim(0, zoom_len)
ax2.grid(True, linestyle='--', alpha=0.6)  # 网格线虚线化

# 调整整体布局防止重叠
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # 调整两个图之间的间距

plt.show()
