import numpy as np
import pandas as pd
import json
import time
import os
from scipy.optimize import linprog

# ==========================================
# 1. 数据加载
# ==========================================
data_file = 'newsvendor_simple_data.csv'
config_file = 'data_config.json'

if not os.path.exists(data_file) or not os.path.exists(config_file):
    raise FileNotFoundError("找不到数据文件，请先运行 data_generator.py！")

# 读取 CSV
df = pd.read_csv(data_file)
Demand = df['Demand'].values

# 提取特征 (DayC, Time)
# PDF 中提到特征空间 X 是 p 维的 [cite: 6]
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
alpha = b / (b + h)  # Target Quantile alpha [cite: 36]

# 非参数设置
# PDF Section: Nonparametric Binning
# 我们需要定义 J_n (每个维度的分箱数)。
# 理论建议 J_n ~ (n/log n)^(1/(2*beta+p))。这里我们设一个启发式固定值。
# 对于 1300+ 样本，2维特征，设 J=5 (总共 5x5=25 个箱子) 是比较稳健的。
J_grid = 5

# 多项式阶数 k. PDF 建议 k = floor(beta).
# 我们使用 Local Linear (k=1)，这也是最常用的非参数回归形式。
poly_degree = 1


# ==========================================
# 3. 辅助函数：分位数回归 LP 求解器
# ==========================================
def solve_local_quantile_regression(X_local, Y_local, x_center, delta_n, alpha):
    """
    求解局部加权分位数回归 [cite: 92]
    Objective: min sum rho_alpha(Y_i - P_n(theta, X_i, x_center))
    """
    n_samples, n_features = X_local.shape

    # 构造设计矩阵 (Design Matrix) [cite: 77, 84]
    # Local Linear: [1, (x - x_c)/delta]
    # 归一化特征
    X_design = (X_local - x_center) / delta_n
    # 添加截距项 (Intercept)
    X_design = np.hstack([np.ones((n_samples, 1)), X_design])

    num_params = X_design.shape[1]  # s(A)

    # 线性规划形式
    # min sum(alpha * u_plus + (1-alpha) * u_minus)
    # s.t. Y - X*theta = u_plus - u_minus
    # theta 无约束 (拆分为 theta_plus - theta_minus)

    # 变量向量 c: [theta_plus(m), theta_minus(m), u_plus(n), u_minus(n)]
    # 长度: 2*m + 2*n

    c = np.concatenate([
        np.zeros(2 * num_params),  # theta 部分系数为 0
        alpha * np.ones(n_samples),  # u+ 系数
        (1 - alpha) * np.ones(n_samples)  # u- 系数
    ])

    # 等式约束 A_eq * x = b_eq
    # Y - X(t+ - t-) - u+ + u- = 0
    # => X*t+ - X*t- + u+ - u- = Y

    A_eq = np.hstack([
        X_design,
        -X_design,
        np.eye(n_samples),
        -np.eye(n_samples)
    ])
    b_eq = Y_local

    # 边界约束: theta >= 0, u >= 0
    bounds = [(0, None)] * len(c)

    # 求解 LP
    # method='highs' 是 scipy 较新的高性能求解器
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        # 还原 theta = theta_plus - theta_minus
        theta_plus = res.x[:num_params]
        theta_minus = res.x[num_params: 2 * num_params]
        theta = theta_plus - theta_minus

        # 预测值: P_n(theta, x_center, x_center)
        # 对于局部多项式，在中心点 x_center 处的预测值就是截距项 theta[0]
        # 因为 (x_center - x_center) = 0
        y_pred = theta[0]
        return y_pred
    else:
        # 如果 LP 求解失败（极少见），回退到样本分位数
        return np.quantile(Y_local, alpha)


def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环 (Local Polynomial Estimation)
# ==========================================
Q_pred = np.zeros(lnte)
Cost_realized = np.zeros(lnte)
Cost_in_sample = np.zeros(lnte)

print(f"Start Local Polynomial Quantile Regression (Binning J={J_grid})...")
start_time = time.time()

# 为了符合 PDF 的 [-0.5, 0.5] 假设或单位立方体假设 [cite: 55]
# 我们需要对特征进行全局归一化到 [0, 1]
# 注意：必须只使用训练集的统计量来归一化，防止数据泄露
train_feat_raw = Features_Raw[:lntr + lnva]  # 使用所有历史数据作为参考范围
feat_min = np.min(train_feat_raw, axis=0)
feat_max = np.max(train_feat_raw, axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0


# 归一化函数
def normalize(X):
    return (X - feat_min) / feat_range


# Side length delta_n
delta_n = 1.0 / J_grid

start_idx = lntr + lnva

for k in range(lnte):
    t = start_idx + k

    if k % 50 == 0:
        print(f"Step {k}/{lnte}")

    # A. 准备数据
    # ----------------
    # 滚动窗口训练数据
    X_train_raw = Features_Raw[t - lntr: t]
    Y_train = Demand[t - lntr: t]

    # 归一化特征到 [0, 1]
    X_train_norm = normalize(X_train_raw)

    # 当前测试点
    x_test_raw = Features_Raw[t, :].reshape(1, -1)
    x_test_norm = normalize(x_test_raw).flatten()

    # B. 确定分箱 (Binning)
    # ----------------
    # 确定测试点所在的 Bin 索引 (例如: [2, 3])
    # 索引 = floor(x / delta_n)
    bin_indices = np.floor(x_test_norm / delta_n).astype(int)
    # 边界处理：如果 x=1.0，会变成 J，需要限制在 J-1
    bin_indices = np.clip(bin_indices, 0, J_grid - 1)

    # Bin 的中心点 x_{n,r} [cite: 58]
    # center = (index + 0.5) * delta_n
    x_bin_center = (bin_indices + 0.5) * delta_n

    # C. 筛选 Bin 内样本 S_{n,r}
    # ----------------
    # 找到落在该 Bin 内的所有训练样本
    # 判断条件: bin_idx * delta <= x < (bin_idx+1) * delta
    # 简单做法：对所有训练样本计算其 Bin 索引，看是否与测试点一致
    train_bin_indices = np.floor(X_train_norm / delta_n).astype(int)
    train_bin_indices = np.clip(train_bin_indices, 0, J_grid - 1)

    # 匹配所有维度索引
    # all(axis=1) 表示该行的所有特征维度都必须匹配
    in_bin_mask = np.all(train_bin_indices == bin_indices, axis=1)

    X_local = X_train_norm[in_bin_mask]
    Y_local = Y_train[in_bin_mask]

    # D. 求解局部估计
    # ----------------
    q0 = 0
    min_samples_required = (X_train_norm.shape[1] + 1) * 2  # 经验值：至少样本数 > 参数数

    if len(Y_local) >= min_samples_required:
        # 样本充足，进行局部多项式回归
        q0 = solve_local_quantile_regression(X_local, Y_local, x_bin_center, delta_n, alpha)
    else:
        # Fallback: 如果 Bin 内样本太少（稀疏数据问题）
        # 策略：扩大搜索范围（最近邻）或使用全局 SAA
        # 这里为了稳健，如果局部为空，回退到全局 SAA
        # 这是一个常见的工程处理，PDF 中提到边界处理可用平均 [cite: 98]
        q0 = np.quantile(Y_train, alpha)

    q0 = max(0, q0)

    # E. 记录
    # ----------------
    Q_pred[k] = q0

    # 估算样本内成本 (仅作为参考，使用局部数据)
    if len(Y_local) > 0:
        Cost_in_sample[k] = np.mean(nv_cost(q0, Y_local, b, h))
    else:
        Cost_in_sample[k] = np.mean(nv_cost(q0, Y_train, b, h))

    # 样本外测试
    if t < TOTAL_LEN:
        actual_demand = Demand[t]
        Cost_realized[k] = nv_cost(q0, actual_demand, b, h)

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'nv_local_poly_J{J_grid}_python.csv'

df_out = pd.DataFrame({
    'Decision_Q': Q_pred,
    'Realized_Cost': Cost_realized,
    'Local_InSample_Cost': Cost_in_sample
})

df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")