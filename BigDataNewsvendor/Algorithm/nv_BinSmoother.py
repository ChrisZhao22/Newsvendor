import numpy as np
import pandas as pd
import json
import time
import os

from cvxpy import promote
from scipy.optimize import linprog
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

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

print(f"已加载数据: {data_file}. Total Samples: {TOTAL_LEN}")
print(f"   Features Shape: {Features_Raw.shape}")
# ==========================================
# 2. 参数设置
# ==========================================
# 报童参数
b = 2.5 / 3.5
h = 1 / 3.5
alpha = b / (b + h)  # Target Quantile alpha


# 多项式阶数 k. PDF 定义 s(A) 为局部多项式的系数个数
poly_degree = 2
# 训练样本数量
n = lntr
# 特征维数
p = Features_Raw.shape[1]
# 分箱指数及每一个维度分箱数
r = 1 / (2*poly_degree+p)
print('计算所得的分箱阶数：', (n / np.log(n))**r)
J_grid = round((n / np.log(n))**r)





# ==========================================
# 3. 辅助函数：分位数回归 LP 求解器
# ==========================================
def solve_local_quantile_regression(X_local, Y_local, x_center, delta_n, alpha, degree):
    """
    求解局部加权分位数回归
    Objective: min sum rho_alpha(Y_i - P_n(theta, X_i, x_center))
    """
    n_samples, n_features = X_local.shape

    # 1. 基础中心化/归一化特征 U_i = (X_i - x_{n,r}) / delta_n
    X_base = (X_local - x_center) / delta_n

    # 2. 构造多项式设计矩阵 (Polynomial Design Matrix)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_design = poly.fit_transform(X_base)

    num_params = X_design.shape[1]  # s(A): 系数总个数

    # 3. 线性规划形式
    # min sum(alpha * u_plus + (1-alpha) * u_minus)
    # s.t. Y - X*theta = u_plus - u_minus

    # 变量向量 c: [theta_plus(m), theta_minus(m), u_plus(n), u_minus(n)]
    c = np.concatenate([
        np.zeros(2 * num_params),  # theta 部分系数为 0
        alpha * np.ones(n_samples),  # u+ 系数
        (1 - alpha) * np.ones(n_samples)  # u- 系数
    ])

    # 等式约束: X*t+ - X*t- + u+ - u- = Y
    A_eq = np.hstack([
        X_design,
        -X_design,
        np.eye(n_samples),
        -np.eye(n_samples)
    ])
    b_eq = Y_local

    # 边界约束: theta >= 0 (实际上是分裂变量 >=0), u >= 0
    bounds = [(0, None)] * len(c)

    # 求解 LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        # print(res.message)
        # print(res)
        # 还原 theta = theta_plus - theta_minus
        theta_plus = res.x[:num_params]
        theta_minus = res.x[num_params: 2 * num_params]
        theta = theta_plus - theta_minus

        # 4. 预测 (Estimation)
        # 使用中心点代替来进行预测
        # y_pred = theta[0]
        # return y_pred

        # a. 局部中心化/缩放测试点
        x_test_base = (x_test_norm - x_center) / delta_n

        # b. 构造多项式设计向量
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        # 注意: fit_transform 需要二维输入，所以使用 reshape(1, -1)
        X_design_test = poly.fit_transform(x_test_base.reshape(1, -1))

        # c. 预测: y_pred = theta^T * X_design_test
        y_pred = np.dot(X_design_test, theta)[0]  # [0] 提取标量结果
        return y_pred
    else:
        # Fallback
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
train_feat_raw = Features_Raw[:lntr + lnva]
feat_min = np.min(train_feat_raw, axis=0)
feat_max = np.max(train_feat_raw, axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0


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
    X_train_raw = Features_Raw[t - lntr: t]
    Y_train = Demand[t - lntr: t]
    X_train_norm = normalize(X_train_raw)

    x_test_raw = Features_Raw[t, :].reshape(1, -1)
    x_test_norm = normalize(x_test_raw).flatten()

    # B. 确定分箱 (Binning)
    bin_indices = np.floor(x_test_norm / delta_n).astype(int)
    bin_indices = np.clip(bin_indices, 0, J_grid - 1)

    # Bin 中心点
    x_bin_center = (bin_indices + 0.5) * delta_n

    # C. 筛选 Bin 内样本
    train_bin_indices = np.floor(X_train_norm / delta_n).astype(int)
    train_bin_indices = np.clip(train_bin_indices, 0, J_grid - 1)
    in_bin_mask = np.all(train_bin_indices == bin_indices, axis=1)

    X_local = X_train_norm[in_bin_mask]
    Y_local = Y_train[in_bin_mask]

    # D. 求解局部估计
    q0 = 0

    # 计算当前 degree 下需要的最小样本数 (参数个数)
    # 对于 p=2, degree=2，参数个数为 6 (1, x1, x2, x1^2, x1x2, x2^2)
    poly_test = PolynomialFeatures(degree=poly_degree, include_bias=True)
    num_params_needed = poly_test.fit_transform(np.zeros((1, X_local.shape[1]))).shape[1]

    # if len(Y_local) >= min_samples_required:
    q0 = solve_local_quantile_regression(X_local, Y_local, x_bin_center, delta_n, alpha, poly_degree)


    # E. 记录
    Q_pred[k] = q0

    Cost_in_sample[k] = np.mean(nv_cost(q0, Y_local, b, h))

    if t < TOTAL_LEN:
        actual_demand = Demand[t]
        Cost_out_of_sample[k] = nv_cost(q0, actual_demand, b, h)
end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

# ==========================================
# 5. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_bin_smoother.csv'

df_out = pd.DataFrame({
    'Decision_Q': Q_pred,
    'Decision_D': Demand[start_idx:start_idx+lnte],
    'Out_of_sample_cost': Cost_out_of_sample,
    'Local_InSample_Cost': Cost_in_sample
})

df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")