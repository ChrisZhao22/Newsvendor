import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. 数据加载与预处理
# ==========================================
train_file = '../data/input_data/train_set.csv'
test_file = '../data/input_data/test_set.csv'

# 读取数据
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# 定义特征列和目标列
feature_cols = ['temp', 'humidity', 'windspeed', 'season', 'weather']
target_col = 'demand'

# 提取特征矩阵 (X) 和 目标向量 (Y)
X_train_raw = train_df[feature_cols].values
Y_train = train_df[target_col].values

X_test_raw = test_df[feature_cols].values
Y_test = test_df[target_col].values

# 归一化 (Min-Max Normalization)
# 注意：必须使用【训练集】的统计量来归一化【测试集】，防止数据泄露
feat_min = np.min(X_train_raw, axis=0)
feat_max = np.max(X_train_raw, axis=0)
feat_range = feat_max - feat_min
# 防止除以0
feat_range[feat_range == 0] = 1.0


def normalize(X):
    return (X - feat_min) / feat_range


X_train_norm = normalize(X_train_raw)
X_test_norm = normalize(X_test_raw)

n_train = len(X_train_norm)
n_test = len(X_test_norm)
p_dim = X_train_norm.shape[1]

print(f"数据加载完成.")
print(f"训练集样本数: {n_train}, 测试集样本数: {n_test}")
print(f"使用特征: {feature_cols}")

# ==========================================
# 2. 参数设置
# ==========================================
# 报童参数
b = 30  # 缺货成本 (Backorder Cost)
h = 10  # 持有成本 (Holding Cost)
alpha = b / (b + h)  # 目标分位数 (Target Quantile) 0.75

# 算法参数
smoothness_parameter = 1
if smoothness_parameter < 1:
    poly_degree = 1
else:
    poly_degree = int(np.floor(smoothness_parameter))


r = 1 / (2 * smoothness_parameter + p_dim)
J_val = (n_train / np.log(n_train)) ** r
print(f"J_val: {J_val}")
J_grid = max(1, round(J_val))  # 至少为1
J_grid = 1
print(f"\n--- 算法配置 ---")
print(f"多项式阶数 (Degree): {poly_degree}")
print(f"特征维数 (d): {p_dim}")
print(f"计算出的分箱数 (J_grid): {J_grid} (每个维度切成 {J_grid} 份)")
print(f"总箱子数 (Total Bins): {J_grid ** p_dim}")

# Side length delta_n
delta_n = 1.0 / J_grid


# ==========================================
# 3. 辅助函数：分位数回归 LP 求解器
# ==========================================
def solve_local_quantile_regression(X_local, Y_local, x_target, x_bin_center, delta, quantile, degree):
    """
    求解局部加权分位数回归
    """
    n_samples = X_local.shape[0]

    # 1. 局部中心化 (以 Bin 中心为原点)
    # 这一步是为了数值稳定性，并让截距项更有物理意义
    X_base = (X_local - x_bin_center) / delta

    # 2. 构造多项式特征
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_design = poly.fit_transform(X_base)
    num_params = X_design.shape[1]

    # 3. CVXPY 建模
    theta = cp.Variable(num_params)
    u_plus = cp.Variable(n_samples)
    u_minus = cp.Variable(n_samples)

    # Pinball Loss
    loss = cp.sum(quantile * u_plus + (1 - quantile) * u_minus)
    objective = cp.Minimize(loss)

    constraints = [
        Y_local == X_design @ theta + u_plus - u_minus,
        u_plus >= 0,
        u_minus >= 0
    ]

    prob = cp.Problem(objective, constraints)

    # 求解
    try:
        prob.solve(solver=cp.GUROBI, verbose=False)
    except:
        try:
            prob.solve(verbose=False)
        except:
            return None  # 求解彻底失败

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        theta_val = theta.value

        # 预测：对目标点 x_target 进行同样的变换
        x_target_base = (x_target - x_bin_center) / delta
        # 变成 (1, p) 形状
        x_target_design = poly.fit_transform(x_target_base.reshape(1, -1))

        y_pred = np.dot(x_target_design, theta_val)[0]
        return [y_pred,theta.value]
    else:
        return None


def nv_cost(q, d, b, h):
    # 标量计算
    if q >= d:
        return (q - d) * h
    else:
        return (d - q) * b


# ==========================================
# 4. 主循环 (对测试集进行预测)
# ==========================================
Q_pred = np.zeros(n_test)
fallback_idx = []
Cost_out_of_sample = np.zeros(n_test)

print(f"\n开始在测试集上进行预测 (Total: {n_test})...")
start_time = time.time()

success_count = 0
fallback_count = 0

# 预计算训练集的 Bin Indices，加速查找
# shape: (n_train, p_dim)
train_bin_indices = np.floor(X_train_norm / delta_n).astype(int)
train_bin_indices = np.clip(train_bin_indices, 0, J_grid - 1)

poly_dummy = PolynomialFeatures(degree=poly_degree, include_bias=True)

for i in range(n_test):
    # 当前测试样本
    x_test_curr = X_test_norm[i]
    y_test_curr = Y_test[i]

    # 1. 确定该测试样本属于哪个 Bin
    test_bin_idx = np.floor(x_test_curr / delta_n).astype(int)
    test_bin_idx = np.clip(test_bin_idx, 0, J_grid - 1)

    # Bin 的几何中心
    x_bin_center = (test_bin_idx + 0.5) * delta_n

    # 2. 在训练集中找到同属该 Bin 的样本 (Hard Binning)
    # 匹配逻辑：所有维度的 index 都必须相同
    in_bin_mask = np.all(train_bin_indices == test_bin_idx, axis=1)

    X_local = X_train_norm[in_bin_mask]
    Y_local = Y_train[in_bin_mask]

    # 3. 求解
    # 检查样本量是否足够支持多项式拟合
    # 特征维度扩展后的列数
    n_poly_features = poly_dummy.fit_transform(x_test_curr.reshape(1, -1)).shape[1]

    decision_q = 0

    if len(Y_local) >= n_poly_features + 1:  # 稍微多留一点余量
        pred = solve_local_quantile_regression(X_local, Y_local, x_test_curr, x_bin_center, delta_n, alpha, poly_degree)
        if i == 0:
            print('theta:',pred[1])
        if pred[0] is not None:
            decision_q = max(0, pred[0])  # 需求不能为负
            success_count += 1
        else:
            # 求解器失败，回退到全局分位数
            decision_q = np.quantile(Y_train, alpha)
            fallback_idx.append(i)
            fallback_count += 1
    else:
        # 样本不足 (空箱子)，回退到全局分位数
        # 这里实际上就是导致你之前困扰的地方。
        # 在高维+Hard Binning下，这里很容易发生。
        decision_q = np.quantile(Y_train, alpha)
        fallback_idx.append(i)
        fallback_count += 1

    Q_pred[i] = decision_q
    Cost_out_of_sample[i] = nv_cost(decision_q, y_test_curr, b, h)

end_time = time.time()
print(f"预测完成. 耗时: {end_time - start_time:.2f}s")
print(f"  - 局部回归成功 (Local Fit): {success_count} 次")
print(f"  - 回退到全局均值 (Fallback): {fallback_count} 次 (样本不足或求解失败)")

# ==========================================
# 5. 结果保存与可视化
# ==========================================
results_df = pd.DataFrame({
    'Actual_Demand': Y_test,
    'Predicted_Order': Q_pred,
    'Cost': Cost_out_of_sample
})
results_df.to_csv('../data/output_data/nv_BinSmoother.csv', index=False)
print("结果已保存到 nv_BinSmoother.csv")





# 可视化
plt.figure(figsize=(14, 6))
t = np.arange(n_test)

# 绘制真实需求
plt.plot(t, Y_test, label='Actual Demand (Test Set)', color='black', marker='o', markersize=4, linestyle='-', alpha=0.6)

# 绘制决策量
plt.plot(t, Q_pred, label='Optimal Order Quantity', color='#e74c3c', linewidth=2)

# 填充成本区域
plt.fill_between(t, Y_test, Q_pred, where=(Q_pred >= Y_test),
                 interpolate=True, color='green', alpha=0.2, label='Over-stocking (Holding Cost)')
plt.fill_between(t, Y_test, Q_pred, where=(Q_pred < Y_test),
                 interpolate=True, color='red', alpha=0.2, label='Under-stocking (Lost Sales)')

plt.title(f'Newsvendor Performance on Test Set\n(Success Rate: {success_count}/{n_test}, J={J_grid})', fontsize=14)
plt.xlabel('Test Sample Index')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 打印平均成本
avg_cost = np.mean(Cost_out_of_sample)
print(f"\n平均测试集成本 (Average Cost): {avg_cost:.2f}")

