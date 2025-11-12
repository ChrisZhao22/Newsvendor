import numpy as np
import time
from ERM import ERM_Newsvendor
from KO import KO_Newsvendor
from SAA import SAA_Newsvendor


# ----------------------------------------------------
# 1. 定义成本计算函数
# ----------------------------------------------------
def calculate_cost(q_decision, d_real, b, h):
    """
    计算给定决策 q 和真实需求 d 的报童成本。

    参数:
    q_decision (array): 模型的订单决策
    d_real (array): 对应的真实需求
    b (float): 缺货成本
    h (float): 积压成本

    返回:
    float: 平均报童成本
    """
    # C(q;D) = b(D-q)^+ + h(q-D)^+
    underage_cost = b * np.maximum(0, d_real - q_decision)
    overage_cost = h * np.maximum(0, q_decision - d_real)

    total_cost = underage_cost + overage_cost

    # 我们关心的是在测试集上的“平均样本外成本”
    return np.mean(total_cost)


# ----------------------------------------------------
# 2. 生成模拟数据
# ----------------------------------------------------
np.random.seed(42)
n_samples = 200
p_features = 2

# X 特征
X = np.random.rand(n_samples, p_features)

# 需求 d 是 X 的线性函数 + 噪声
# d = 5*x1 + 3*x2 + 10 + noise
d = 5 * X[:, 0] + 3 * X[:, 1] + 10 + np.random.randn(n_samples) * 2
# 确保需求非负
d[d < 0] = 0

# 划分训练集和测试集 (训练集:150, 测试集:50)
X_train, X_test = X[:150], X[150:]
d_train, d_test = d[:150], d[150:]
n_test = len(d_test)  # 测试集样本数

# ----------------------------------------------------
# 3. 定义报童问题参数
# ----------------------------------------------------
b_cost = 3  # 缺货成本
h_cost = 1  # 积压成本

# 这里对应3/4分位数
target_fractile = b_cost / (b_cost + h_cost)
print(f"目标分位数: {target_fractile:.2f}")

# ----------------------------------------------------
# 4. 运行 ERM-l2 算法 (NV-ERM2)
# ----------------------------------------------------
print("\n--- ERM (l2 正则化) 算法 ---")
erm_model = ERM_Newsvendor(b=b_cost, h=h_cost, regularization='l2', lambda_reg=0.01)

# 计时: 训练
start_fit = time.time()
erm_model.fit(X_train, d_train)
erm_fit_time = time.time() - start_fit
print(f"学到的系数 (q): {erm_model.q_}")

# 计时: 预测
start_pred = time.time()
erm_preds = erm_model.predict(X_test)
erm_pred_time = time.time() - start_pred
print(f"ERM 预测前5个: {erm_preds[:5].round(2)}")

# ----------------------------------------------------
# 5. 运行 KO 算法
# ----------------------------------------------------
print("\n--- KO (核权重) 算法 ---")
ko_model = KO_Newsvendor(b=b_cost, h=h_cost, bandwidth=0.5)

# 计时: 训练 (KO 是 lazy learner, 训练应该非常快)
start_fit = time.time()
ko_model.fit(X_train, d_train)
ko_fit_time = time.time() - start_fit

# 计时: 预测 (KO 在预测时进行大量计算)
start_pred = time.time()
ko_preds = ko_model.predict(X_test)
ko_pred_time = time.time() - start_pred
print(f"KO 预测前5个: {ko_preds[:5].round(2)}")

# ----------------------------------------------------
# 6. 运行 SAA 算法 (Benchmark)
# ----------------------------------------------------
print("\n--- SAA (基准) 算法 ---")
saa_model = SAA_Newsvendor(b=b_cost, h=h_cost)

# 计时: 训练
start_fit = time.time()
saa_model.fit(X_train, d_train)
saa_fit_time = time.time() - start_fit
print(f"学到的 SAA 订单量 (q): {saa_model.q_hat_n_:.2f}")

# 计时: 预测
start_pred = time.time()
saa_preds = saa_model.predict(X_test)
saa_pred_time = time.time() - start_pred
print(f"SAA 预测前5个: {saa_preds[:5].round(2)}")
print(f"(SAA 的所有 {n_test} 个预测值都是相同的)")

# 真实需求（用于比较）
print(f"\n真实需求前5个: {d_test[:5].round(2)}")

# ----------------------------------------------------
# 7. 计算时间对比
# ----------------------------------------------------
print("\n" + "=" * 30)
print("     计算时间对比 (秒)")
print("=" * 30)
print(f"         | {'训练 (fit)':<15} | {'预测 (predict)':<15}")
print(f"---------+-----------------+-----------------")
print(f"ERM      | {erm_fit_time:<15.6f} | {erm_pred_time:<15.6f}")
print(f"KO       | {ko_fit_time:<15.6f} | {ko_pred_time:<15.6f}")
print(f"SAA      | {saa_fit_time:<15.6f} | {saa_pred_time:<15.6f}")

# ----------------------------------------------------
# 8. 性能 (成本) 对比
# ----------------------------------------------------
erm_cost = calculate_cost(erm_preds, d_test, b_cost, h_cost)
ko_cost = calculate_cost(ko_preds, d_test, b_cost, h_cost)
saa_cost = calculate_cost(saa_preds, d_test, b_cost, h_cost)

print("\n" + "=" * 30)
print("     性能 (平均样本外成本) 对比")
print("=" * 30)
print(f"ERM 算法平均成本: {erm_cost:.4f}")
print(f"KO  算法平均成本: {ko_cost:.4f}")
print(f"SAA 算法平均成本: {saa_cost:.4f}")

print("\n" + "=" * 30)
print("     成本节约 (相对 SAA)")
print("=" * 30)
print(f"ERM 节约: {((saa_cost - erm_cost) / saa_cost * 100):.2f} %")
print(f"KO  节约: {((saa_cost - ko_cost) / saa_cost * 100):.2f} %")