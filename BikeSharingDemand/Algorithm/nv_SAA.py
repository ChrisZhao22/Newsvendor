import numpy as np
import pandas as pd
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

target_col = 'demand'

# SAA 只需要目标变量 (Demand)，不需要特征矩阵
y_train = train_df[target_col].values
y_test = test_df[target_col].values

n_train = len(y_train)
n_test = len(y_test)

print(f"数据加载完成. 训练集: {n_train}, 测试集: {n_test}")

# ==========================================
# 2. 参数设置
# ==========================================
b = 30  # 缺货成本
h = 10  # 持有成本
ratio = b / (b + h)  # 目标分位数 (Target Quantile) = 0.75

print(f"参数: b={b}, h={h} => Target Quantile (Ratio)={ratio:.2f}")

# ==========================================
# 3. 模型训练 (Static SAA)
# ==========================================
print(f"\n开始计算 SAA 最优订货量...")
start_time = time.time()

# SAA 核心逻辑：直接取历史数据的分位数
# Q* = F^(-1)(ratio)
# 这相当于最小化样本内的报童损失
optimal_Q = np.quantile(y_train, ratio)

train_time = time.time() - start_time
print(f"计算完成. 耗时: {train_time:.6f} s")
print(f"SAA 建议的固定订货量: {optimal_Q:.2f}")

# ==========================================
# 4. 测试集评估
# ==========================================
# SAA 策略：每一天都订同样的货
Q_pred = np.full(n_test, optimal_Q)

# 计算成本
# 向量化计算，比循环快
cost_h = (Q_pred - y_test) * h
cost_b = (y_test - Q_pred) * b
# 取 max(0, x)
out_of_sample_cost = np.where(Q_pred >= y_test, cost_h, cost_b)

avg_cost = np.mean(out_of_sample_cost)
print(f"\n测试集平均成本: {avg_cost:.2f}")

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = '../data/output_data/nv_SAA.csv'

df_out = pd.DataFrame({
    'Actual_Demand': y_test,
    'Predicted_Order': Q_pred,
    'Realized_Cost': out_of_sample_cost
})
df_out.to_csv(output_filename, index=False)
print(f"结果已保存至 {output_filename}")

# ==========================================
# 6. 可视化
# ==========================================
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(14, 6))

t = np.arange(n_test)

# 1. 真实需求
plt.plot(t, y_test, label='Actual Demand', color='#2c3e50', alpha=0.8, marker='.', linestyle='-')

# 2. 预测订货量 (一条水平线)
plt.plot(t, Q_pred, label=f'Optimal Order (SAA, Q={optimal_Q:.0f})', color='#e74c3c', linewidth=2.5, linestyle='--')

# 3. 填充成本区域
plt.fill_between(t, y_test, Q_pred, where=(Q_pred >= y_test),
                 interpolate=True, color='#27ae60', alpha=0.2, label='Over-stocking')
plt.fill_between(t, y_test, Q_pred, where=(Q_pred < y_test),
                 interpolate=True, color='#c0392b', alpha=0.2, label='Under-stocking')

plt.title(f'NV-SAA (Sample Average Approximation)\nAvg Cost: {avg_cost:.2f} | Static Policy', fontsize=16)
plt.ylabel('Quantity')
plt.xlabel('Test Sample Index')
plt.legend()
plt.tight_layout()
plt.show()