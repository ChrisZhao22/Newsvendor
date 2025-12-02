import numpy as np
import scipy.io
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# Data Generator
# ==========================================
np.random.seed(2025)

TOTAL_LEN = 10000
lntr = 1344*2 # train
lnva = 672*2  # validation
lnte = 672*2  # test

# 1. 生成特征
# -----------------------------
# DayC: 星期几 (1-7)
DayC = np.random.randint(1, 8, (TOTAL_LEN, 1))
# Time: 时间趋势 (0-100)
Time = np.linspace(0, 100, TOTAL_LEN).reshape(-1, 1)

print(f"DayC shape: {DayC.flatten().shape}")

# 2. Demand Generator (复杂化：满足 Hölder ~ 2.3)
# -----------------------------

# 为了方便计算非线性函数，先将 Time 归一化到 [0, 1] 和 [-0.5, 0.5]
t_norm = (Time.flatten() - Time.min()) / (Time.max() - Time.min()) # [0, 1]
t_centered = t_norm - 0.5 # [-0.5, 0.5]

# --- 构造 Component 1: Hölder 特征项 ---
# 核心：使用 |x|^2.3 构造严格符合 Hölder 系数的要求
# 在 t=0.5 (即中间点) 处，二阶导数存在但三阶导数发散
holder_term = 40 * np.abs(t_centered) ** 2.3

# --- 构造 Component 2: 平滑的非线性趋势 (Polynomial) ---
# 二次函数部分，属于 C_infinity，不影响 Hölder 限制
smooth_trend = 50 + 20 * t_norm + 10 * (t_norm ** 2)

# --- 构造 Component 3: 特征交互 (Interaction) ---
# 让 DayC 对 Demand 的影响不再是线性的，而是随时间变化的
# 例如：周末(DayC大)在旺季(sin波峰)销量更高
day_normalized = DayC.flatten() / 7.0 # 归一化到 [0.14, 1]
interaction = 15 * np.sin(4 * np.pi * t_norm) * (day_normalized ** 2)

# --- 合成基础均值 ---
# Mean Demand
mu_demand = smooth_trend + holder_term + interaction

# --- 异方差噪音 (Heteroscedasticity) ---
# 噪音标准差随时间变得更复杂：不仅随时间增大，还受 DayC 影响
# 例如：周末的波动比工作日大
noise_level_base = 5 + 5 * t_norm
noise_level_day = 2 * day_normalized
sigma_total = noise_level_base + noise_level_day

noise = np.random.normal(0, 1, TOTAL_LEN) * sigma_total

# --- 最终需求 ---
Demand = mu_demand + noise
Demand = np.maximum(Demand, 0) # 需求非负截断

# 3. 可视化检查
# -----------------------------
plt.figure(figsize=(12, 6))

# 子图1: 最终生成的 Demand
plt.subplot(2, 1, 1)
plt.plot(Demand[:500], label='First 500 samples', alpha=0.7)
plt.title(f'Complex Demand ($\mu={Demand.mean():.2f}, \sigma={Demand.std():.2f}$)')
plt.ylabel('Demand')
plt.legend()

# 子图2: 分解查看 Hölder 趋势项（验证函数形状）
plt.subplot(2, 1, 2)
plt.plot(t_centered, holder_term, label=r'Hölder Term: $|t-0.5|^{2.3}$', color='orange', linewidth=2)
plt.title('Underlying Trend Component (Satisfying $\\beta \\approx 2.3$)')
plt.xlabel('Normalized Time (Centered)')
plt.legend()

plt.tight_layout()
plt.show()

# 4. 保存为统一的数据文件
# -----------------------------
# 确保目录存在
output_dir = '../data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_matrix = np.hstack([
    DayC,
    Time,
    Demand.reshape(-1, 1)
])

# 定义列名
columns = ['DayC', 'Time', 'Demand']

# 创建 DataFrame
df = pd.DataFrame(data_matrix, columns=columns)

# 保存数据文件
csv_filename = os.path.join(output_dir, 'newsvendor_simple_data.csv')
df.to_csv(csv_filename, index=False)

print(f"数据已成功保存至: {csv_filename}")
print(f"数据维度: {df.shape}")
print(f"前5行预览:\n{df.head()}")

# ==========================================
# 5. 保存切分配置
# ==========================================
config = {
    'lntr': lntr,
    'lnva': lnva,
    'lnte': lnte,
    'description': 'Complex features with Holder smoothness ~ 2.3 (includes fractional power term)'
}
json_filename = os.path.join(output_dir, 'data_config.json')
with open(json_filename, 'w') as f:
    json.dump(config, f)

print(f"切分参数已保存至: {json_filename}")