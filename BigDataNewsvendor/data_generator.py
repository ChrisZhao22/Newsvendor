import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# ==========================================
# 统一数据生成器 (Ground Truth Generator)
# ==========================================
np.random.seed(2025) # 固定随机种子，保证每次生成的一样

TOTAL_LEN = 5000
lntr = 1344
lnva = 672
lnte = 672

# 1. 生成特征
# -----------------------------
# DayC: 星期几 (1-7)
DayC = np.random.randint(1, 8, (TOTAL_LEN, 1))
# Time: 时间趋势 (0-100)
Time = np.linspace(0, 100, TOTAL_LEN).reshape(-1, 1)
# TimeC: 一天内的时间段 (1-24)
TimeC = np.random.randint(1, 25, (TOTAL_LEN, 1))

# Past: 历史特征 (模拟一些滞后变量)
Past = np.random.randn(TOTAL_LEN, 300)

# 2. 生成需求 (构造一个复杂的真实函数)
# -----------------------------
# 基础线性部分
base_demand = 50 + 2 * DayC.flatten() + 0.2 * Time.flatten()

# 非线性部分 (比如周期性波动，模拟淡旺季)
seasonality = 10 * np.sin(Time.flatten() / 10)

# 异方差噪音 (Heteroscedasticity)
# 随着时间推移(Time变大)，波动(方差)越来越大
# 噪音标准差 sigma 从 5 线性增加到 15
noise_sigma = 5 + 0.1 * Time.flatten()
noise = np.random.normal(0, 1, TOTAL_LEN) * noise_sigma

# 最终需求
Demand = base_demand + seasonality + noise
Demand = np.maximum(Demand, 0) # 需求不能为负

# 3. 可视化检查
# -----------------------------
plt.figure(figsize=(12, 4))
plt.plot(Demand[:1000], label='First 1000 days Demand')
plt.title(f'Generated Data (Mean: {Demand.mean():.2f}, Std: {Demand.std():.2f})')
plt.legend()
plt.show()

# 4. 保存为统一的数据文件
# -----------------------------
filename = 'ground_truth.mat'
data = {
    'Demand': Demand,
    'DayC': DayC,
    'Time': Time,
    'TimeC': TimeC,
    'Past': Past,
    # 顺便把切分参数也存进去，保证大家切分一致
    'lntr': lntr,
    'lnva': lnva,
    'lnte': lnte
}

scipy.io.savemat(filename, data)
print(f"统一数据已保存至: {filename}")