import numpy as np
import scipy.io
import json
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Data Generator
# ==========================================
np.random.seed(2025)

TOTAL_LEN = 5000
lntr = 1344 # train
lnva = 672 # validation
lnte = 672 # test

# 1. 生成特征
# -----------------------------
# DayC: 星期几 (1-7)
DayC = np.random.randint(1, 8, (TOTAL_LEN, 1))
# Time: 时间趋势 (0-100)
Time = np.linspace(0, 100, TOTAL_LEN).reshape(-1, 1)


print(DayC.flatten().shape)
# 2. Demand Generator
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
plt.plot(Demand[:188], label='First 188 days Demand')
plt.title(f'Generated Data (Mean: {Demand.mean():.2f}, Std: {Demand.std():.2f})')
plt.legend()
plt.show()

# 4. 保存为统一的数据文件
# -----------------------------
data_matrix = np.hstack([
    DayC,
    Time,
    Demand.reshape(-1, 1)
])

# 定义列名
columns = ['DayC', 'Time','Demand']

# 创建 DataFrame
df = pd.DataFrame(data_matrix, columns=columns)

# 保存数据文件
csv_filename = '../data/newsvendor_simple_data.csv'
df.to_csv(csv_filename, index=False)

print(f"数据已成功保存至: {csv_filename}")
print(f"数据维度: {df.shape}")
print(f"前5行预览:\n{df.head()}")

# ==========================================
# 4. 保存切分配置
# ==========================================
# 将切分参数单独保存，方便后续模型调用
config = {
    'lntr': lntr,
    'lnva': lnva,
    'lnte': lnte,
    'description': 'Simple feature set without Past variables'
}
json_filename = '../data/data_config.json'
with open(json_filename, 'w') as f:
    json.dump(config, f)

print(f"切分参数已保存至: {json_filename}")