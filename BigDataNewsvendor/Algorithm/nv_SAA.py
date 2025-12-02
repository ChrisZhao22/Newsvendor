import numpy as np
import pandas as pd
import json
import time
import os

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

# 读取 CSV
df = pd.read_csv(data_file)
Demand = df['Demand'].values

# 读取配置
with open(config_file, 'r') as f:
    config = json.load(f)

lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']
TOTAL_LEN = len(Demand)

print(f"已加载数据: {data_file} (Total: {TOTAL_LEN})")

# ==========================================
# 2. 参数设置
# ==========================================
b = 2.5 / 3.5
h = 1 / 3.5
r = b / (b + h)

# 结果存储
QSAA = np.zeros(lnte)
CostSAA = np.zeros(lnte)
TestSAA = np.zeros(lnte)


# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环逻辑 (SAA)
# ==========================================
print(f"Start SAA Loop (Target Quantile: {r:.4f})")
start_time = time.time()

start_idx = lntr + lnva # test beginning index

for k in range(lnte):
    t = start_idx + k

    # 获取训练数据 (滚动窗口)
    demand_train = Demand[t - lntr: t]

    # SAA 求解
    q0 = np.quantile(demand_train, r)
    q0 = max(0, q0)

    # 计算样本内成本
    in_sample_costs = nv_cost(q0, demand_train, b, h)
    avg_in_sample_cost = np.mean(in_sample_costs)

    # 记录
    QSAA[k] = q0
    CostSAA[k] = avg_in_sample_cost

    # 样本外测试
    if t < TOTAL_LEN:
        actual_demand = Demand[t]
        TestSAA[k] = nv_cost(q0, actual_demand, b, h)

end_time = time.time()
print(f"Loop finished in {end_time - start_time:.4f} seconds")

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = f'../data/nv_SAA.csv'

# 构建 DataFrame
df_out = pd.DataFrame({
    'Q_Decision': QSAA,
    'Demand_D': Demand[start_idx:start_idx+lnte],
    'InSample_Cost': CostSAA,
    'OutOfSample_Cost': TestSAA
})

df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")