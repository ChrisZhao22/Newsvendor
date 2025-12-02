import numpy as np
import pandas as pd
import json
from scipy.stats import norm
import time
import os

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

df = pd.read_csv(data_file)
Demand = df['Demand'].values
DayC = df['DayC'].values.reshape(-1, 1)
Time = df['Time'].values.reshape(-1, 1)
Features_Raw = np.hstack((DayC, Time))


with open(config_file, 'r') as f: config = json.load(f)
lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']
TOTAL_LEN = len(Demand)

print(f"已加载数据. Features shape: {Features_Raw.shape}")

# ==========================================
# 2. 参数设置
# ==========================================
bandvec = [0.08]  # 带宽列表
b = 2.5 / 3.5
h = 1 / 3.5
r = b / (b + h)


def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 5. 主循环逻辑
# ==========================================
results_dict = {}

start_time = time.time()

for bandwidth in bandvec:
    print(f"Processing bandwidth: {bandwidth}")

    Q_list = np.zeros(lnte)
    Cost_list = np.zeros(lnte)

    start_idx = lntr + lnva

    for k in range(lnte):
        t = start_idx + k

        # A. 特征归一化
        window_features = Features_Raw[t - lntr: t + 1, :].astype(float)
        norms = np.linalg.norm(window_features, ord=np.inf, axis=0)
        norms[norms == 0] = 1.0
        FeaturesT = window_features / norms

        current_feat = FeaturesT[-1, :]
        history_feats = FeaturesT[:-1, :]

        # B. 核权重
        dists = np.linalg.norm(history_feats - current_feat, axis=1)
        weights = norm.pdf(dists / bandwidth)
        weights_norm = weights / (np.sum(weights) if np.sum(weights) > 0 else 1.0)


        demand_h = Demand[t - lntr: t]
        # 对齐长度
        min_len = min(len(demand_h), len(weights_norm))
        demand_h = demand_h[:min_len]
        weights_n = weights_norm[:min_len]

        sort_idx = np.argsort(demand_h)
        sDemand = demand_h[sort_idx]
        sWeights = weights_n[sort_idx]

        kernel_cdf = np.cumsum(sWeights)
        idx_opt = np.argmax(kernel_cdf >= r)
        q0 = sDemand[idx_opt]

        # D. 记录
        Q_list[k] = q0
        actual = Demand[t]
        Cost_list[k] = nv_cost(q0, actual, b, h)

    # 将当前带宽的结果存入字典
    results_dict[f'Decision_Q_bw{bandwidth}'] = Q_list
    results_dict[f'Demand_D_bw{bandwidth}'] = Demand[start_idx:start_idx + lnte]
    results_dict[f'Cost_bw{bandwidth}'] = Cost_list
print(f"Total time: {time.time() - start_time:.4f} s")

# ==========================================
# 6. 保存结果 (CSV)
# ==========================================
output_filename = f'../data/nv_kernel.csv'

df_out = pd.DataFrame(results_dict)
df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")