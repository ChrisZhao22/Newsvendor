import numpy as np
import scipy.io
import time

# ==========================================
# 1. 数据加载
# ==========================================
import os
data_file = 'ground_truth.mat'

if not os.path.exists(data_file):
    raise FileNotFoundError(f"找不到 {data_file}，请先运行 data_generator.py！")

data = scipy.io.loadmat(data_file)

# 读取变量
Demand = data['Demand'].flatten() # 确保是一维数组
DayC = data['DayC']
Time = data['Time']
TimeC = data['TimeC']
Past = data['Past']

# 读取统一的参数
lntr = int(data['lntr'][0][0])
lnva = int(data['lnva'][0][0])
lnte = int(data['lnte'][0][0])
TOTAL_LEN = len(Demand)

print(f"✅ 已加载统一数据: ground_truth.mat (Total: {TOTAL_LEN})")
# ==========================================

# ==========================================
# 2. 参数设置
# ==========================================
b = 30
h = 20
r = b / (b + h) # 临界分位数 (Critical Ratio)

lntr = 12 * 7 * 16  # 训练窗口长度
lnva = int(lntr / 2)
lnte = lnva         # 测试长度

# 结果存储
QSAA = np.zeros(lnte)
CostSAA = np.zeros(lnte) # 样本内成本 (In-sample cost)
TestSAA = np.zeros(lnte) # 样本外成本 (Out-of-sample cost)

# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    """计算报童成本"""
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h

# ==========================================
# 4. 主循环逻辑 (SAA)
# ==========================================
print(f"Start SAA Loop (Total steps: {lnte})")
print(f"Target Quantile: {r:.4f}")

start_time = time.time()

# 循环范围: t 对应测试集的每一个时间点
# MATLAB: for t=lntr+lnva+1 : lntr+lnva+lnte
start_idx = lntr + lnva

for k in range(lnte):
    t = start_idx + k
    
    if k % 100 == 0:
        print(f"Step {k}/{lnte}")

    # --------------------------------------
    # A. 获取训练数据
    # --------------------------------------
    # 对应 MATLAB: Demand(t-lntr : t-1)
    # Python 切片: [t-lntr : t]
    demand_train = Demand[t-lntr : t]
    
    # --------------------------------------
    # B. SAA 求解
    # --------------------------------------
    q0 = np.quantile(demand_train, r)
    
    # 确保 q0 >= 0 (虽然需求本身非负，q0自然非负)
    q0 = max(0, q0)
    
    # --------------------------------------
    # C. 计算样本内成本 (In-sample Cost)
    # --------------------------------------
    in_sample_costs = nv_cost(q0, demand_train, b, h)
    avg_in_sample_cost = np.mean(in_sample_costs)
    
    # --------------------------------------
    # D. 记录结果
    # --------------------------------------
    QSAA[k] = q0
    CostSAA[k] = avg_in_sample_cost
    
    # 计算样本外成本 (Out-of-sample Test)
    actual_demand = Demand[t]
    TestSAA[k] = nv_cost(q0, actual_demand, b, h)

end_time = time.time()
print(f"Loop finished in {end_time - start_time:.4f} seconds")

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = f'nv_emerg_SAA_lntr_{lntr}_lnte_{lnte}_python.mat'

results = {
    'QSAA': QSAA,
    'CostSAA': CostSAA,
    'TestSAA': TestSAA,
    'lntr': lntr,
    'lnte': lnte
}

scipy.io.savemat(output_filename, results)
print(f"Results saved to {output_filename}")