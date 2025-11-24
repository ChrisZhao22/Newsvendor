import numpy as np
from scipy.stats import norm
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
delay = 3           # 决策延迟
p = 12 * 14         # 特征参数
# p = 0 # 你可以修改 p 来测试不同的特征组合

# 带宽向量 (Bandwidth vector)
bandvec = [0.08] 
Blen = len(bandvec)

# 报童模型成本参数
b = 2.5 / 3.5
h = 1 / 3.5
r = b / (b + h)     # 临界分位数 (Target Quantile)

# ==========================================
# 3. 特征构建逻辑
# ==========================================
if p == -36:
    Features_Raw = TimeC
elif p == -24:
    Features_Raw = DayC
elif p == -12:
    Features_Raw = np.hstack((DayC, TimeC))
elif p == 0:
    Features_Raw = np.hstack((DayC, TimeC, Time))
else:
    # 对应 Features = [DayC, TimeC, Time, Past(:,[1:p])]
    # 注意：Python切片是前闭后开，所以用 :p
    Features_Raw = np.hstack((DayC, TimeC, Time, Past[:, :p]))

lF = Features_Raw.shape[1]
print(f"Feature shape: {Features_Raw.shape}")

# ==========================================
# 4. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h

# ==========================================
# 5. 主循环逻辑 (Kernel Optimization)
# ==========================================

# 初始化结果矩阵
Valfac = np.zeros((Blen, lnte)) 
QfacD = np.zeros((Blen, lnte))

start_time = time.time()

for bi in range(Blen):
    bandwidth = bandvec[bi]
    print(f"Processing bandwidth: {bandwidth}")
    run_steps = 100 
    start_idx = lntr + lnva
    
    for k in range(run_steps):
        t = start_idx + k
        
        # --------------------------------------
        # A. 特征归一化 (Window Normalization)
        # --------------------------------------
        
        # 获取当前窗口的数据 (包含历史 + 当前这一行)
        # 窗口大小 = lntr + 1
        window_features = Features_Raw[t - lntr : t + 1, :].astype(float)
        
        # 计算每一列的 Inf 范数 (即绝对值的最大值)
        # axis=0 表示沿列计算
        norms = np.linalg.norm(window_features, ord=np.inf, axis=0)
        
        # 避免除以 0
        norms[norms == 0] = 1.0
        
        # 归一化
        FeaturesT = window_features / norms
        
        # 分离出“当前特征”和“历史特征”
        # 当前特征是最后一行 (索引 t)
        current_feat = FeaturesT[-1, :]
        # 历史特征是前 lntr 行 (索引 t-lntr 到 t-1)
        history_feats = FeaturesT[:-1, :]
        
        # --------------------------------------
        # B. 计算核权重 (Kernel Weights) - 向量化优化
        # --------------------------------------
        
        # 计算 history_feats 中每一行与 current_feat 的距离
        # axis=1 表示对每一行求范数
        dists = np.linalg.norm(history_feats - current_feat, axis=1)
        
        # 高斯核函数计算权重
        # kernel_weight = normpdf(dist / bandwidth)
        weights = norm.pdf(dists / bandwidth)
        
        # 归一化权重
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            weights_norm = np.ones(lntr) / lntr # 避免除以0，平均分配
        else:
            weights_norm = weights / weights_sum
            
        # --------------------------------------
        # C. 排序与分位数查找 (Sort & Quantile)
        # --------------------------------------
        demand_history = Demand[t - lntr + delay : t + delay]
        
        # 对需求进行排序
        sort_idx = np.argsort(demand_history)
        sDemand = demand_history[sort_idx]
        
        # 按照同样的顺序排列权重
        sWeights = weights_norm[sort_idx]
        
        # 计算累积分布 (CDF)
        kernel_cdf = np.cumsum(sWeights)
        
        # 找到第一个 CDF >= r 的位置
        # 使用 argmax(condition)
        idx_opt = np.argmax(kernel_cdf >= r)
        
        q0 = sDemand[idx_opt]
        
        # --------------------------------------
        # D. 记录结果与计算成本
        # --------------------------------------
        # 记录最优订货量 (注意索引偏移，QfacD 是从 0 开始存)
        
        save_idx = t - lntr #
        if save_idx < QfacD.shape[1]:
            QfacD[bi, save_idx] = q0
            
            # 计算真实成本
            actual_demand = Demand[t + delay]
            Valfac[bi, save_idx] = nv_cost(q0, actual_demand, b, h)

end_time = time.time()
print(f"Total time: {end_time - start_time:.4f} seconds")

# ==========================================
# 6. 保存结果
# ==========================================
output_filename = f'nv_kernelG_de2_{delay}_lntr_{lntr}_python.mat'

results = {
    'Valfac': Valfac,
    'QfacD': QfacD,
    'bandvec': bandvec,
    'p': p
}

scipy.io.savemat(output_filename, results)
print(f"Results saved to {output_filename}")