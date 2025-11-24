import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import time
import scipy.io
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

Features = np.column_stack((DayC, TimeC, Time))

# 读取统一的参数
lntr = int(data['lntr'][0][0])
lnva = int(data['lnva'][0][0])
lnte = int(data['lnte'][0][0])
TOTAL_LEN = len(Demand)

print(f"✅ 已加载统一数据: ground_truth.mat (Total: {TOTAL_LEN})")


# ==========================================
# 2. 参数设置
# ==========================================
lntr = 12 * 7 * 16  # 训练窗口长度 (Training size)
lnva = int(lntr / 2)     # 验证窗口长度 (Validation size)

delay = 3           # 决策延迟
pvec = [0]          # 对应 pvec = 12*[0] (简化演示)

# 报童模型成本参数
b = 2.5 / 3.5
h = 1 / 3.5
r = b / (b + h)     # 临界分位数 (Critical fractile)

# ==========================================
# 3. 辅助函数定义
# ==========================================
def nv_cost(q, d, b, h):
    """计算报童模型成本 (Newsvendor Cost)"""
    # max(d-q, 0)*b + max(q-d, 0)*h
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h

# ==========================================
# 4. 主循环逻辑 (Estimate then Optimize)
# ==========================================

for p in pvec:
    print(f"Processing p = {p}")
    
    # 初始化结果存储容器
    Valfac = np.zeros(lnva)
    ResOpt = np.zeros(lnva)
    
    lF = Features.shape[1] # 特征数量
    
    # 存储系数和预测值
    # Qfac 对应 beta0 (intercept) 和 beta1 (coefs)
    beta0_history = np.zeros(lnva)
    beta1_history = np.zeros((lnva, lF))
    
    muD = np.zeros(lnva)
    sigmaD = np.zeros(lnva)
    QfacD = np.zeros(lnva)
    
    start_time = time.time()
    # Python 索引从 0 开始，所以对应范围要平移
    # 假设数据足够长，这里模拟滑动窗口
    
    # t 的含义：当前预测的时间点索引
    # 训练数据范围：[t-lntr, t) (不包含 t)
    for i in range(lnva):
        t = lntr + i 
        
        if i % 100 == 0:
            print(f"Step {i}/{lnva}")
            
        # --------------------------------------
        # A. 准备训练数据
        # --------------------------------------
        # 特征窗口: t-lntr 到 t-1
        X_train_raw = Features[t-lntr : t, :]
        
        # 归一化: ./ norm(..., Inf)
        # MATLAB norm(matrix, Inf) 通常是指最大行和(maximum row sum)
        # 这里为了数值稳定性，我们计算整个矩阵的最大绝对值或行范数
        # 假设原意是缩放因子
        norm_val = np.linalg.norm(X_train_raw, ord=np.inf)
        if norm_val == 0: norm_val = 1.0
        
        X_train = X_train_raw / norm_val
        
        # 目标 y (Demand)
        # 目标值相对于特征有 'delay' 的偏移
        y_train = Demand[t - lntr + delay : t + delay]
        
        # --------------------------------------
        # B. 第一步回归：均值估计 (Mean Estimation)
        # --------------------------------------
        model_mean = LinearRegression(fit_intercept=True)
        model_mean.fit(X_train, y_train)
        
        beta0 = model_mean.intercept_
        beta1 = model_mean.coef_
        
        # 计算残差
        residuals = y_train - model_mean.predict(X_train)
        
        # --------------------------------------
        # C. 第二步回归：方差估计 (Variance Estimation)
        # --------------------------------------
        # 目标: log(Res^2)
        # 注意：为了防止 log(0)，通常加一个极小值，或者假设残差不为0
        y_train_var = np.log(residuals**2 + 1e-8)
        
        model_var = LinearRegression(fit_intercept=True)
        model_var.fit(X_train, y_train_var)
        
        delta0 = model_var.intercept_
        delta1 = model_var.coef_
        
        # --------------------------------------
        # D. 预测与优化 (Prediction & Optimization)
        # --------------------------------------
        # 当前时间点的特征 (用于预测)
        # 注意：必须使用训练时的 norm_val 进行相同的缩放
        X_current = Features[t, :].reshape(1, -1) / norm_val
        
        # 预测均值 mu
        mu_pred = model_mean.predict(X_current)[0]
        
        # 预测方差 sigma (log-linear 恢复)
        # MATLAB: exp((delta0 + delta1*feat)/2)
        log_sigma2_pred = model_var.predict(X_current)[0]
        sigma_pred = np.exp(log_sigma2_pred / 2)
        
        # 存储结果
        muD[i] = mu_pred
        sigmaD[i] = sigma_pred
        
        # 计算最优订货量 Q
        # Python norm.ppf 对应 MATLAB norminv
        # Q = mu + sigma * z_score
        z_score = norm.ppf(r)
        optimal_Q = mu_pred + sigma_pred * z_score
        QfacD[i] = optimal_Q
        
        # 计算实际成本
        # 使用 t+delay 时刻的真实需求来验证
        actual_demand = Demand[t + delay]
        Valfac[i] = nv_cost(optimal_Q, actual_demand, b, h)
        
        # 保存系数
        beta0_history[i] = beta0
        beta1_history[i, :] = beta1

    end_time = time.time()
    print(f"Loop finished in {end_time - start_time:.2f} seconds")

# ==========================================
# 5. 保存结果
# ==========================================
# 使用 numpy save 或 scipy.io.savemat 保存为 .mat 文件
import scipy.io

output_filename = f'nv_emerg_estopt_os_{delay}_lntr_{lntr}_lnva_{lnva}_python.mat'

results = {
    'Valfac': Valfac,
    'QfacD': QfacD,
    'muD': muD,
    'sigmaD': sigmaD,
    'p': p,
    'lntr': lntr,
    'lnva': lnva
}

scipy.io.savemat(output_filename, results)
print(f"Results saved to {output_filename}")