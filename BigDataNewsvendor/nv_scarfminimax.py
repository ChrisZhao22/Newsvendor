import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
delay = 3
# pvec = 12 * [10, 12, 14, 16, 18, 20, 22, 24]
# pvec = [120, 144, 168, 192, 216, 240, 264, 288]
pvec = [120]

b = 2.5 / 3.5
h = 1 / 3.5
r = b / (b + h)

# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h

# ==========================================
# 4. 主循环 (Outer loop over pvec)
# ==========================================

for p in pvec:
    print(f"\nProcessing Feature Lag p = {p}")
    
    # 构建特征矩阵 Features
    if p == 0:
        Features_Raw = np.hstack((DayC, Time))
    else:
        # MATLAB: [DayC, Time, Past(:,[1:p])]
        Features_Raw = np.hstack((DayC, Time, Past[:, :p]))
        
    lF = Features_Raw.shape[1]
    
    # 初始化结果存储
    Valfac = np.zeros(lnva)
    QfacD = np.zeros(lnva)
    muD = np.zeros(lnva)
    sigmaD = np.zeros(lnva)
    
    # 记录系数 (beta0 + beta1)
    # Qfac 存储 beta0 和 beta1
    Qfac_coefs = np.zeros((lnva, 1 + lF))
    ResOpt = np.zeros(lnva)

    start_time = time.time()
    
    # --------------------------------------
    # 滚动预测循环
    # --------------------------------------
    
    for k in range(lnva):
        t = lntr + k
        
        if k % 100 == 0:
            print(f"  Step {k}/{lnva}")
            
        # --------------------------------------
        # A. 准备数据与归一化
        # --------------------------------------
        # 训练窗口: [t-lntr, t)
        X_train_raw = Features_Raw[t-lntr : t, :]
        
        # 归一化 (Inf norm)
        # 沿列取最大绝对值作为缩放因子
        scale_factor = np.max(np.abs(X_train_raw), axis=0)
        # 防止除以0
        scale_factor[scale_factor == 0] = 1.0
        
        X_train = X_train_raw / scale_factor
        
        # 目标 y (考虑 delay)
        # Demand(t-lntr+delay : t+delay) -> Length lntr
        y_train = Demand[t - lntr + delay : t + delay]
        
        # --------------------------------------
        # B. 第一步回归: 均值 (Mean)
        # --------------------------------------
        model_mean = LinearRegression(fit_intercept=True)
        model_mean.fit(X_train, y_train)
        
        beta0 = model_mean.intercept_
        beta1 = model_mean.coef_
        
        residuals = y_train - model_mean.predict(X_train)
        res_sq = residuals**2

        ResOpt[k] = np.sum(res_sq)
        
        # --------------------------------------
        # C. 第二步回归: 方差 (Variance)
        # --------------------------------------
        # Log-linear variance estimation
        y_train_var = np.log(res_sq + 1e-8) # 加微小值防止 log(0)
        
        model_var = LinearRegression(fit_intercept=True)
        model_var.fit(X_train, y_train_var)
        
        delta0 = model_var.intercept_
        delta1 = model_var.coef_
        
        # --------------------------------------
        # D. 预测与优化 (Prediction & Scarf Rule)
        # --------------------------------------
        # 当前特征 (t)
        X_current = Features_Raw[t, :] / scale_factor
        X_current = X_current.reshape(1, -1)
        
        # 预测 mu
        mu_val = model_mean.predict(X_current)[0]
        
        # 预测 sigma
        log_sigma2 = model_var.predict(X_current)[0]
        sigma_val = np.exp(log_sigma2 / 2) # exp(log(s^2)/2) = s
        
        # 记录 mu, sigma
        muD[k] = mu_val
        sigmaD[k] = sigma_val
        
        # 计算 Scarf 订货量 Q
        scarf_term = (np.sqrt(b) - 1.0/np.sqrt(b))
        q_star = mu_val + (sigma_val / 2.0) * scarf_term
        
        QfacD[k] = q_star
        
        # 记录系数 (beta0, beta1)
        Qfac_coefs[k, 0] = beta0
        Qfac_coefs[k, 1:] = beta1
        
        # --------------------------------------
        # E. 计算成本 (Validation)
        # --------------------------------------
        actual_demand = Demand[t + delay]
        Valfac[k] = nv_cost(q_star, actual_demand, b, h)

    loop_time = time.time() - start_time
    print(f"  Finished p={p} in {loop_time:.2f}s")
    
    # ==========================================
    # 5. 保存当前 p 的结果
    # ==========================================
    filename = f'nv_emerg_scarf_de_{delay}_lntr_{lntr}_lnva_{lnva}_p_{p}_python.mat'
    
    save_dict = {
        'Valfac': Valfac,
        'QfacD': QfacD,
        'muD': muD,
        'sigmaD': sigmaD,
        'ResOpt': ResOpt,
        'Qfac': Qfac_coefs
    }
    
    scipy.io.savemat(filename, save_dict)
    print(f"  Saved {filename}")

print("\nAll processing complete.")