import numpy as np
import cvxpy as cp
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
p = 0
delay = 3
lambda_val = 0.1  # 正则化参数 lambda (Python关键字回避)
is_lasso = True   # True=L1 (Lasso), False=L2 (Ridge)

# 报童参数
b = 2.5 / 3.5
h = 1 / 3.5
# r = b / (b + h)
# Cost = b * max(y - q, 0) + h * max(q - y, 0)

# ==========================================
# 3. 特征构建
# ==========================================
if p == 0:
    Features_Raw = np.hstack((DayC, TimeC, Time))
else:
    Features_Raw = np.hstack((DayC, TimeC, Time, Past[:, :p]))

lF = Features_Raw.shape[1]

# ==========================================
# 4. 主循环逻辑 (Regularized Optimization)
# ==========================================

# 初始化结果容器
Qfac = np.zeros((lnte, 1 + lF))
QfacD = np.zeros(lnte)
Valfac = np.zeros(lnte)
Costfac = np.zeros(lnte)

print(f"Start Optimization loop using CVXPY (Lasso={is_lasso})...")
start_time = time.time()

# 循环范围: t 对应测试集的每一个时间点
start_idx = lntr + lnva
for k in range(lnte):
    t = start_idx + k
    
    if k % 10 == 0:
        print(f"Step {k}/{lnte}")

    # --------------------------------------
    # A. 数据准备与归一化
    # --------------------------------------
    # 训练窗口: [t-lntr, t)
    X_train_raw = Features_Raw[t-lntr : t, :]
    
    # 归一化 (Inf Norm of the window)
    norm_val = np.linalg.norm(X_train_raw, ord=np.inf, axis=0) # 沿列取最大
    scale_factor = np.max(np.abs(X_train_raw))
    if scale_factor == 0: scale_factor = 1.0
    
    X_train = X_train_raw / scale_factor
    
    # 对应的需求 (考虑 delay)
    # Demand(t-lntr+delay : t+delay) -> 长度 lntr
    y_train = Demand[t - lntr + delay : t + delay]
    
    # --------------------------------------
    # B. CVXPY 建模
    # --------------------------------------
    # 变量
    q0 = cp.Variable(1)
    q = cp.Variable(lF)
    
    # 预测值 (向量化)
    # predictions shape: (lntr,)
    predictions = q0 + X_train @ q
    
    # 损失函数 (Newsvendor Cost / Pinball Loss)
    # residual = y_train - predictions
    # loss = b * pos(residual) + h * pos(-residual)
    underage = b * cp.maximum(y_train - predictions, 0)
    overage = h * cp.maximum(predictions - y_train, 0)
    empirical_risk = cp.sum(underage + overage) / lntr
    
    # 正则化项
    if is_lasso:
        regularization = lambda_val * cp.norm(q, 1)
    else:
        regularization = lambda_val * cp.norm(q, 2)**2
        
    # 目标函数
    objective = cp.Minimize(empirical_risk + regularization)
    
    # 约束条件 (Non-negative coefficients)
    constraints = [
        q0 >= 0,
        q >= 0
    ]
    
    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # --------------------------------------
    # C. 记录结果
    # --------------------------------------
    # 保存优化得到的最优值
    Costfac[k] = prob.value
    
    # 保存系数
    Qfac[k, 0] = q0.value
    Qfac[k, 1:] = q.value
    
    # --------------------------------------
    # D. 样本外测试 (Out-of-sample Test)
    # --------------------------------------
    # 当前时刻特征 (用于决策)
    X_current = Features_Raw[t, :] / scale_factor
    
    # 计算决策量 Q
    # Python 的 @ 是点积
    decision_q = q0.value + X_current @ q.value
    
    # 记录决策
    QfacD[k] = decision_q
    
    # 计算真实发生的成本
    actual_demand = Demand[t + delay]
    
    # nvcost 函数逻辑内联
    real_cost = 0
    if actual_demand > decision_q:
        real_cost = b * (actual_demand - decision_q)
    else:
        real_cost = h * (decision_q - actual_demand)
        
    Valfac[k] = real_cost

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = f'nv_emerg_reg_L1_{lambda_val}_python.mat'

results = {
    'Valfac': Valfac,
    'Costfac': Costfac,
    'QfacD': QfacD,
    'Qfac': Qfac,
    'lambda': lambda_val
}

scipy.io.savemat(output_filename, results)
print(f"Saved to {output_filename}")