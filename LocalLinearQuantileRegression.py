import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def calculate_hp(h_mean, p):
    """
    根据均值带宽 h_mean 计算特定分位数 p 对应的带宽 h_p。
    (section 2.3 中的实用公式) 。

    h_p = h_mean * { [p(1-p)] / [phi(Phi^-1(p))^2] }^(1/5)

    Args:
        h_mean (float): 用于均值回归的基础带宽。
        p (float): 分位数水平 (0 < p < 1)。

    Returns:
        float: 特定分位数的带宽 h_p。
    """
    if not (0 < p < 1):
        raise ValueError("p 必须在 0 和 1 之间")

    # 处理边缘情况下的数值不稳定性
    if p < 1e-10: p = 1e-10
    if p > 1 - 1e-10: p = 1 - 1e-10

    # Phi^-1(p) 求分位数
    norm_quantile = norm.ppf(p)

    # phi(Phi^-1(p)) 求分位数点的概率密度
    norm_pdf_at_quantile = norm.pdf(norm_quantile)

    # 分子: p(1-p)
    numerator = p * (1 - p)

    # 分母: [phi(Phi^-1(p))]^2
    denominator = norm_pdf_at_quantile ** 2

    # 指数内的项
    ratio = numerator / denominator

    # h_p
    h_p = h_mean * (ratio ** 0.2)

    return h_p


def check_function(u, p):
    """
    "检查函数" rho_p(u) 。
    rho_p(u) = p * u if u >= 0
             = (p - 1) * u if u < 0
    """
    return np.where(u >= 0, p * u, (p - 1) * u)


def kernel_function(u):
    """
    标准正态核函数 K(u) 。
    """
    return norm.pdf(u)


def local_linear_quantile_regression(X, Y, p, h_mean):
    """
    对所有 X 执行局部线性分位数回归。

    该函数为输入 X 中的每个 x 估计 q_p(x)。

    Args:
        X (np.array): 1D 特征数组。
        Y (np.array): 1D 目标值数组。
        p (float): 分位数水平 (0 < p < 1)。
        h_mean (float): 均值回归的基础带宽。
                        必须预先计算好。

    Returns:
        np.array: X 中每个 x 对应的估计分位数 q_p(x)。
    """

    n = len(X)
    q_p_estimates = np.zeros(n)

    # 1. 计算特定分位数的带宽 h_p
    h_p = calculate_hp(h_mean, p)
    if h_p == 0:
        raise ValueError("带宽 h_p 为零。请检查 h_mean 和 p。")

    print(f"为 p = {p} 计算 h_p = {h_p:.4f} (基于 h_mean = {h_mean})")

    # 2. 遍历每个点 x_i 来估计 q_p(x_i)
    for i in range(n):
        x = X[i]  # 当前需要估计分位数的点 x

        # 定义需要最小化的目标函数 (公式 4)
        # 需要优化的参数是 params = [a, b]
        def objective_function(params):
            a, b = params

            # Y_i - a - b(X_i - x)
            residuals = Y - a - b * (X - x)

            # rho_p( Y_i - a - b(X_i - x) )
            check_loss = check_function(residuals, p)

            # K( (x - X_i) / h_p )
            weights = kernel_function((x - X) / h_p)

            # 加权 check loss 的总和
            weighted_loss = np.sum(check_loss * weights)

            return weighted_loss

        # 使用数值优化器 (如 'Nelder-Mead') 来找到最小化损失的 a 和 b
        # 需要一个初始猜测值。使用 x 附近的 Y 中位数作为 'a' 的初始值
        local_indices = np.abs(X - x) < h_p # 表示我们只用
        if np.any(local_indices):
            initial_a = np.median(Y[local_indices])
        else:
            initial_a = np.median(Y)
        initial_params = [initial_a, 0.0]  # 假设初始斜率 b=0

        result = minimize(objective_function, initial_params, method='Nelder-Mead')

        if result.success:
            # 估计的分位数 q_p(x) 就是优化后的 'a' 参数
            q_p_estimates[i] = result.x[0]
        else:
            # 处理最小化失败的情况
            print(f"警告: 在 x = {x:.2f} 处最小化失败")
            q_p_estimates[i] = np.nan

    return q_p_estimates