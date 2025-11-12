import numpy as np
import cvxpy as cp


def add_intercept(X):
    """
    在特征矩阵 X 的第一列添加一个截距项（全为1）
    对应论文中的 x^1 = 1
    这里 X 应该是输入特征feature
    """
    # np.atleast_2d 确保 X 至少是二维的
    X_ = np.atleast_2d(X)
    return np.hstack([np.ones((X_.shape[0], 1)), X_])


class ERM_Newsvendor:
    """
    实现论文中的经验风险最小化 (ERM) 算法。

    NV-ERM1 (无正则化)  和
    NV-ERM2 (带 l1 或 l2 正则化) 。
    """

    def __init__(self, b, h, regularization=None, lambda_reg=0.1):
        """
        初始化模型。

        参数:
        b (float): backorder cost
        h (float): holding cost
        regularization (str): 'l1', 'l2' 或 None
        lambda_reg (float): 正则化系数 lambda
        """
        if b <= 0 or h <= 0:
            raise ValueError("成本 b 和 h 必须为正")
        self.b = b
        self.h = h
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        # q_ 是学到的系数向量
        self.q_ = None

    def fit(self, X_train, d_train):
        """
        根据训练数据 (X_train, d_train) 拟合线性决策规则 q。

        X_train (array): 训练集特征, shape (n, p_basic)
        d_train (array): 训练集需求, shape (n,)
        """

        # 1. 准备数据和变量
        # 添加截距项，X_aug 的维度变为 (n, p)
        X_aug = add_intercept(X_train)
        n, p = X_aug.shape  # n = 观测数, p = 特征数 (含截距)

        # 2. 定义 CVXPY 优化变量
        # q: 决策规则的系数向量, shape (p,)
        q = cp.Variable(p)
        # u: 缺货量, o: 积压量
        u = cp.Variable(n)
        o = cp.Variable(n)

        # 3. 定义线性预测
        # q(x_i) = q^T * x_i
        q_pred = X_aug @ q

        # 4. 定义基础成本 (经验风险)
        # 目标: (1/n) * sum(b*u_i + h*o_i)
        base_cost = (1 / n) * cp.sum(self.b * u + self.h * o)

        # 5. 定义正则化成本
        reg_cost = 0
        if self.regularization is not None:
            # 我们只对特征系数(q[1:])进行正则化，不包括截距项(q[0])
            if self.regularization == 'l1':
                # NV-ERM2 with l1-norm (LASSO)
                reg_cost = self.lambda_reg * cp.norm1(q[1:])
            elif self.regularization == 'l2':
                # NV-ERM2 with l2-norm squared (Ridge)
                reg_cost = self.lambda_reg * cp.sum_squares(q[1:])
            else:
                raise ValueError("正则化必须是 'l1', 'l2' 或 None")

        # 6. 定义目标函数
        objective = cp.Minimize(base_cost + reg_cost)

        # 7. 定义约束
        # 对应 NV-ERM1 中的 s.t.
        constraints = [
            u >= d_train - q_pred,  # u_i >= d_i - q(x_i)
            o >= q_pred - d_train,  # o_i >= q(x_i) - d_i
            u >= 0,
            o >= 0
        ]

        # 8. 求解问题
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != 'optimal':
            print(f"警告: 优化问题未达到最优解，状态: {problem.status}")

        self.q_ = q.value

    def predict(self, X_new):
        """
        使用学到的决策规则 q 预测新特征 X_new 对应的订单量。

        X_new (array): 新的特征, shape (m, p_basic)
        返回 (array): 预测的订单量, shape (m,)
        """
        if self.q_ is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合")

        X_aug = add_intercept(X_new)

        # 预测 q(x_new) = q^T * x_new
        return X_aug @ self.q_