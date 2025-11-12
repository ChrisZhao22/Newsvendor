import numpy as np


class SAA_Newsvendor:
    """
    实现了论文中描述的 benchmark SAA (Sample Average Approximation) 算法。

    该算法无需特征，其决策是历史需求的 r-分位数。
    """

    def __init__(self, b, h):
        """
        初始化模型。

        参数:
        b (float): 单位缺货成本
        h (float): 单位积压成本
        """
        if b <= 0 or h <= 0:
            raise ValueError("成本 b 和 h 必须为正")
        self.b = b
        self.h = h
        self.fractile = b / (b + h)  # 目标分位数

        # q_hat_n 是学到的 SAA 订单量
        self.q_hat_n_ = None

    def fit(self, X_train, d_train):
        """
        根据训练数据 d_train 拟合 SAA 订单量。

        X_train (array): 训练集特征 (将被忽略)
        d_train (array): 训练集需求, shape (n,)
        """

        # SAA 的解是历史需求的 r-分位数
        # 我们使用 numpy.quantile 来高效地计算它
        self.q_hat_n_ = np.quantile(d_train, self.fractile)

        # 注意：X_train 被完全忽略了

    def predict(self, X_new):
        """
        预测新特征 X_new 对应的订单量。

        X_new (array): 新的特征 (将被忽略)
        返回 (array): 预测的订单量, shape (m,)
        """
        if self.q_hat_n_ is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合")

        # SAA 决策不依赖于特征
        # 无论 X_new 是什么，都返回相同的订单量

        # np.atleast_2d 确保 X_new 至少是二维的
        X_new_ = np.atleast_2d(X_new)
        m = X_new_.shape[0]  # 获取新样本的数量

        # 返回一个 shape 为 (m,) 的数组，所有值都是 q_hat_n_
        return np.full(m, self.q_hat_n_)