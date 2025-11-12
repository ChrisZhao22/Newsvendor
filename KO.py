import numpy as np

class KO_Newsvendor:
    """
    实现了论文中的核权重优化 (KO) 算法。

    该算法通过计算加权分位数来找到最优订单量。
    """
    X_train_: None

    def __init__(self, b, h, bandwidth=1.0):
        """
        初始化模型。

        参数:
        b (float): backorder cost
        h (float): holding cost
        bandwidth (float): 高斯核的带宽 w
        """
        if b <= 0 or h <= 0:
            raise ValueError("成本 b 和 h 必须为正")
        self.b = b
        self.h = h
        self.bandwidth = bandwidth
        self.fractile = b / (b + h)  # 目标分位数

        self.X_train_ = None
        self.d_train_ = None

    def fit(self, X_train, d_train):
        """
        KO 算法是一种 "lazy learner"，fit 步骤只是存储数据。

        X_train (array): 训练集特征, shape (n, p)
        d_train (array): 训练集需求, shape (n,)
        """
        # 注意：KO算法不使用截距项
        self.X_train_ = np.atleast_2d(X_train)
        self.d_train_ = d_train

    def _gaussian_kernel_weights(self, x_new):
        """
        计算 x_new 与所有 X_train 样本之间的高斯核权重。
        kappa_i = K_w(x_new - x_i)

        我们使用高斯核 K(u) = exp(-||u||^2 / (2*w^2))
        """
        # 计算 x_new 与 X_train 中每个样本的欧氏距离的平方
        diff = self.X_train_ - x_new
        dist_sq = np.sum(diff ** 2, axis=1)

        # 计算高斯核权重 (kappa)
        weights = np.exp(-dist_sq / (2 * self.bandwidth ** 2))
        return weights

    def _weighted_quantile(self, demands, weights, fractile):
        """
        高效地计算加权分位数，对应 proposition 1 的求解过程 。
        """
        # 1. 根据需求对 需求 和 权重 进行排序(ranking the past demand in increasing order)
        idx_sort = np.argsort(demands)
        d_sorted = demands[idx_sort]
        w_sorted = weights[idx_sort]

        # 2. 计算权重的累积和
        w_cumsum = np.cumsum(w_sorted)
        w_total = w_cumsum[-1]

        # 3. 归一化，得到加权的经验累积分布函数 (ECDF)
        if w_total == 0:
            # 如果所有权重都为0 (例如 x_new 离所有点都极远)
            # 我们退化为标准的 (未加权) 分位数
            return np.quantile(demands, fractile)

        w_cdf = w_cumsum / w_total

        # 4. 找到第一个 ECDF >= 目标分位数的位置
        # np.searchsorted 完美匹配 "inf{q: ...}" (求下确界) 的定义
        q_index = np.searchsorted(w_cdf, fractile, side='left')

        # 边界处理
        if q_index == len(d_sorted):
            q_index = len(d_sorted) - 1

        return d_sorted[q_index]

    def predict(self, X_new):
        """
        预测新特征 X_new 对应的订单量。

        X_new (array): 新的特征, shape (m, p)
        返回 (array): 预测的订单量, shape (m,)
        """
        if self.X_train_ is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合")

        X_new_ = np.atleast_2d(X_new)
        predictions = []

        # 对 X_new 中的每一个样本 x_new 进行预测
        for x_new in X_new_:
            # 1. 计算核权重 kappa_i
            weights = self._gaussian_kernel_weights(x_new)

            # 2. 根据命题1计算加权分位数
            q_pred = self._weighted_quantile(
                self.d_train_,
                weights,
                self.fractile
            )
            predictions.append(q_pred)

        return np.array(predictions)