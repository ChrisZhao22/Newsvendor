import numpy as np
import itertools
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import platform # 导入 platform 库

# --- 处理 Matplotlib 中文显示问题 ---
# 根据操作系统选择字体
system_name = platform.system()
if system_name == "Windows":
    # Windows 系统: 尝试使用 'SimHei' (黑体) 或 'Microsoft YaHei' (微软雅黑)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif system_name == "Darwin":
    # macOS 系统: 尝试使用 'Arial Unicode MS' 或 'PingFang SC'
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
elif system_name == "Linux":
     # Linux 系统: 尝试使用 'WenQuanYi Micro Hei' 或 'Noto Sans CJK JP'
     # 注意：Linux 下字体名称可能需要根据您的安装情况调整
     plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP']
else:
    # 其他系统，可以添加更多判断或使用通用后备字体
    print("未知的操作系统，尝试通用字体设置。如果中文显示仍有问题，请手动指定您系统上安装的中文字体。")

# 解决负号显示问题
# 当使用非 Latin 字体时，需要设置此项以正确显示负号
plt.rcParams['axes.unicode_minus'] = False
# --- 中文显示设置结束 ---

def check_function(u, p):
    """
    检查函数 (Check Function) rho_p(u)，与 (Yu and Jones, 1998) 一致。
    """
    return np.where(u >= 0, p * u, (p - 1) * u)


class MultivariateBinSmoother:
    """
    实现了 Chaudhuri (1991) 论文中的“分箱平滑器”方法。

    该算法执行多元局部多项式分位数回归。

    1. 将 d 维特征空间划分为 (n_bins_per_dim)^d 个箱子。
    2. 在每个箱子内部，使用该箱子内的数据点拟合一个 k 阶多项式。
    3. 拟合目标是最小化该箱子内的检查函数 (check function) 损失。
    """

    def __init__(self, p=0.5, k=1, n_bins_per_dim=10):
        """
        初始化

        Args:
            p (float): 目标分位数 (0 < p < 1)。
            k (int): 局部多项式的次数。k=0 (局部常数), k=1 (局部线性), k=2 (局部二次) 等。
            n_bins_per_dim (int): 每个特征维度要切分的箱子数量。
                                  这对应 Chaudhuri 论文中的 J_n。
                                  总箱子数为 (n_bins_per_dim)^d。
        """
        self.p = p
        self.k = k
        self.n_bins_per_dim = n_bins_per_dim

        self.models_ = {}  # 存储每个箱子的 (PolynomialFeatures, beta)
        self.bin_edges_ = None  # 存储箱子的边界
        self.fallback_value_ = None  # 当箱子为空时使用的全局回退值
        self.n_features_in_ = None

    def _get_bin_tuple(self, X_row):
        """辅助函数：获取单个数据点 X_row 所在的箱子索引元组"""
        indices = np.zeros(self.n_features_in_, dtype=int)
        for d in range(self.n_features_in_):
            # np.digitize 找到 X_row[d] 属于哪个 bin
            idx = np.digitize(X_row[d], self.bin_edges_[d]) - 1
            # 确保索引在 [0, n_bins_per_dim - 1] 范围内
            indices[d] = np.clip(idx, 0, self.n_bins_per_dim - 1)
        return tuple(indices)

    def fit(self, X, y):
        """
        拟合多元分箱平滑器模型。
        """
        self.n_features_in_ = X.shape[1]

        # 1. 确定并存储箱子边界
        self.bin_edges_ = []
        for d in range(self.n_features_in_):
            min_val, max_val = np.min(X[:, d]), np.max(X[:, d])
            # 略微扩大边界以包含所有数据
            edges = np.linspace(min_val - 1e-9, max_val + 1e-9, self.n_bins_per_dim + 1)
            self.bin_edges_.append(edges)

        # 2. 将所有数据点分配到箱子中
        # bin_assignments 是一个 (n_samples,) 的元组列表，例如 [(0,1), (2,3), ...]
        bin_assignments = [self._get_bin_tuple(X[i]) for i in range(len(X))]

        # 3. 拟合每个箱子的模型
        # 生成所有可能的箱子索引元组
        all_bin_tuples = itertools.product(range(self.n_bins_per_dim), repeat=self.n_features_in_)

        for bin_tuple in all_bin_tuples:
            # 找到落入这个箱子的所有数据点的索引
            mask = [assign == bin_tuple for assign in bin_assignments]
            X_bin = X[mask]
            y_bin = y[mask]

            # 如果箱子为空，存储 None
            if len(y_bin) == 0:
                self.models_[bin_tuple] = None
                continue

            # --- 在箱子内部拟合多项式 ---
            # 对应 Chaudhuri (1991) 论文中的公式

            # a. 创建多项式特征
            poly = PolynomialFeatures(degree=self.k, include_bias=True)
            X_bin_poly = poly.fit_transform(X_bin)
            n_coeffs = X_bin_poly.shape[1]

            # b. 如果数据点太少，无法拟合 k 阶多项式，则降级为 k=0 (局部常数)
            current_k = self.k
            if len(y_bin) < n_coeffs:
                current_k = 0
                poly = PolynomialFeatures(degree=current_k, include_bias=True)
                X_bin_poly = poly.fit_transform(X_bin)
                n_coeffs = X_bin_poly.shape[1]

            # c. 定义目标函数：最小化箱子内的检查损失
            def objective_function(beta):
                residuals = y_bin - X_bin_poly @ beta
                loss = check_function(residuals, self.p)
                return np.sum(loss)

            # d. 使用一个好的初始值（截距=局部样本分位数）
            initial_beta = np.zeros(n_coeffs)
            initial_beta[0] = np.quantile(y_bin, self.p)

            # e. 优化
            result = minimize(objective_function, initial_beta, method='Nelder-Mead')

            # f. 存储拟合好的模型（特征转换器 + 系数）
            self.models_[bin_tuple] = (poly, result.x)

        # 4. 计算一个全局的回退值，用于预测时遇到空箱子
        self.fallback_value_ = np.quantile(y, self.p)
        return self

    def predict(self, X):
        """
        使用拟合好的局部多项式模型进行预测。
        """
        if self.bin_edges_ is None:
            raise RuntimeError("Must call fit() before predict().")

        y_pred = np.zeros(len(X))

        for i in range(len(X)):
            # 1. 找到该点所属的箱子
            bin_tuple = self._get_bin_tuple(X[i])

            # 2. 检索该箱子的模型
            model = self.models_.get(bin_tuple)

            # 3. 预测
            if model is not None:
                # 如果该箱子有模型，则使用它
                poly_features, beta = model
                X_i_poly = poly_features.transform(X[i].reshape(1, -1))
                y_pred[i] = X_i_poly @ beta
            else:
                # 如果该箱子在训练时为空，则使用全局回退值
                y_pred[i] = self.fallback_value_

        return y_pred

# --- 演示 ---
print("正在生成二维合成数据...")
np.random.seed(42)
N_POINTS = 500
# X 是 d=2 维特征
X_data = np.random.rand(N_POINTS, 2) * 10
# X1 = X_data[:, 0], X2 = X_data[:, 1]

# y 是 X1, X2 的非线性函数，且噪声是异方差的
y_true_mean = 5 + (X_data[:, 0] * np.sin(X_data[:, 0] / 2)) + (X_data[:, 1] * 0.5)
noise = (0.5 + X_data[:, 0]/5 + X_data[:, 1]/5) * np.random.randn(N_POINTS)
y_data = y_true_mean + noise

# --- 拟合模型 ---
# 我们将拟合 k=1 (局部线性) 和 n_bins_per_dim=8 (总共 8x8=64 个箱子)
N_BINS = 8
K_DEGREE = 1 # 局部线性

print(f"正在拟合 p=0.75 模型 (k={K_DEGREE}, {N_BINS}x{N_BINS} bins)...")
model_75 = MultivariateBinSmoother(p=0.75, k=K_DEGREE, n_bins_per_dim=N_BINS)
model_75.fit(X_data, y_data)

print(f"正在拟合 p=0.25 模型 (k={K_DEGREE}, {N_BINS}x{N_BINS} bins)...")
model_25 = MultivariateBinSmoother(p=0.25, k=K_DEGREE, n_bins_per_dim=N_BINS)
model_25.fit(X_data, y_data)

print("拟合完成，正在生成 3D 可视化...")

# --- 可视化 ---
# 创建一个用于绘制曲面的网格
grid_size = 20
xx, yy = np.meshgrid(
    np.linspace(0, 10, grid_size),
    np.linspace(0, 10, grid_size)
)
grid_X = np.c_[xx.ravel(), yy.ravel()]

# 预测网格上两个分位数曲面的 Z 值
grid_z_75 = model_75.predict(grid_X).reshape(xx.shape)
grid_z_25 = model_25.predict(grid_X).reshape(xx.shape)

# 绘制 3D 图
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据散点
ax.scatter(X_data[:, 0], X_data[:, 1], y_data, c='gray', alpha=0.3, label='原始数据 (y)')

# 绘制 p=0.75 曲面 (红色)
ax.plot_surface(xx, yy, grid_z_75, color='red', alpha=0.5, label='p=0.75 (第75分位数)')
# 绘制 p=0.25 曲面 (蓝色)
ax.plot_surface(xx, yy, grid_z_25, color='blue', alpha=0.5, label='p=0.25 (第25分位数)')

ax.set_xlabel('特征 X1')
ax.set_ylabel('特征 X2')
ax.set_zlabel('目标 y')
ax.set_title(f'多元分箱分位数回归 (k={K_DEGREE}, {N_BINS}x{N_BINS} bins)')

# 解决 3D 图例问题
red_proxy = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.5)
blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
ax.legend([red_proxy, blue_proxy], ['p=0.75 曲面', 'p=0.25 曲面'])

plt.savefig("multivariate_bin_smoother_plot.png")
print("3D 图表已保存到 'multivariate_bin_smoother_plot.png'")