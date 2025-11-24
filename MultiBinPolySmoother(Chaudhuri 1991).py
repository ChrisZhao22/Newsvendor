from dataclasses import dataclass
from itertools import combinations_with_replacement,product
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']

# -------------- 基础工具：局部多项式基 --------------
def _multi_indices(d, k):
    """生成所有 |u|<=k 的多重指数 u（含 0 向量）"""
    idx = [(0,)*d]
    for deg in range(1, k+1):
        for comb in combinations_with_replacement(range(d), deg):
            u = [0]*d
            for j in comb:
                u[j] += 1
            idx.append(tuple(u))
    return idx

def _monomials(Z, multi_index):
    """Z: (n,d) 的局部坐标；返回该多重指数下的 z^u 列向量 (n,1)"""
    if sum(multi_index) == 0:
        return np.ones((Z.shape[0], 1))
    col = np.ones((Z.shape[0], 1))
    for j, m in enumerate(multi_index):
        if m > 0:
            col *= Z[:, j:j+1] ** m
    return col

def _design_local_poly(X_bin, center, delta, degree):
    """以块中心为原点并按 delta 缩放：z=(x-center)/delta，构造局部多项式设计矩阵"""
    Z = (X_bin - center) / delta # 在每一个分箱内，进行中心化和尺度化。
    idx_list = _multi_indices(Z.shape[1], degree) # 生成所有 |u|<=degree 的多重指数 u（含 0 向量）
    Phi = np.hstack([_monomials(Z, u) for u in idx_list])  # (n, s(A)) 在列上堆积，将每一个多项式基给拼接起来，形成广义vandermonde矩阵。
    return Phi, idx_list

def _eval_poly(x, center, delta, beta, idx_list):
    """在点 x 上评估局部多项式 P(beta, x)"""
    z = (x - center)/delta  # 中心化和尺度化
    val = 0.0
    for b, u in zip(beta, idx_list): # 遍历所有多项式基，b为beta系数，u为多重指标
        term = 1.0 # term表示每一个多项式基的值
        for j, m in enumerate(u): # 利用多重指标计算多项式基的取值
            if m > 0:
                term *= (z[j] ** m) # 利用多重索引来得到
        val += b * term
    return float(val) # 得到待估计数据点的取值。

# -------------- 主类：估计分位数函数 --------------
@dataclass
class ChaudhuriQuantileBinSmoother:
    """
    Chaudhuri (1991) 式分箱局部多项式分位数回归（仅函数值，不估导数）。

    alpha: 分位数水平 τ
    k:     局部多项式阶（k=0/1/2...）
    p_smooth: 理论中的 p = k + γ（γ∈(0,1)），用于选择块数 J_n
    sup_norm: 若 True，则按 sup 范数速率做 (n/log n) 修正来定 J_n
    c_bins:   选择 J_n 的常数系数，可用于调参
    """
    alpha: float = 0.5 # 分位数水平，将对应b/(b+h)
    k: int = 1 # 局部多项式阶数，将有函数光滑度参数决定。
    p_smooth: float = 1.5 # 函数的光滑度，为分箱的个数提供帮助。
    sup_norm: bool = True # 使用无穷积分函数，对应Chaudhuri (1991)中的定理2
    c_bins: float = 1.0 # 待定

    # 拟合后属性
    centers_: dict = None
    models_: dict = None            # key -> (beta, idx_list)
    delta_: float = None
    aff_shift_: np.ndarray = None   # 映射到 C 的仿射偏移
    aff_scale_: np.ndarray = None   # 映射到 C 的仿射尺度
    d_: int = None
    J_: int = None
    global_fallback_: float = 0.0

    # ---------- 内部工具 ----------
    def _affine_fit_C(self, X):
        """把原始 X 仿射映射到 C=[-0.5,0.5]^d"""
        mn, mx = X.min(0), X.max(0)
        scale = (mx - mn)
        scale[scale == 0] = 1.0
        Z = (X - mn) / scale - 0.5
        return Z, mn, scale

    def _choose_J(self, n, d):
        denom = (2 * self.p_smooth + d)
        print(denom)
        print((n / np.log(max(n, 3)))** (1.0 / denom))
        print((n)** (1.0 / denom))


        if self.sup_norm:
            J = int(max(1, np.floor(self.c_bins * (n / np.log(max(n, 3))) ** (1.0 / denom))))
        else:
            J = int(max(1, np.floor(self.c_bins * n ** (1.0 / denom))))
        return J

    def _which_blocks_contain(self, x_c, tol=1e-12):
        """返回在闭包意义上包含 x_c 的所有方块键（用于边界平均）"""
        J = self.J_
        hits_per_dim = []
        for j in range(self.d_):
            pos = (x_c[j] + 0.5) / self.delta_ # 这里默认x_c是已经中心化和尺度化之后的了。
            k = int(np.floor(pos + tol)) # 向下取整，得到浮点类型的数。
            cand = set([int(np.clip(k, 0, J-1))])
            # 贴近分界时，左右两侧都算（闭包包含）
            if abs(pos - np.floor(pos)) < tol and 0 < k <= J-1:
                cand.add(k-1)
            hits_per_dim.append(sorted(cand)) # hits_per_dim是一个列表，每一个元素是x_c每一个维度上所对应的箱bin。
        # 笛卡尔积
        keys = [tuple(t) for t in product(*hits_per_dim)]
        return keys

    # ---------- 训练 ----------
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).ravel()
        n, d = X.shape
        self.d_ = d

        # 1) 仿射映射到 C=[-0.5,0.5]^d
        Xc, shift, scale = self._affine_fit_C(X)
        self.aff_shift_, self.aff_scale_ = shift, scale

        # 2) 选块数 J 与边长 delta
        J = self._choose_J(n, d)
        self.J_ = J
        delta = 1.0 / J
        self.delta_ = delta

        # 3) 每维网格中心
        grid_1d = [np.linspace(-0.5 + delta/2, 0.5 - delta/2, J) for _ in range(d)]

        # 4) 为每个样本确定所属方块键
        idxs = np.floor((Xc + 0.5) / delta).astype(int)
        idxs = np.clip(idxs, 0, J-1)
        bin_keys = [tuple(ind) for ind in idxs]

        # 5) 收集每个“出现过”的方块的数据索引
        bin_to_indices = defaultdict(list)
        for i, key in enumerate(bin_keys):
            bin_to_indices[key].append(i)

        self.centers_ = {}
        self.models_  = {}
        idx_list_global = _multi_indices(d, self.k)

        # 6) 逐方块做 LP 分位数回归
        intercepts = []
        for key, idx_list in bin_to_indices.items(): #这里key是对应的bin的索引（是一个数组，每个元素对应样本在这个维度上属于第几个bin
            # ），idx_list这个列表表示是这个bin块中样本点索引，用来识别是是哪些样本落在了对应的这个bin上。
            X_bin_c = Xc[idx_list]
            y_bin = y[idx_list]
            center = np.array([grid_1d[j][key[j]] for j in range(d)])

            Phi, idx_list_loc = _design_local_poly(X_bin_c, center, delta, self.k) # 得到这个方块上的广义vandermonde设计矩阵。
            m_feats = Phi.shape[1] # 特征数目，对应广义vandermonde矩阵的列数s(A)

            # 样本不足时退化到 k=0（常数），但这种情况一般不会出现，因为theorem3.1保证了每个小方块上有解。
            if len(y_bin) < m_feats:
                Phi, idx_list_loc = _design_local_poly(X_bin_c, center, delta, 0)
                m_feats = Phi.shape[1]

            # LP: min τ*1^T u + (1-τ)*1^T v, s.t. y - Phi b = u - v, u,v>=0
            n_b = len(y_bin)
            c = np.hstack([np.zeros(m_feats),
                           self.alpha*np.ones(n_b),
                           (1-self.alpha)*np.ones(n_b)])  # 标准LP问题中，目标函数中的c向量，与决策变量的内积构成cost，在这里就是pinball lost。
            Aeq = np.hstack([-Phi, np.eye(n_b), -np.eye(n_b)]) # 标准LP问题中等式约束的LHS的A矩阵
            beq = y_bin # 标准LP问题中等式约束的RHS
            bounds = [(-np.inf, np.inf)]*m_feats + [(0, None)]*n_b + [(0, None)]*n_b #标准LP问题中对决策变量的取值的约束。

            res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
            if not res.success:
                # 失败则用块内 α-分位数常数模型兜底
                beta = np.zeros(1); beta[0] = np.quantile(y_bin, self.alpha)
                idx_list_loc = [(0,)*d]
            else:
                beta = res.x[:m_feats]

            # 记录模型
            self.centers_[key] = center
            self.models_[key]  = (beta, idx_list_loc)

            # 收集截距作为全局回退的参考
            try:
                zero_idx = idx_list_loc.index(tuple([0]*d))
                intercepts.append(beta[zero_idx])
            except Exception:
                intercepts.append(beta[0])

        # 全局回退：用各块截距的中位数
        self.global_fallback_ = float(np.median(intercepts)) if len(intercepts) else 0.0
        return self

    # ---------- 预测（仅函数值） ----------
    def predict(self, X):
        """返回 q_alpha(x) 的估计值"""
        if self.models_ is None:
            raise RuntimeError("Must call fit() before predict().")
        X = np.asarray(X)
        # 映射到 C
        Xc = (X - self.aff_shift_) / self.aff_scale_ - 0.5 # 中心化和尺度化
        yhat = np.zeros(len(X))
        for i, xc in enumerate(Xc):
            keys = self._which_blocks_contain(xc)
            vals = []
            for key in keys:
                if key in self.models_:
                    beta, idx_list_loc = self.models_[key]
                    center = self.centers_[key]
                    vals.append(_eval_poly(xc, center, self.delta_, beta, idx_list_loc))
            # 若预测点位于训练期未出现的块，使用全局回退，有理论保障，这个情况不会出现。
            yhat[i] = np.mean(vals) if len(vals) else self.global_fallback_
        return yhat

# --- 数值模拟与可视化实验 ---

# 1. 合成数据
print("正在生成二维合成数据...")
np.random.seed(2)
n = 10000
X = np.random.uniform(-2, 2, size=(n, 2))
f = lambda x1, x2: np.sin(1.2 * x1) + 0.5 * np.cos(1.0 * x2)
sig = lambda x1, x2: 0.5 + 0.25 * np.sqrt((x1 ** 2 + x2 ** 2) / 5 + 0.1)
y = f(X[:, 0], X[:, 1]) + sig(X[:, 0], X[:, 1]) * np.random.randn(n)

# 2. 拟合多个分位数模型
quantiles_to_fit = [0.25, 0.50, 0.75]
models = {}
K_DEGREE = 1     # 局部多项式阶数 k
P_SMOOTH = K_DEGREE + 0.5 # 假设函数光滑度 p
C_BINS_FACTOR = 1.0 # J_n 选择公式中的常数因子 c

print(f"\n--- 模型拟合 (k={K_DEGREE}, p={P_SMOOTH}, 自适应 J) ---")
for p in quantiles_to_fit:
    print(f"正在拟合 p={p} 模型...")
    # 使用无穷范数对应的 J_n 选择 (sup_norm=True)
    # J 将在 fit 方法内部通过 _choose_J 计算
    model = ChaudhuriQuantileBinSmoother(alpha=p, k=K_DEGREE, p_smooth=P_SMOOTH, c_bins=C_BINS_FACTOR, sup_norm=True)
    model.fit(X, y)
    models[p] = model
    print(f"  完成. 实际计算每维的分箱数 J = {model.J_}")

# 3. 准备可视化数据
print("\n--- 准备可视化数据 ---")
grid_size = 30 # 我们的网格在每个维度上有多少个点。
x1_lin = np.linspace(X[:,0].min(), X[:,0].max(), grid_size)
x2_lin = np.linspace(X[:,1].min(), X[:,1].max(), grid_size)
xx, yy = np.meshgrid(x1_lin, x2_lin)
grid_X = np.c_[xx.ravel(), yy.ravel()]

grid_Z = {}
for p, model in models.items(): #
    print(f"正在预测 p={p} 曲面...")
    grid_Z[p] = model.predict(grid_X).reshape(xx.shape)

# （所有模型应该计算出相同的 J，因为 n, d, p_smooth 等都相同）
representative_J = models[quantiles_to_fit[0]].J_

# 有点疑问：散点图画的是原始数据点，但是拟合的时候使用的xx和yy是其最大值和最小值的中间差值。这合理么？

# 4. 可视化 1: 3D 曲面图
print("--- 绘制 3D 曲面图 ---")
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d') # 明确指出使用3D绘图
ax.scatter(X[:, 0], X[:, 1], y, c='gray', alpha=0.2, label='原始数据 (y)', s=10)
colors = {0.25: 'blue', 0.50: 'green', 0.75: 'red'} # 字典
labels = {0.25: 'p=0.25 (第25分位数)', 0.50: 'p=0.50 (中位数)', 0.75: 'p=0.75 (第75分位数)'} # 字典
proxies = []
for p, Z_surf in grid_Z.items(): # 遍历字典中的键值对。
    ax.plot_surface(xx, yy, Z_surf, color=colors[p], alpha=0.6, rstride=1, cstride=1, linewidth=0.1, edgecolors='k')
    proxies.append(plt.Rectangle((0, 0), 1, 1, fc=colors[p], alpha=0.6))
ax.set_xlabel('特征 X1')
ax.set_ylabel('特征 X2')
ax.set_zlabel('目标 y')
ax.set_title(f'多元分箱分位数回归 (k={K_DEGREE}, 自适应 J≈{representative_J})')
ax.legend(proxies, [labels[p] for p in quantiles_to_fit])
ax.view_init(elev=20., azim=-65)
plt.savefig("multivariate_bin_smoother_3D_plot_enhanced_adaptiveJ.png")
print("3D 图表已保存到 'multivariate_bin_smoother_3D_plot_enhanced_adaptiveJ.png'")
# plt.show()

# 5. 可视化 2: 2D 横截面图
print("--- 绘制 2D 横截面图 ---")
x2_slice_val = np.median(X[:, 1])
slice_width = 0.2
slice_mask = np.abs(X[:, 1] - x2_slice_val) < slice_width
X_slice = X[slice_mask]
y_slice = y[slice_mask]
if len(X_slice) > 0:
    x1_curve = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    X_curve = np.column_stack((x1_curve, np.full_like(x1_curve, x2_slice_val)))
    y_curves = {}
    for p, model in models.items():
        y_curves[p] = model.predict(X_curve)
    plt.figure(figsize=(12, 7))
    plt.scatter(X_slice[:, 0], y_slice, alpha=0.3, label=f'数据 (X2 ≈ {x2_slice_val:.2f} ± {slice_width})', s=15, c='gray')
    for p, y_curve in y_curves.items():
        plt.plot(x1_curve, y_curve, color=colors[p], linewidth=2.5, label=labels[p])
    plt.xlabel('特征 X1')
    plt.ylabel('目标 y')
    plt.title(f'多元分箱分位数回归 - 横截面 (X2 ≈ {x2_slice_val:.2f}, J≈{representative_J})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(y_slice.min() - 1, y_slice.max() + 1)
    plt.savefig("multivariate_bin_smoother_2D_slice_plot_enhanced_adaptiveJ.png")
    print("2D 横截面图表已保存到 'multivariate_bin_smoother_2D_slice_plot_enhanced_adaptiveJ.png'")
    # plt.show()
else:
    print("警告：在选定的 X2 切片范围内没有足够的数据点，无法生成 2D 横截面图。")

# --- 示例预测 ---
x_new = np.array([[0.0, 0.0], [1.0, -1.0]])
print("\n--- 示例预测 ---")
for p, model in models.items():
    try:
        q_hat = model.predict(x_new)
        print(f"p={p}: 预测值 for {x_new.tolist()} => {q_hat}")
    except Exception as e:
        print(f"p={p}: 预测时出错: {e}")

print("\n模拟实验完成。")