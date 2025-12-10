import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 数据加载
# ==========================================
data_file = '../data/newsvendor_simple_data.csv'
config_file = '../data/data_config.json'

# 读取 CSV
df = pd.read_csv(data_file)
Demand = df['Demand'].values

# 读取配置
with open(config_file, 'r') as f:
    config = json.load(f)

lntr = config['lntr']
lnva = config['lnva']
lnte = config['lnte']
TOTAL_LEN = len(Demand)

print(f"已加载数据: {data_file} (Total: {TOTAL_LEN})")

# ==========================================
# 2. 参数设置
# ==========================================
b = 3 / 4
h = 1 / 4
r = b / (b + h)

# 结果存储
Q_pred = np.zeros(lnte)
CostSAA = np.zeros(lnte)
TestSAA = np.zeros(lnte)


# ==========================================
# 3. 辅助函数
# ==========================================
def nv_cost(q, d, b, h):
    return np.maximum(d - q, 0) * b + np.maximum(q - d, 0) * h


# ==========================================
# 4. 主循环逻辑 (SAA)
# ==========================================
print(f"Start SAA Loop (Target Quantile: {r:.4f})")
start_time = time.time()

start_idx = lntr + lnva  # test beginning index

for k in range(lnte):
    t = start_idx + k

    # 获取训练数据 (滚动窗口)
    demand_train = Demand[t - lntr: t]

    # SAA 求解
    q0 = np.quantile(demand_train, r)

    # 计算样本内成本
    in_sample_costs = nv_cost(q0, demand_train, b, h)
    avg_in_sample_cost = np.mean(in_sample_costs)

    # 记录
    Q_pred[k] = q0
    CostSAA[k] = avg_in_sample_cost

    # 样本外测试
    if t < TOTAL_LEN:
        actual_demand = Demand[t]
        TestSAA[k] = nv_cost(q0, actual_demand, b, h)

end_time = time.time()
print(f"Loop finished in {end_time - start_time:.4f} seconds")

# ==========================================
# 5. 保存结果
# ==========================================
output_filename = f'../data/nv_SAA.csv'

# 构建 DataFrame
df_out = pd.DataFrame({
    'Q_Decision': Q_pred,
    'Demand_D': Demand[start_idx:start_idx + lnte],
    'InSample_Cost': CostSAA,
    'OutOfSample_Cost': TestSAA
})

df_out.to_csv(output_filename, index=False)
print(f"Results saved to {output_filename}")

# ==========================================
# 6. AI可视化：美化版 (Professional Style)
# ==========================================

# 设置风格 (如果没有安装 seaborn，可以注释掉这两行，单纯用 matplotlib 也可以)
sns.set_theme(style="whitegrid", context="talk")

# 创建画布，包含两个子图 (上下排列)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1]})

# 定义时间轴和数据片段
t = np.arange(lnte)
y_demand = Demand[start_idx:start_idx + lnte]
y_pred = Q_pred

# ====================
# 子图 1: 全局趋势 (Global View)
# ====================
# 1. 画线
ax1.plot(t, y_demand, label='Actual Demand', color='#2c3e50', linewidth=1.5, alpha=0.8)  # 深蓝灰色
ax1.plot(t, y_pred, label='Optimal Order (Q)', color='#e74c3c', linewidth=2.0, linestyle='-')  # 亮红色

# 2. 填充成本区域 (高亮差异)
# 当 订货 > 需求 (库存积压/Holding Cost) -> 用浅黄色/绿色填充
ax1.fill_between(t, y_demand, y_pred, where=(y_pred >= y_demand),
                 interpolate=True, color='#27ae60', alpha=0.15, label='Inventory (Overage)')
# 当 订货 < 需求 (缺货损失/Stockout Cost) -> 用浅红色填充
ax1.fill_between(t, y_demand, y_pred, where=(y_pred < y_demand),
                 interpolate=True, color='#c0392b', alpha=0.15, label='Stockout (Underage)')

# 3. 装饰
ax1.set_title('NV-SAA: Newsvendor Decision Analysis: Global Overview', fontsize=18, fontweight='bold', pad=15)
ax1.set_ylabel('Quantity', fontsize=14)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax1.set_xlim(0, lnte)
ax1.margins(x=0)

# ====================
# 子图 2: 细节放大 (Zoomed View)
# ====================
zoom_len = 300
t_zoom = t[:zoom_len]
y_demand_zoom = y_demand[:zoom_len]
y_pred_zoom = y_pred[:zoom_len]

# 1. 画线 (带数据点 Marker，方便看具体点的位置)
ax1.plot(t_zoom, y_demand_zoom, color='#2c3e50', alpha=0, linewidth=0)  # 仅用于统一颜色逻辑，实际画在下面
ax2.plot(t_zoom, y_demand_zoom, label='Actual Demand', color='#2c3e50',
         linewidth=1.8, linestyle='-', marker='.', markersize=4, alpha=0.7)
ax2.plot(t_zoom, y_pred_zoom, label='Optimal Order (Q)', color='#e74c3c',
         linewidth=2.5, linestyle='-', alpha=0.9)  # 预测线通常平滑，不加marker防止太乱

# 2. 填充区域 (同样高亮成本)
ax2.fill_between(t_zoom, y_demand_zoom, y_pred_zoom, where=(y_pred_zoom >= y_demand_zoom),
                 interpolate=True, color='#27ae60', alpha=0.2)
ax2.fill_between(t_zoom, y_demand_zoom, y_pred_zoom, where=(y_pred_zoom < y_demand_zoom),
                 interpolate=True, color='#c0392b', alpha=0.2)

# 3. 装饰
ax2.set_title(f'NV-SAA: Zoomed Detail (First {zoom_len} Samples)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Time Step / Index', fontsize=14)
ax2.set_ylabel('Quantity', fontsize=14)
ax2.set_xlim(0, zoom_len)
ax2.grid(True, linestyle='--', alpha=0.6)  # 网格线虚线化

# 调整整体布局防止重叠
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)  # 调整两个图之间的间距

plt.show()
