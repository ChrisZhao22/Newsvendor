import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置：定义要对比的model
# ==========================================

files = {
    'Est-Opt (OLS)': 'nv_emerg_estopt_os_3_lntr_1344_lnva_672_python.mat',
    'Kernel Opt': 'nv_kernelG_de2_3_lntr_1344_python.mat',  # 注意检查文件名
    'Regularized': 'nv_emerg_reg_L1_0.1_python.mat',
    'SAA': 'nv_emerg_SAA_lntr_1344_lnte_672_python.mat',
    'Minimax (Scarf)': 'nv_emerg_scarf_de_3_lntr_1344_lnva_672_p_120_python.mat'  # 取其中一个p值
}

results = {}

print("正在读取结果文件...")

# ==========================================
# 2. 读取数据与提取指标
# ==========================================
for model_name, filename in files.items():
    if not os.path.exists(filename):
        print(f"⚠️ 警告: 找不到文件 {filename}，跳过该模型。")
        continue

    try:
        data = scipy.io.loadmat(filename)

        # 提取成本数据
        # 不同的脚本用了不同的变量名存储成本
        # SAA用的是 'TestSAA', 其他的基本是 'Valfac'
        if 'Valfac' in data:
            cost_array = data['Valfac'].flatten()
        elif 'TestSAA' in data:
            cost_array = data['TestSAA'].flatten()
        else:
            print(f"⚠️ 在 {filename} 中找不到成本数据变量 (Valfac/TestSAA)")
            continue

        # 我们取前 100 个有效数据
        valid_len = 100
        if len(cost_array) > valid_len:
            cost_array = cost_array[:valid_len]

        results[model_name] = cost_array
        print(f"✅ 已加载: {model_name} (数据长度: {len(cost_array)})")

    except Exception as e:
        print(f"❌ 读取 {filename} 失败: {e}")

if not results:
    print("没有加载到任何数据，请先运行之前的 5 个模型生成 .mat 文件。")
    exit()

# ==========================================
# 3. 数据分析 (Pandas DataFrame)
# ==========================================
# 将字典转为 DataFrame 方便计算
df = pd.DataFrame(results)

# 计算统计指标
summary = pd.DataFrame({
    'Total Cost': df.sum(),
    'Mean Cost': df.mean(),
    'Std Dev': df.std(),
    'Min Cost': df.min(),
    'Max Cost': df.max()
})

# 按平均成本排序 (越低越好)
summary = summary.sort_values(by='Mean Cost')

print("\n" + "=" * 40)
print("📊 模型性能对比排行榜 (Cost 越低越好)")
print("=" * 40)
print(summary)
print("=" * 40)

# ==========================================
# 4. 可视化对比 (Matplotlib)
# ==========================================
plt.figure(figsize=(14, 6))

# 图 1: 平均单步成本对比 (柱状图)
plt.subplot(1, 2, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(summary)))
bars = plt.bar(summary.index, summary['Mean Cost'], color=colors)
plt.title('Average Cost per Period (Lower is Better)')
plt.ylabel('Cost')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)

# 在柱子上标数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# 图 2: 累计成本增长曲线 (折线图)
plt.subplot(1, 2, 2)
for model_name in df.columns:
    # 计算累计和
    cumsum = df[model_name].cumsum()
    plt.plot(cumsum, label=model_name, linewidth=2)

plt.title('Cumulative Cost Over Time')
plt.xlabel('Time Step')
plt.ylabel('Total Accumulated Cost')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()