import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置：定义要对比的 model 及对应的 CSV 文件
# ==========================================

# 请确保这些文件名与您实际运行生成的 CSV 文件名一致
files = {
    'Est-Opt (OLS)': '../data/nv_emerg_estopt_os_3_simple_python.csv',
    'Kernel Opt': '../data/nv_kernelG_de2_3_simple_python.csv',
    'Regularized': '../data/nv_emerg_reg_L1_0.1_simple_python.csv',
    'SAA': '../data/nv_emerg_SAA_lntr_1344_lnte_672_python.csv',
    'Minimax (Scarf)': '../data/nv_emerg_scarf_de_3_simple_python.csv',
    'BinSmoother': '../data/nv_local_poly_J5_python.csv',
    'RKHS': '../data/nv_kernel_quantile_lambda0.01_python.csv',
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
        # 读取 CSV
        df_model = pd.read_csv(filename)

        # 提取成本数据
        # 不同算法的 CSV 中，成本列的名称可能不同，这里做兼容处理
        # 常见列名: 'Cost', 'Realized_Cost', 'Valfac', 'OutOfSample_Cost', 'Cost_bw0.08'

        cost_array = None

        # 1. 尝试直接匹配常见列名
        possible_cols = ['Cost', 'Realized_Cost', 'Valfac', 'OutOfSample_Cost', 'TestSAA']
        for col in possible_cols:
            if col in df_model.columns:
                cost_array = df_model[col].values
                break

        # 2. 如果没找到，尝试模糊匹配 (比如 Kernel Opt 输出的是 Cost_bw0.08)
        if cost_array is None:
            for col in df_model.columns:
                if 'Cost' in col or 'Valfac' in col:
                    cost_array = df_model[col].values
                    break

        if cost_array is None:
            print(f"⚠️ 在 {filename} 中找不到成本数据列 (Available: {df_model.columns.tolist()})")
            continue

        # 我们取前 100 个有效数据进行绘图 (或者取全部)
        valid_len = 100
        if len(cost_array) > valid_len:
            cost_array = cost_array[:valid_len]

        results[model_name] = cost_array
        print(f"✅ 已加载: {model_name} (数据长度: {len(cost_array)})")

    except Exception as e:
        print(f"❌ 读取 {filename} 失败: {e}")

if not results:
    print("没有加载到任何数据，请先运行之前的 5 个算法脚本生成 .csv 文件。")
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

print("\n" + "=" * 60)
print("📊 模型性能对比排行榜 (Cost 越低越好)")
print("=" * 60)
print(summary)
print("=" * 60)

# ==========================================
# 4. 可视化对比 (Matplotlib)
# ==========================================
plt.figure(figsize=(14, 6))

# 图 1: 平均单步成本对比 (柱状图)
plt.subplot(1, 2, 1)
# 生成颜色
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