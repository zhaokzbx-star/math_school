import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（兼容 Windows 和 Mac，避免框框乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 【第四问终极优化版】基于连续物理机制的多目标规划与不确定性量化
# ---------------------------------------------------------

np.random.seed(42) # 锁定随机种子，确保结果可复现

# 1. 蒙特卡洛模拟生成初始决策变量解空间
# 决策变量边界 (基于附表6的物理约束)
N_samples = 15000
x1 = np.random.uniform(0, 5000, N_samples)      # 管网扩建 (0-5000 m)
x2 = np.random.randint(0, 4, N_samples)         # 调蓄池数量 (0-3 座)
x3 = np.random.uniform(0, 30000, N_samples)     # 透水路面面积 (0-30000 m2)
x4 = np.random.randint(0, 2, N_samples)         # 智慧预警系统 (0或1)

# 2. 目标一：计算总工程成本 F1(X) (单位统一为：万元)
cost = 1.8 * x1 + 1200 * x2 + 0.35 * x3 + 80 * x4

# 强约束：总预算必须 <= 5000 万元
valid_idx = cost <= 5000
x1, x2, x3, x4, cost = x1[valid_idx], x2[valid_idx], x3[valid_idx], x4[valid_idx], cost[valid_idx]

# 3. 目标二：综合韧性增益 F2(X) 
# 直接调用第二问(MC-EWM)算出的客观权重，实现逻辑闭环
w_drain, w_store, w_emg = 0.45, 0.35, 0.20 
gain_drain = (x1 / 5000.0) * 0.25 
gain_store = (x2 * 800.0 + x3 * 0.1) / 3500.0 
gain_emg = x4 * 0.40 
resilience_gain = w_drain * gain_drain + w_store * gain_store + w_emg * gain_emg

# 4. 目标三：计算真实退水时间 F3(X) (基于连续物理机理)
def simulate_drain_time(x1_val, x2_val, x3_val, rain_multiplier=1.0, efficiency_decay=1.0):
    # 第一问 L5 路段基础环境参数 (受致灾因子扰动)
    initial_water_vol = 8000.0 * rain_multiplier
    base_drain_rate = 50.0 * efficiency_decay 
    
    # 工程改造带来的真实物理参数改变
    extra_drain_rate = (x1_val / 5000.0) * (0.25 * 50.0) * efficiency_decay
    extra_storage = x2_val * 800.0 + x3_val * 0.1
    
    # 物理推演：剩余积水体积 = 初始降雨洪峰 - 就地削减体积
    current_volume = np.maximum(0.0, initial_water_vol - extra_storage)
    total_drain_rate = base_drain_rate + extra_drain_rate
    
    # 连续时间求解：退水时长(min) = 剩余积水 / 联合排水速率
    return current_volume / total_drain_rate

drain_time = simulate_drain_time(x1, x2, x3)

# 5. 提取帕累托前沿 (Pareto Front)
points = np.column_stack((cost, -resilience_gain, drain_time))
is_efficient = np.ones(points.shape[0], dtype=bool)
for i, c in enumerate(points):
    if is_efficient[i]:
        is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1)  
        is_efficient[i] = True 

pareto_cost = cost[is_efficient]
pareto_gain = resilience_gain[is_efficient]
pareto_time = drain_time[is_efficient]
pareto_X = np.column_stack((x1[is_efficient], x2[is_efficient], x3[is_efficient], x4[is_efficient]))

# 6. K-Means 聚类提取三大典型方案
F_clean = np.column_stack((pareto_cost, pareto_gain, pareto_time))
scaler = StandardScaler()
F_scaled = scaler.fit_transform(F_clean)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(F_scaled)

typical_indices = []
for center in kmeans.cluster_centers_:
    distances = np.linalg.norm(F_scaled - center, axis=1)
    typical_indices.append(np.argmin(distances))

typical_indices = sorted(typical_indices, key=lambda idx: F_clean[idx, 0])
plan_names = ["保守防御型 (低成本)", "均衡拐点型 (高性价比)", "激进海绵型 (最高韧性)"]
colors = ['red', 'orange', 'fuchsia']
markers = ['o', 'D', 's']

# ==========================================
# 打印终端输出结果 (直接用于填入论文表格)
# ==========================================
print("==========================================================")
print("🏆 多目标规划求解完毕：基于物理机制的典型改造方案")
print("==========================================================")
for i, idx in enumerate(typical_indices):
    X_opt = pareto_X[idx]
    normal_t = simulate_drain_time(X_opt[0], X_opt[1], X_opt[2])
    robust_t = simulate_drain_time(X_opt[0], X_opt[1], X_opt[2], rain_multiplier=1.15, efficiency_decay=0.80)
    
    print(f"\n【{plan_names[i]}】")
    print(f"👉 决策变量配置:")
    print(f"   x1 (管网扩容): {X_opt[0]:.1f} m")
    print(f"   x2 (调蓄池):   {int(X_opt[1])} 座")
    print(f"   x3 (透水铺装): {X_opt[2]:.1f} m²")
    print(f"   x4 (预警系统): {'建设 (1套)' if X_opt[3]==1 else '不建 (0套)'}")
    print(f"📊 系统预期目标:")
    print(f"   总改造成本:   {F_clean[idx,0]:.2f} 万元")
    print(f"   韧性综合增益: {F_clean[idx,1]:.4f} (基于客观权重)")
    print(f"🛡️ 鲁棒性测试 (致灾因子+15%, 效能-20%):")
    print(f"   正常退水时间: {normal_t:.1f} min")
    print(f"   极端扰动退水: {robust_t:.1f} min (延迟率: {((robust_t-normal_t)/normal_t)*100:.1f}%)")

# ===============================
# 画图 1: 3D 帕累托前沿
# ===============================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pareto_cost, pareto_gain, pareto_time, 
                c=pareto_gain, cmap='viridis', s=40, alpha=0.7, label='Pareto 解集')

for i, idx in enumerate(typical_indices):
    ax.scatter(F_clean[idx, 0], F_clean[idx, 1], F_clean[idx, 2], 
               color=colors[i], s=250, edgecolors='k', linewidth=1.5, marker='*', label=plan_names[i])

ax.set_xlabel('总改造成本 (万元)')
ax.set_ylabel('韧性综合增益')
ax.set_zlabel('物理退水时间 (min)')
ax.set_title('基于物理演化机制的内涝改造 - 3D帕累托前沿', fontsize=14)
ax.legend(loc='upper left')

plt.colorbar(sc, label='韧性综合增益', pad=0.1)
plt.tight_layout()
plt.savefig("answer4图一_帕累托前沿_物理版.png", dpi=300, bbox_inches='tight')
plt.close()

# ===============================
# 画图 2: 成本-效益 ROI 曲线
# ===============================
fig2 = plt.figure(figsize=(12, 7))
ax2 = fig2.add_subplot(111)

ax2.scatter(pareto_cost, pareto_gain, c=pareto_time, 
            cmap='coolwarm_r', s=50, alpha=0.6, label='可行解 (颜色示退水快慢)')

sorted_indices = np.argsort(pareto_cost)
ax2.plot(pareto_cost[sorted_indices], pareto_gain[sorted_indices], 
         '--', color='gray', alpha=0.5, linewidth=1.5)

for i, idx in enumerate(typical_indices):
    ax2.scatter(F_clean[idx, 0], F_clean[idx, 1], 
                color=colors[i], s=150, edgecolors='k', linewidth=1.5, 
                marker=markers[i], label=plan_names[i])

ax2.set_xlabel('总改造成本 (万元)', fontsize=12)
ax2.set_ylabel('韧性综合增益 (无量纲)', fontsize=12)
ax2.set_title('基于实体成本与物理退水的投资回报率(ROI)分析', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("answer4图二_成本效益曲线_物理版.png", dpi=300, bbox_inches='tight')
plt.close()