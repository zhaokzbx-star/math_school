import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 定义多目标优化问题 (MO-MIP)
# ==========================================
class UrbanResilienceProblem(Problem):
    def __init__(self):
        # 定义4个变量的上下界
        # x1:管网扩容[0, 5000], x2:调蓄池[0, 3], x3:透水路面[0, 30000], x4:预警系统[0, 1]
        xl = np.array([0, 0, 0, 0])
        xu = np.array([5000, 3, 30000, 1])
        # n_var=变量数, n_obj=目标数, n_ieq_constr=不等式约束数
        super().__init__(n_var=4, n_obj=3, n_ieq_constr=4, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # --- 变量解码与类型约束 ---
        x1 = x[:, 0]               # 连续变量
        x2 = np.round(x[:, 1])     # 整数变量（离散化处理）
        x3 = x[:, 2]               # 连续变量
        x4 = np.round(x[:, 3])     # 0-1变量（离散化处理）

        # --- 目标函数计算 ---
        # 目标1：最小化成本 (万元)
        f1 = 1.8 * x1 + 1200 * x2 + 0.35 * x3 + 80 * x4
        
        # 目标2：最大化韧性增益 (pymoo默认求最小，故加负号)
        # 假设 AHP/熵权法得出的权重为 w1=0.4, w2=0.4, w3=0.2
        w1, w2, w3 = 0.4, 0.4, 0.2
        resilience_gain = w1 * ((0.25*x1 + 0.15*x3)/32) + \
                          w2 * ((800*x2 + 0.1*x3)/2300) + \
                          w3 * ((4.8*x4)/7)
        f2 = -resilience_gain 

        # 目标3：最大化退水缩短时间 (加负号转化为求最小)
        time_reduction = 0.15 * x1 + 0.2 * x2 + 0.1 * x3
        f3 = -time_reduction

        out["F"] = np.column_stack([f1, f2, f3])

        # --- 约束条件计算 (形式需满足 g(x) <= 0) ---
        # 约束1：总预算不超过 5000 万元
        g1 = f1 - 5000
        # 约束2：排水达标率提升上限 <= 32
        g2 = (0.25 * x1 + 0.15 * x3) - 32
        # 约束3：调蓄容积提升上限 <= 2300
        g3 = (800 * x2 + 0.1 * x3) - 2300
        # 约束4：退水时间缩短上限 <= 25
        g4 = (0.15 * x1 + 0.2 * x2 + 0.1 * x3) - 25

        out["G"] = np.column_stack([g1, g2, g3, g4])

# ==========================================
# 2. 算法配置与求解
# ==========================================
problem = UrbanResilienceProblem()
# 初始化 NSGA-II 算法，种群规模 100
algorithm = NSGA2(pop_size=100)

print("🚀 正在运行 NSGA-II 算法求解多目标帕累托前沿...")
res = minimize(problem,
               algorithm,
               ('n_gen', 200), # 迭代 200 代
               seed=42,
               verbose=False)

# 提取非支配解集 (Pareto Front)
# 注意：把 f2 和 f3 的负号转回来，恢复真实业务含义
pareto_F = np.copy(res.F)
pareto_F[:, 1] = -pareto_F[:, 1] # 真实韧性增益
pareto_F[:, 2] = -pareto_F[:, 2] # 真实退水缩短时间
pareto_X = np.copy(res.X)
pareto_X[:, 1] = np.round(pareto_X[:, 1]) # 确保输出为整数
pareto_X[:, 3] = np.round(pareto_X[:, 3]) # 确保输出为整数

# ==========================================
# 3. 基于 K-Means 的典型方案聚类提取
# ==========================================
# 将目标空间标准化以消除量纲影响，然后聚为3类
scaler = StandardScaler()
F_scaled = scaler.fit_transform(pareto_F)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(F_scaled)

typical_indices = []
for center in kmeans.cluster_centers_:
    distances = np.linalg.norm(F_scaled - center, axis=1)
    typical_indices.append(np.argmin(distances))

# 根据成本排序，依次命名为保守型、均衡型、激进型
typical_indices = sorted(typical_indices, key=lambda idx: pareto_F[idx, 0])
plan_names = ["保守防御型 (低成本)", "均衡拐点型 (高性价比)", "激进海绵型 (最高韧性)"]

print("\n🏆 --- 推荐的三大典型综合改造方案 ---")
for i, idx in enumerate(typical_indices):
    x_opt = pareto_X[idx]
    f_opt = pareto_F[idx]
    print(f"\n【{plan_names[i]}】")
    print(f"👉 决策变量:")
    print(f"   管网扩容长度: {x_opt[0]:.1f} m")
    print(f"   新建调蓄池数: {int(x_opt[1])} 座")
    print(f"   透水路面面积: {x_opt[2]:.1f} m²")
    print(f"   智慧预警系统: {'建设' if x_opt[3]==1 else '不建'} (1套)")
    print(f"📊 预期成效:")
    print(f"   总改造成本: {f_opt[0]:.2f} 万元")
    print(f"   韧性综合增益: {f_opt[1]:.4f} (归一化分值)")
    print(f"   退水时间缩短: {f_opt[2]:.2f} 分钟")

# ==========================================
# 4. 3D 帕累托前沿可视化 (直接放入论文的杀手锏图表)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制所有的 Pareto 解
sc = ax.scatter(pareto_F[:, 0], pareto_F[:, 1], pareto_F[:, 2], 
                c=pareto_F[:, 1], cmap='viridis', s=40, alpha=0.7, label='Pareto 解集')

# 高亮标注三个典型方案
colors = ['red', 'orange', 'fuchsia']
for i, idx in enumerate(typical_indices):
    ax.scatter(pareto_F[idx, 0], pareto_F[idx, 1], pareto_F[idx, 2], 
               color=colors[i], s=150, edgecolors='k', marker='*', label=plan_names[i])

ax.set_xlabel('总改造成本 (万元)')
ax.set_ylabel('韧性综合增益')
ax.set_zlabel('退水时间缩短量 (min)')
ax.set_title('内涝韧性提升改造方案 - 3D 帕累托前沿分析', fontsize=14)
ax.legend(loc='upper left')

plt.colorbar(sc, label='韧性综合增益', pad=0.1)
plt.tight_layout()
plt.savefig("answer4图一_帕累托前沿.png", dpi=300, bbox_inches='tight')
plt.close()
print("已生成: answer4图一_帕累托前沿.png")