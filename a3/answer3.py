import heapq
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 全局参数配置 (显式声明，方便灵敏度调节)
# ==========================================
BETA_DEFAULT = 15.0      # 安全惩罚系数
ALPHA_DEFAULT = 1.0      # 时间权重
PRUNING_BUFFER = 15.0    # 剪枝时间冗余(min)
V_MAX = 500.0            # 基础车速 500 m/min (30 km/h)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

IMG_PREFIX = "answer3图"
SAVE_DIR = "."

COLORS = {
    'stat': '#E63946', 'dyn': '#2A9D8F',
    'node_s': '#2A9D8F', 'node_e': '#E76F51', 'node_m': '#457B9D',
    'edge_normal': '#CED4DA', 'edge_path': '#E63946',
}

# ==========================================
# 1. 基础数据 (附表1、4)
# ==========================================
L = {'L1': 450, 'L2': 320, 'L3': 580, 'L4': 260,
     'L5': 390, 'L6': 410, 'L7': 290, 'L8': 520}

water_depth = {
    'L1': [0, 8.0, 14.0, 18.0, 16.0], 'L2': [0, 12.0, 18.0, 24.0, 20.0],
    'L3': [0, 5.0, 9.0, 12.0, 10.0],  'L4': [0, 7.0, 13.0, 17.0, 15.0],
    'L5': [0, 18.0, 28.0, 32.0, 29.0], 'L6': [0, 6.0, 11.0, 15.0, 13.0],
    'L7': [0, 4.0, 7.0, 9.0, 8.0],    'L8': [0, 10.0, 16.0, 22.0, 19.0]
}
T_points = [0, 15, 30, 45, 60]

# 拓扑定义：L5是陷阱，L7-L8是避险通道
edges = [
    ('起点', 'A', 'L1'), ('A', '终点', 'L5'),   
    ('起点', 'B', 'L4'), ('B', 'D', 'L7'), ('D', '终点', 'L8'),
    ('A', 'C', 'L2'), ('C', '终点', 'L6')
]

graph = {}
for u, v, road in edges:
    graph.setdefault(u, []).append((v, road))
    graph.setdefault(v, []).append((u, road))

# ==========================================
# 2. 状态映射与算法
# ==========================================
def get_h(road_id, t):
    if t >= 60: return water_depth[road_id][-1]
    return np.interp(t, T_points, water_depth[road_id])

def get_capacity_and_safety(h):
    if h < 5: return 1.0, 1.0
    elif 5 <= h < 10: return 0.8, 0.9
    elif 10 <= h < 20: return 0.4, 0.6
    elif 20 <= h < 30: return 0.1, 0.2
    else: return 0.0, 0.0

def baseline_static_simulation(start, end, start_t):
    planned_path = ['起点', 'A', '终点']
    duration = 0.0
    total_saf = 1.0
    curr_t = start_t

    for i in range(len(planned_path)-1):
        u, v = planned_path[i], planned_path[i+1]
        road_id = next(r for n, r in graph[u] if n == v)
        actual_h = get_h(road_id, curr_t)
        c, s = get_capacity_and_safety(actual_h)

        if c == 0:
            return planned_path[:i+1] + [f"(断路)"], duration + 12.0, total_saf * 0.05

        edge_t = L[road_id] / (V_MAX * c)
        duration += edge_t
        curr_t += edge_t
        total_saf *= s

    return planned_path, duration, total_saf

def plot_water_evolution():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')

    t_smooth = np.linspace(0, 60, 200)
    y_L5 = [get_h('L5', t) for t in t_smooth]
    y_L2 = [get_h('L2', t) for t in t_smooth]
    y_L7 = [get_h('L7', t) for t in t_smooth]

    ax.axhspan(30, 40, color='#E63946', alpha=0.15, label='道路完全中断 (>30cm)')
    ax.axhspan(20, 30, color='#F4A261', alpha=0.15, label='通行严重受阻 (20-30cm)')

    ax.plot(t_smooth, y_L5, color='#E63946', lw=3, label='L5 (低洼瓶颈路段)')
    ax.plot(t_smooth, y_L2, color='#2A9D8F', lw=3, label='L2 (中等共建道路)')
    ax.plot(t_smooth, y_L7, color='#457B9D', lw=3, label='L7 (备用绕行道路)')

    ax.set_title('图一：积水深度动态演变', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('时间 (分钟)', fontsize=12)
    ax.set_ylabel('水深 (cm)', fontsize=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 38)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, edgecolor='none', facecolor='#ffffff', shadow=True)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存 {SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png")

# ==========================================
# 3. 时空动态规划算法
# ==========================================
def time_varying_dijkstra(start, end, start_t, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
    pq = [(0.0, start, start_t, [start], 0.0, 1.0)]
    best_cost = {start: 0.0}

    while pq:
        cost, u, curr_t, path, duration, curr_saf = heapq.heappop(pq)
        if u == end: return path, duration, curr_saf

        for v, road_id in graph.get(u, []):
            h = get_h(road_id, curr_t)
            c, s = get_capacity_and_safety(h)
            if c == 0 or s == 0: continue

            edge_t = L[road_id] / (V_MAX * c)
            new_t = curr_t + edge_t
            if v in best_cost and new_t > best_cost[u] + PRUNING_BUFFER: continue

            new_saf = curr_saf * s
            edge_cost = alpha * edge_t - beta * math.log(s)
            new_cost = cost + edge_cost

            if v not in best_cost or new_cost < best_cost[v]:
                best_cost[v] = new_cost
                heapq.heappush(pq, (new_cost, v, new_t, path + [v], duration + edge_t, new_saf))
    return None, float('inf'), 0.0

# ==========================================
# 4. 可视化函数
# ==========================================
def plot_water_evolution():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    t_smooth = np.linspace(0, 60, 200)
    y_L5 = [get_h('L5', t) for t in t_smooth]
    y_L2 = [get_h('L2', t) for t in t_smooth]
    y_L7 = [get_h('L7', t) for t in t_smooth]
    ax.axhspan(30, 40, color='#E63946', alpha=0.15, label='道路完全中断 (>30cm)')
    ax.axhspan(20, 30, color='#F4A261', alpha=0.15, label='通行严重受阻 (20-30cm)')
    ax.plot(t_smooth, y_L5, color='#E63946', lw=3, label='L5 (低洼瓶颈路段)')
    ax.plot(t_smooth, y_L2, color='#2A9D8F', lw=3, label='L2 (中等共建道路)')
    ax.plot(t_smooth, y_L7, color='#457B9D', lw=3, label='L7 (备用绕行道路)')
    ax.set_title('图一：积水深度动态演变', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('时间 (分钟)', fontsize=12)
    ax.set_ylabel('水深 (cm)', fontsize=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 38)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, edgecolor='none', facecolor='#ffffff', shadow=True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存 {SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png")

def plot_network(opt_path):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    G = nx.Graph()
    pos = {'起点': (0, 1), 'A': (1, 1.8), 'B': (1, 0.2), 'C': (2, 1.6), 'D': (2, 0.4), '终点': (3, 1)}
    for u, v, road in edges:
        G.add_edge(u, v, label=f"{road}\n({L[road]}m)")
    path_edges = [(opt_path[i], opt_path[i+1]) for i in range(len(opt_path)-1)] if opt_path else []
    path_edges += [(v, u) for u, v in path_edges]
    edge_colors = [COLORS['edge_path'] if e in path_edges else COLORS['edge_normal'] for e in G.edges()]
    edge_widths = [5.0 if e in path_edges else 2.0 for e in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1600, edgecolors='white', linewidths=2.5,
                           node_color=[COLORS['node_s'] if n=='起点' else COLORS['node_e'] if n=='终点' else COLORS['node_m'] for n in G.nodes()])
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='white', font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))
    ax.set_title('图二：动态路网拓扑与 TDSPP 最优避险路径标注', fontsize=15, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}二_路网拓扑与最优路径.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存 {SAVE_DIR}/{IMG_PREFIX}二_路网拓扑与最优路径.png")

def plot_comparison_chart(stat_res, dyn_res):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor='white')
    labels = ['静态方案', '动态方案']
    times = [stat_res[1], dyn_res[1]]
    safeties = [stat_res[2], dyn_res[2]]
    x = np.arange(len(labels))
    bars1 = ax1.bar(x, times, 0.5, color=[COLORS['stat'], COLORS['dyn']])
    ax1.set_title('疏散耗时对比', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('时间 (分钟)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, max(times) * 1.3)
    bars2 = ax2.bar(x, safeties, 0.5, color=[COLORS['stat'], COLORS['dyn']])
    ax2.set_title('安全度对比', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('安全度', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylim(0, 1.2)
    for ax, bars, is_safe in zip([ax1, ax2], [bars1, bars2], [False, True]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.yaxis.grid(True, linestyle='--', color='#E9ECEF', alpha=0.7)
        ax.set_axisbelow(True)
        for bar in bars:
            height = bar.get_height()
            txt = f'{height:.2f}' if is_safe else f'{height:.1f} min'
            ax.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                         fontsize=12, fontweight='bold')
    fig.suptitle('图三：不同决策机制下的双目标指标对比', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}三_疏散方案对比.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存 {SAVE_DIR}/{IMG_PREFIX}三_疏散方案对比.png")

if __name__ == "__main__":
    START_TIME = 36.0
    print(f"🚀 开始进行动态疏散推演 (出发时刻 t={START_TIME} 分钟)...\n")

    stat_path, stat_dur, stat_saf = baseline_static_simulation('起点', '终点', START_TIME)
    dyn_path, dyn_dur, dyn_saf = time_varying_dijkstra('起点', '终点', START_TIME)

    print("--- 结果比对 ---")
    print(f"❌ 静态方案 (走最短路): {'->'.join(stat_path)}")
    print(f"   结果: 行驶到 L5 时路已断，惩罚后耗时 {stat_dur:.2f} min，安全度跌至 {stat_saf:.4f}\n")
    print(f"✅ 动态方案 (智能避险): {'->'.join(dyn_path)}")
    print(f"   结果: 提前绕开 L5，纯行驶耗时 {dyn_dur:.2f} min，安全到达！安全度 {dyn_saf:.4f}\n")

    print("--- 灵敏度分析：安全惩罚系数 Beta 的影响 ---")
    print(f"{'Beta':<8} {'策略倾向':<12} {'路径':<25} {'耗时':<10} {'安全度':<10}")
    print("-" * 70)
    for b in [0, 5, 15, 30]:
        p, d, s = time_varying_dijkstra('起点', '终点', START_TIME, beta=b)
        desc = "时间优先" if b==0 else "均衡型" if b<=15 else "安全优先"
        print(f"{b:<8} {desc:<12} {'->'.join(p):<25} {d:.2f} min   {s:.4f}")

    print("\n生成图表...")
    plot_water_evolution()
    plot_network(dyn_path)
    plot_comparison_chart((stat_path, stat_dur, stat_saf), (dyn_path, dyn_dur, dyn_saf))
    print("\n📊 图表已更新！")