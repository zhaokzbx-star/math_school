import heapq
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# ==========================================
# 0. 全局配置与路径管理
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

SAVE_DIR = "a3"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

IMG_PREFIX = "answer3图"

def clean_old_images(directory, prefix):
    for f in os.listdir(directory):
        if f.startswith(prefix) and f.endswith('.png'):
            os.remove(os.path.join(directory, f))
            print(f"  删除旧图: {f}")

clean_old_images(SAVE_DIR, IMG_PREFIX)

COLORS = {
    'stat': '#A8DADC', 'dyn': '#1D3557',
    'edge_path': '#E63946', 'edge_norm': '#CED4DA',
    'node_s': '#2A9D8F', 'node_e': '#E76F51', 'node_m': '#457B9D',
    'L5': '#E63946', 'L2': '#F4A261', 'L7': '#2A9D8F'
}

# ==========================================
# 1. 基础数据定义
# ==========================================
V_MAX = 500.0
L = {'L1': 450, 'L2': 320, 'L3': 580, 'L4': 260,
     'L5': 390, 'L6': 410, 'L7': 290, 'L8': 520}

water_depth = {
    'L1': [0, 8.0, 14.0, 18.0, 16.0], 'L2': [0, 12.0, 18.0, 24.0, 20.0],
    'L3': [0, 5.0, 9.0, 12.0, 10.0],  'L4': [0, 7.0, 13.0, 17.0, 15.0],
    'L5': [0, 18.0, 28.0, 32.0, 29.0], 'L6': [0, 6.0, 11.0, 15.0, 13.0],
    'L7': [0, 4.0, 7.0, 9.0, 8.0],    'L8': [0, 10.0, 16.0, 22.0, 19.0]
}
T_points = [0, 15, 30, 45, 60]

edges = [('起点', 'A', 'L1'), ('起点', 'B', 'L4'), ('A', 'C', 'L2'), ('B', 'C', 'L5'),
         ('A', '终点', 'L3'), ('C', '终点', 'L6'), ('B', 'D', 'L7'), ('D', '终点', 'L8')]

graph = {}
for u, v, road in edges:
    graph.setdefault(u, []).append((v, road))
    graph.setdefault(v, []).append((u, road))

# ==========================================
# 2. 状态映射模型
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

# ==========================================
# 3. 算法核心
# ==========================================
def true_static_myopic_simulation(start, end, start_t):
    pq = [(0.0, start, [start])]
    best_c = {start: 0.0}
    planned_path = None
    while pq:
        cost, u, path = heapq.heappop(pq)
        if u == end:
            planned_path = path
            break
        for v, road_id in graph.get(u, []):
            h_static = get_h(road_id, start_t)
            c, _ = get_capacity_and_safety(h_static)
            if c == 0: continue
            edge_t = L[road_id] / (V_MAX * c)
            if v not in best_c or cost + edge_t < best_c[v]:
                best_c[v] = cost + edge_t
                heapq.heappush(pq, (cost + edge_t, v, path + [v]))

    if not planned_path: return planned_path, float('inf'), 0.0, "规划失败"

    curr_t = start_t
    duration = 0.0
    total_saf = 1.0
    actual_path = [start]

    for i in range(len(planned_path)-1):
        u, v = planned_path[i], planned_path[i+1]
        road_id = next(r for n, r in graph[u] if n == v)
        actual_h = get_h(road_id, curr_t)
        c, s = get_capacity_and_safety(actual_h)

        if c == 0:
            duration += 10.0
            total_saf *= 0.1
            return actual_path + [f"({road_id}遇阻中断)"], duration, total_saf, "途中被淹断"

        edge_t = L[road_id] / (V_MAX * c)
        duration += edge_t
        curr_t += edge_t
        total_saf *= s
        actual_path.append(v)

    return actual_path, duration, total_saf, "成功到达"

def time_varying_dijkstra(start, end, start_t, alpha=1.0, beta=15.0):
    pq = [(0.0, start, start_t, [start], 0.0, 1.0)]
    best_cost = {start: 0.0}

    while pq:
        cost, u, curr_t, path, duration, curr_saf = heapq.heappop(pq)

        if u == end:
            return path, duration, curr_saf

        for v, road_id in graph.get(u, []):
            h = get_h(road_id, curr_t)
            c, s = get_capacity_and_safety(h)
            if c == 0 or s == 0: continue

            edge_t = L[road_id] / (V_MAX * c)
            new_t = curr_t + edge_t
            new_saf = curr_saf * s
            edge_cost = alpha * edge_t - beta * math.log(s)
            new_cost = cost + edge_cost

            if v not in best_cost or new_cost < best_cost[v]:
                best_cost[v] = new_cost
                heapq.heappush(pq, (new_cost, v, new_t, path + [v], duration + edge_t, new_saf))

    return None, float('inf'), 0.0

# ==========================================
# 4. 灵敏度分析
# ==========================================
def run_sensitivity_and_plot(start_t):
    betas = np.linspace(0, 40, 5)
    results = []
    for b in betas:
        p, d, s = time_varying_dijkstra('起点', '终点', start_t, alpha=1.0, beta=b)
        if p and (p, round(d,2), round(s,3)) not in [res[1:] for res in results]:
            results.append((b, p, round(d,2), round(s,3)))

    print("\n--- 动态双目标灵敏度分析 (出发时刻 t={}) ---".format(start_t))
    print("策略倾向\tBeta权重\t推荐路径\t\t路途耗时(min)\t绝对到达(min)\t综合安全度")
    for b, p, d, s in results:
        strat = "极速冒进型" if b < 5 else "均衡折中型" if b < 25 else "绝对安全型"
        path_str = "->".join(p)
        print(f"{strat}\t{b:^8}\t{path_str:<20}\t{d:^10}\t{start_t+d:^10}\t{s:^10}")

    return results[-1]

# ==========================================
# 5. 可视化
# ==========================================
def plot_water_evolution():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')

    t_smooth = np.linspace(0, 60, 200)
    y_L5 = [get_h('L5', t) for t in t_smooth]
    y_L2 = [get_h('L2', t) for t in t_smooth]
    y_L7 = [get_h('L7', t) for t in t_smooth]

    ax.axhspan(30, 40, color=COLORS['L5'], alpha=0.15, label='道路完全中断 (>30cm)')
    ax.axhspan(20, 30, color=COLORS['L2'], alpha=0.15, label='通行严重受阻 (20-30cm)')

    ax.plot(t_smooth, y_L5, color=COLORS['L5'], lw=3, label='L5 (低洼路段)')
    ax.plot(t_smooth, y_L2, color=COLORS['L2'], lw=3, label='L2 (主干道路)')
    ax.plot(t_smooth, y_L7, color=COLORS['L7'], lw=3, label='L7 (支路)')

    ax.set_title('图一：积水深度动态演化', fontsize=16, fontweight='bold', pad=15)
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
    print(f"已保存: {SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png")

def plot_network(optimal_path):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    G = nx.Graph()
    display_edges = [
        ('起点', 'A', 'L1'), ('起点', 'B', 'L4'),
        ('A', 'C', 'L2'), ('B', 'C', 'L5'),
        ('A', '终点', 'L3'), ('C', '终点', 'L6'),
        ('B', 'D', 'L7'), ('D', '终点', 'L8')
    ]
    pos = {'起点': (0, 1), 'A': (1, 2), 'B': (1, 0), 'C': (2, 1.5), 'D': (2, 0.5), '终点': (3, 1)}

    for u, v, label in display_edges:
        G.add_edge(u, v, label=label)

    path_edges = []
    if optimal_path:
        path_edges = [(optimal_path[i], optimal_path[i+1]) for i in range(len(optimal_path)-1)]
        path_edges += [(v, u) for u, v in path_edges]

    normal_edges = [(u, v) for u, v in G.edges() if not ((u, v) in path_edges or (v, u) in path_edges)]
    path_edge_list = [(u, v) for u, v in G.edges() if (u, v) in path_edges or (v, u) in path_edges]

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=normal_edges, width=2.0, edge_color=COLORS['edge_norm'])
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edge_list, width=4.5, edge_color=COLORS['edge_path'])

    node_colors = []
    for node in G.nodes():
        if node == '起点': node_colors.append(COLORS['node_s'])
        elif node == '终点': node_colors.append(COLORS['node_e'])
        else: node_colors.append(COLORS['node_m'])

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1200, edgecolors='white', linewidths=3)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='white', font_weight='bold')

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=9,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'),
                                 rotate=False)

    ax.set_title('图二：路网拓扑与最优路径', fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')

    legend_elements = [
        mlines.Line2D([0], [0], color=COLORS['edge_path'], lw=4, label='最优路径'),
        mlines.Line2D([0], [0], color=COLORS['edge_norm'], lw=2, label='普通路段'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True, edgecolor='none')

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}二_路网拓扑与最优路径.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {SAVE_DIR}/{IMG_PREFIX}二_路网拓扑与最优路径.png")

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

    fig.suptitle('图三：不同疏散策略下的双目标指标对比', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}三_疏散方案对比.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {SAVE_DIR}/{IMG_PREFIX}三_疏散方案对比.png")

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    start_time = 25.0
    print("开始进行动态疏散推演...")

    best_beta, dyn_path, dyn_dur, dyn_saf = run_sensitivity_and_plot(start_time)
    dyn_res = (dyn_path, dyn_dur, dyn_saf)

    stat_path, stat_dur, stat_saf, status = true_static_myopic_simulation('起点', '终点', start_time)
    stat_res = (stat_path, stat_dur, stat_saf)

    print("\n--- 核心对比实验 (出发时刻 t={}) ---".format(start_time))
    print(f"静态导航: {'->'.join(stat_path)} [{status}]")
    print(f"   耗时: {stat_dur:.2f} min, 安全度: {stat_saf:.4f}")

    print(f"动态时空规划: {'->'.join(dyn_path)} [顺利到达]")
    print(f"   纯耗时: {dyn_dur:.2f} min (绝对到达: {start_time+dyn_dur:.2f} 分), 安全度: {dyn_saf:.4f}")

    print("\n生成图表...")
    plot_water_evolution()
    plot_network(dyn_path)
    plot_comparison_chart(stat_res, dyn_res)

    print("\n运行完成！所有文件已保存到 a3 文件夹。")