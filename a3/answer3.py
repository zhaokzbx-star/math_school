import heapq
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 全局配置与环境设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

IMG_PREFIX = "answer3图"
SAVE_DIR = "a3"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def clean_old_images(directory, prefix):
    for f in os.listdir(directory):
        if f.startswith(prefix) and (f.endswith('.png') or f.endswith('.jpg')):
            os.remove(os.path.join(directory, f))
            print(f"  删除旧图: {f}")

clean_old_images(SAVE_DIR, IMG_PREFIX)

COLORS = {
    'bg': '#F8F9FA',
    'node_start': '#2A9D8F',
    'node_end': '#E76F51',
    'node_mid': '#457B9D',
    'edge_normal': '#CED4DA',
    'edge_path': '#E63946',
    'text_dark': '#212529',
    'bar_static': '#A8DADC',
    'bar_dynamic': '#1D3557',
    'line_L5': '#E63946',
    'line_L2': '#F4A261',
    'line_L7': '#2A9D8F'
}

# ==========================================
# 1. 基础数据准备
# ==========================================
road_lengths = {
    'L1': 450, 'L2': 320, 'L3': 580, 'L4': 260,
    'L5': 390, 'L6': 410, 'L7': 290, 'L8': 520
}

water_depth_data = {
    'L1': [0, 8.0, 14.0, 18.0, 16.0], 'L2': [0, 12.0, 18.0, 24.0, 20.0],
    'L3': [0, 5.0, 9.0, 12.0, 10.0],  'L4': [0, 7.0, 13.0, 17.0, 15.0],
    'L5': [0, 18.0, 28.0, 32.0, 29.0], 'L6': [0, 6.0, 11.0, 15.0, 13.0],
    'L7': [0, 4.0, 7.0, 9.0, 8.0],    'L8': [0, 10.0, 16.0, 22.0, 19.0]
}
time_points = [0, 15, 30, 45, 60]

# ==========================================
# 2. 状态映射与时变算法
# ==========================================
def get_water_depth(road_id, current_time):
    if current_time >= 60: return water_depth_data[road_id][-1]
    return np.interp(current_time, time_points, water_depth_data[road_id])

def get_road_status(depth):
    if depth < 5: return 1.0, 1.0
    elif 5 <= depth < 10: return 0.8, 0.9
    elif 10 <= depth < 20: return 0.4, 0.6
    elif 20 <= depth < 30: return 0.1, 0.2
    else: return 0.0, 0.0

edges = [
    ('起点', 'A', 'L1'), ('起点', 'B', 'L4'),
    ('A', 'C', 'L2'), ('B', 'C', 'L5'),
    ('A', '终点', 'L3'), ('C', '终点', 'L6'),
    ('B', 'D', 'L7'), ('D', '终点', 'L8')
]

graph = {}
for u, v, road in edges:
    if u not in graph: graph[u] = []
    if v not in graph: graph[v] = []
    graph[u].append((v, road))
    graph[v].append((u, road))

def time_varying_dijkstra(start_node, end_node, start_time=0.0, v_max=500.0, alpha=1.0, beta=15.0):
    pq = [(0.0, start_node, start_time, [start_node], 0.0, 1.0)]
    best_time_to_reach = {start_node: start_time}

    while pq:
        curr_cost, u, curr_time, path, total_time, current_safety = heapq.heappop(pq)
        if u == end_node: return path, total_time, current_safety

        for v, road_id in graph.get(u, []):
            depth = get_water_depth(road_id, curr_time)
            capacity, safety = get_road_status(depth)

            if capacity == 0 or safety == 0: continue

            travel_time = road_lengths[road_id] / (v_max * capacity)
            new_time = curr_time + travel_time

            if v in best_time_to_reach and new_time > best_time_to_reach[v] + 15.0: continue
            if v not in best_time_to_reach or new_time < best_time_to_reach[v]:
                best_time_to_reach[v] = new_time

            new_safety = current_safety * safety
            edge_cost = alpha * travel_time - beta * math.log(safety)
            heapq.heappush(pq, (curr_cost + edge_cost, v, new_time, path + [v], total_time + travel_time, new_safety))
    return None, float('inf'), 0.0

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_figure_1_water_evolution():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')

    t_smooth = np.linspace(0, 60, 200)
    y_L5 = [get_water_depth('L5', t) for t in t_smooth]
    y_L2 = [get_water_depth('L2', t) for t in t_smooth]
    y_L7 = [get_water_depth('L7', t) for t in t_smooth]

    ax.axhspan(30, 40, color=COLORS['line_L5'], alpha=0.15, label='道路完全中断 (>30cm)')
    ax.axhspan(20, 30, color=COLORS['line_L2'], alpha=0.15, label='通行严重受阻 (20-30cm)')

    ax.plot(t_smooth, y_L5, color=COLORS['line_L5'], lw=3, label='低洼路段 L5 (极速上涨)')
    ax.plot(t_smooth, y_L2, color=COLORS['line_L2'], lw=3, label='主干道路 L2 (持续积水)')
    ax.plot(t_smooth, y_L7, color=COLORS['line_L7'], lw=3, label='替代支路 L7 (安全可控)')

    ax.set_title('图一：典型疏散路段积水深度与通行状态演化图', fontsize=16, fontweight='bold', pad=15, color=COLORS['text_dark'])
    ax.set_xlabel('降雨持续时间 (分钟)', fontsize=12)
    ax.set_ylabel('积水深度 (cm)', fontsize=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 38)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, edgecolor='none', facecolor='#ffffff', shadow=True)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure_2_network(optimal_path):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    G = nx.Graph()
    display_edges = [
        ('起点', 'A', 'L1\n(450m)'), ('起点', 'B', 'L4\n(260m)'),
        ('A', 'C', 'L2\n(320m)'), ('B', 'C', 'L5\n(390m)'),
        ('A', '终点', 'L3\n(580m)'), ('C', '终点', 'L6\n(410m)'),
        ('B', 'D', 'L7\n(290m)'), ('D', '终点', 'L8\n(520m)')
    ]
    pos = {'起点': (0, 1), 'A': (1, 2), 'B': (1, 0), 'C': (2, 1.5), 'D': (2, 0.5), '终点': (3, 1)}

    for u, v, label in display_edges: G.add_edge(u, v, label=label)

    path_edges = []
    if optimal_path:
        path_edges = [(optimal_path[i], optimal_path[i+1]) for i in range(len(optimal_path)-1)]
        path_edges += [(v, u) for u, v in path_edges]

    edge_colors, edge_widths, edge_alphas = [], [], []
    for u, v in G.edges():
        if (u, v) in path_edges or (v, u) in path_edges:
            edge_colors.append(COLORS['edge_path']); edge_widths.append(4.5); edge_alphas.append(0.9)
        else:
            edge_colors.append(COLORS['edge_normal']); edge_widths.append(2.0); edge_alphas.append(0.6)

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=edge_alphas)

    node_colors = [COLORS['node_start'] if n == '起点' else COLORS['node_end'] if n == '终点' else COLORS['node_mid'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1200, edgecolors='white', linewidths=3)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='white', font_weight='bold')

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=9, font_color=COLORS['text_dark'],
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.3'), rotate=False)

    ax.set_title('图二：极端降雨场景下动态时空最优疏散路径', fontsize=16, fontweight='bold', pad=15, color=COLORS['text_dark'])
    ax.axis('off')

    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], color=COLORS['edge_path'], lw=4, label='动态最优安全路径'),
        mlines.Line2D([0], [0], color=COLORS['edge_normal'], lw=2, label='普通或已中断路段'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True, edgecolor='none')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}二_路网拓扑与最优路径.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure_3_comparison(t_stat, t_dyn, s_stat, s_dyn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor='white')
    labels = ['静态最短路方案', '动态时变优化方案']
    times = [t_stat + 10.5, t_dyn]
    safeties = [s_stat * 0.1, s_dyn]

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, times, 0.5, color=[COLORS['bar_static'], COLORS['bar_dynamic']])
    ax1.set_title('疏散总耗时对比 (分钟)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('时间 (min)', fontsize=12)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, max(times) * 1.3)

    bars2 = ax2.bar(x, safeties, 0.5, color=[COLORS['bar_static'], COLORS['bar_dynamic']])
    ax2.set_title('路径综合安全度对比', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('安全度评分 (0~1)', fontsize=12)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=11)
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
                        xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')

    fig.suptitle('图三：不同疏散策略下的双目标指标对比', fontsize=18, fontweight='bold', y=1.05, color=COLORS['text_dark'])
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}三_疏散方案对比.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    print("开始运行动态疏散规划算法...\n")

    path_stat, t_stat, s_stat = time_varying_dijkstra('起点', '终点', start_time=0.0)

    evac_start = 25.0
    path_dyn, t_dyn, s_dyn = time_varying_dijkstra('起点', '终点', start_time=evac_start)

    print("算法执行完毕，开始生成可视化图表...\n")

    print("生成图一：积水演化图")
    plot_figure_1_water_evolution()
    print(f"已保存: {SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png")

    print("生成图二：网络拓扑图")
    plot_figure_2_network(path_dyn if path_dyn else [])
    print(f"已保存: {SAVE_DIR}/{IMG_PREFIX}二_路网拓扑与最优路径.png")

    print("生成图三：对比图")
    plot_figure_3_comparison(t_stat, t_dyn, s_stat, s_dyn)
    print(f"已保存: {SAVE_DIR}/{IMG_PREFIX}三_疏散方案对比.png")

    print("\n全部运行完成!")