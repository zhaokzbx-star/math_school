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
START_TIME = 36.0        # 核心剧本触发点：第 36 分钟开始疏散

# Matplotlib 中文字体与高清配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

IMG_PREFIX = "answer3图"
SAVE_DIR = "."  # 保存在当前目录

COLORS = {
    'stat': '#E63946', 'dyn': '#2A9D8F',
    'node_s': '#2A9D8F', 'node_e': '#E76F51', 'node_m': '#457B9D',
    'edge_normal': '#CED4DA', 'edge_path': '#E63946',
}

# ==========================================
# 1. 基础数据 (附表1、附表4)
# ==========================================
# 附表1：路段长度
L = {'L1': 450, 'L2': 320, 'L3': 580, 'L4': 260,
     'L5': 390, 'L6': 410, 'L7': 290, 'L8': 520}

# 附表4：时序水深数据
water_depth = {
    'L1': [0, 8.0, 14.0, 18.0, 16.0], 'L2': [0, 12.0, 18.0, 24.0, 20.0],
    'L3': [0, 5.0, 9.0, 12.0, 10.0],  'L4': [0, 7.0, 13.0, 17.0, 15.0],
    'L5': [0, 18.0, 28.0, 32.0, 29.0], 'L6': [0, 6.0, 11.0, 15.0, 13.0],
    'L7': [0, 4.0, 7.0, 9.0, 8.0],    'L8': [0, 10.0, 16.0, 22.0, 19.0]
}
T_points = [0, 15, 30, 45, 60]

# 拓扑定义
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
# 2. 状态映射与算法模型
# ==========================================
def get_h(road_id, t):
    """根据时间 t 线性插值获取当前积水深度"""
    if t >= 60: return water_depth[road_id][-1]
    return np.interp(t, T_points, water_depth[road_id])

def get_capacity_and_safety(h):
    """附表3规则映射，超过30cm则能力为0(硬中断)"""
    if h < 5: return 1.0, 1.0
    elif 5 <= h < 10: return 0.8, 0.9
    elif 10 <= h < 20: return 0.4, 0.6
    elif 20 <= h < 30: return 0.1, 0.2
    else: return 0.0, 0.0

def baseline_static_simulation(start, end, start_t):
    """
    对照组：常规静态导航（短视模型）
    以物理距离最短为导向，不预判未来天气，遇到断路则折返受罚。
    """
    # 静态最短路必然是 起点 -> A -> 终点 (450 + 390 = 840m)
    planned_path = ['起点', 'A', '终点']
    duration = 0.0
    total_saf = 1.0
    curr_t = start_t
    
    for i in range(len(planned_path)-1):
        u, v = planned_path[i], planned_path[i+1]
        road_id = next(r for n, r in graph[u] if n == v)
        
        # 车辆到达路口时的【真实水深】
        actual_h = get_h(road_id, curr_t)
        c, s = get_capacity_and_safety(actual_h)
        
        if c == 0:
            # 被水淹没，触发折返惩罚
            return planned_path[:i+1] + ["(断路)"], duration + 12.0, total_saf * 0.05
            
        edge_t = L[road_id] / (V_MAX * c)
        duration += edge_t
        curr_t += edge_t
        total_saf *= s
        
    return planned_path, duration, total_saf

def time_varying_dijkstra(start, end, start_t, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
    """
    实验组：时变 Dijkstra 算法 (TDSPP)
    带有上帝视角的动态时空规划
    """
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
            
            # 剪枝：避免在环路中无限绕
            if v in best_cost and new_t > best_cost.get(v, 0) + PRUNING_BUFFER: continue
            
            new_saf = curr_saf * s
            edge_cost = alpha * edge_t - beta * math.log(s)
            new_cost = cost + edge_cost
            
            if v not in best_cost or new_cost < best_cost[v]:
                best_cost[v] = new_cost
                heapq.heappush(pq, (new_cost, v, new_t, path + [v], duration + edge_t, new_saf))
                
    return None, float('inf'), 0.0

# ==========================================
# 3. 三大核心图表生成模块
# ==========================================

def plot_figure_1_water_evolution():
    """生成图一：积水深度动态演变图 (恢复经典双色背景带美学)"""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    t_smooth = np.linspace(0, 60, 200)
    
    # 恢复经典的双色危险背景预警带
    ax.axhspan(30, 40, color='#E63946', alpha=0.15, label='道路完全中断区 (≥30cm)')
    ax.axhspan(20, 30, color='#F4A261', alpha=0.15, label='通行严重受阻区 (20-30cm)')
    
    # 绘制折线
    ax.plot(t_smooth, [get_h('L5', t) for t in t_smooth], color='#E63946', lw=3, label='L5 (极速上涨)')
    ax.plot(t_smooth, [get_h('L8', t) for t in t_smooth], color='#F4A261', lw=3, label='L8 (缓慢积水)')
    ax.plot(t_smooth, [get_h('L7', t) for t in t_smooth], color='#2A9D8F', lw=3, label='L7 (平稳安全)')

    # 标记触发剧本的出发时刻
    ax.axvline(START_TIME, color='#343A40', linestyle='--', lw=2, label=f'开始紧急疏散 (t={START_TIME}min)')

    ax.set_title('图一：典型疏散路段积水深度与通行状态演化验证', fontsize=16, fontweight='bold', pad=15, color='#212529')
    ax.set_xlabel('降雨持续时间 (分钟)', fontsize=12)
    ax.set_ylabel('积水深度 (cm)', fontsize=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 38)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 图例加上美观的阴影和边框
    ax.legend(loc='upper left', frameon=True, edgecolor='none', facecolor='#ffffff', shadow=True)
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}一_积水深度动态演变.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成: 图一_积水深度动态演变.png")

def plot_figure_2_enhanced_network(opt_path):
    """生成图二：带有ABCD明确节点标注的路网拓扑与最优路径图"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    G = nx.Graph()
    
    # 修正坐标，确保线条绝对不共线重叠
    pos = {
        '起点': (0, 1), 
        'A': (1, 1.8), 'B': (1, 0.2),
        'C': (2, 1.6), 'D': (2, 0.4),
        '终点': (3, 1)
    }
    
    for u, v, road in edges:
        G.add_edge(u, v, label=f"{road}\n({L[road]}m)")

    path_edges = [(opt_path[i], opt_path[i+1]) for i in range(len(opt_path)-1)] if opt_path else []
    path_edges += [(v, u) for u, v in path_edges]

    edge_colors = [COLORS['edge_path'] if e in path_edges else COLORS['edge_normal'] for e in G.edges()]
    edge_widths = [5.0 if e in path_edges else 2.0 for e in G.edges()]

    # 画边
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.8)
    
    # 画节点
    node_colors = [COLORS['node_s'] if n=='起点' else COLORS['node_e'] if n=='终点' else COLORS['node_m'] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1600, edgecolors='white', linewidths=2.5, node_color=node_colors)
    
    # 标注节点名称 (A, B, C, D)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='white', font_weight='bold')
    
    # 标注边名称
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2'))

    ax.set_title('图二：动态路网拓扑与 TDSPP 最优路径标注', fontsize=16, fontweight='bold', pad=20, color='#212529')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}二_路网拓扑与最优路径.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成: 图二_路网拓扑与最优路径.png")

def plot_figure_3_comparison_chart(stat_res, dyn_res):
    """生成图三：动静态机制下的疏散指标对比柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
    labels = ['常规静态方案\n(中途涉水断路)', '动态时变优化\n(上帝视角绕行)']
    
    # 耗时对比
    bars1 = ax1.bar(labels, [stat_res[1], dyn_res[1]], width=0.4, color=[COLORS['stat'], COLORS['dyn']])
    ax1.set_title('疏散路途纯耗时对比 (分钟)', fontsize=14, fontweight='bold', pad=15)
    
    # 安全度对比
    bars2 = ax2.bar(labels, [stat_res[2], dyn_res[2]], width=0.4, color=[COLORS['stat'], COLORS['dyn']])
    ax2.set_title('综合路径安全度乘积 (0-1)', fontsize=14, fontweight='bold', pad=15)

    for ax, bars in zip([ax1, ax2], [bars1, bars2]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', ls='--', alpha=0.5)
        for p in bars:
            val = p.get_height()
            ax.annotate(f"{val:.2f}", (p.get_x() + p.get_width()/2., val),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='#212529')
            
    plt.suptitle(f"图三：降雨第 {START_TIME} 分钟触发紧急疏散的双目标指标对比", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}三_疏散方案对比.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成: 图三_疏散方案对比.png")


# ==========================================
# 4. 主程序执行
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print(f"🚀 正在执行第三问核心算法 (触发时刻 t={START_TIME} min)")
    print("==================================================\n")
    
    # 1. 跑 Baseline (静态)
    stat_path, stat_dur, stat_saf = baseline_static_simulation('起点', '终点', START_TIME)
    
    # 2. 跑 TDSPP (动态)
    dyn_path, dyn_dur, dyn_saf = time_varying_dijkstra('起点', '终点', START_TIME)
    
    # 3. 输出论文对比数据
    print("【1】对比实验结果：")
    print(f"静态方案路径: {' -> '.join(stat_path)}")
    print(f"   => 最终耗时: {stat_dur:.2f} min，安全度: {stat_saf:.4f}")
    
    print(f"动态方案路径: {' -> '.join(dyn_path)}")
    print(f"   => 最终耗时: {dyn_dur:.2f} min，安全度: {dyn_saf:.4f}\n")
    
    # 4. 跑 灵敏度分析 (供论文深入讨论)
    print("【2】权重参数(Beta)灵敏度分析与帕累托扫描：")
    for b in [0, 5, 15, 30]:
        p, d, s = time_varying_dijkstra('起点', '终点', START_TIME, beta=b)
        desc = "时间优先" if b==0 else "安全优先" if b>10 else "折中型"
        print(f"   Beta={b:<2} [{desc:<8}] | 路径: {'->'.join(p):<18} | 耗时: {d:.2f} min | 安全度: {s:.3f}")
    print()

    # 5. 调用生成三张图表
    print("【3】正在生成高清可视化图表...")
    plot_figure_1_water_evolution()
    plot_figure_2_enhanced_network(dyn_path)
    plot_figure_3_comparison_chart((stat_path, stat_dur, stat_saf), (dyn_path, dyn_dur, dyn_saf))
    
    print("\n全部任务执行完毕。")