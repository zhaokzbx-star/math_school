#问题一
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 图片名前缀
IMG_PREFIX = "answer1图"

def clean_old_images(prefix):
    """删除以prefix开头的旧图片文件"""
    for f in os.listdir('.'):
        if f.startswith(prefix) and f.endswith('.png'):
            os.remove(f)
            print(f"  删除旧图: {f}")

# 绘图全局设置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 基础物理数据字典
road_data = {
    '路段编号': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'],
    '长度(m)': [450, 320, 580, 260, 390, 410, 290, 520],
    '优先级': ['主干路', '次干路', '主干路', '支路', '次干路', '主干路', '支路', '次干路'],
    '路面类型': ['沥青', '沥青', '沥青', '水泥', '沥青', '沥青', '水泥', '沥青'],
    '坡度(‰)': [1.8, 2.5, 1.2, 3.1, 2.0, 1.5, 3.5, 2.2],
    '排水能力(L/s)': [850, 620, 850, 380, 620, 850, 380, 620],
    '最低标高(m)': [12.5, 11.2, 13.0, 10.5, 10.0, 14.1, 11.8, 12.2], 
    '径流系数': [0.85, 0.85, 0.85, 0.90, 0.85, 0.85, 0.90, 0.85] 
}

# 降雨时序与观测数据
rainfall_time = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
rainfall_depth_mm = [0, 4, 6, 9, 11, 10, 8, 6, 4, 2, 2, 1, 1]

observed_data = {
    15: [8.0, 12.0, 5.0, 7.0, 18.0, 6.0, 4.0, 10.0],
    30: [14.0, 18.0, 9.0, 13.0, 28.0, 11.0, 7.0, 16.0],
    45: [18.0, 24.0, 12.0, 17.0, 32.0, 15.0, 9.0, 22.0],
    60: [16.0, 20.0, 10.0, 15.0, 29.0, 13.0, 8.0, 19.0]
}

road_width_map = {'主干路': 30, '次干路': 20, '支路': 15}

def preprocess_data():
    df = pd.DataFrame(road_data)
    df['路宽(m)'] = df['优先级'].map(road_width_map)
    df['排水能力(m³/min)'] = df['排水能力(L/s)'] * 60.0 / 1000.0

    # 样条插值生成逐分钟降雨率
    rain_rate_mm_min = [val / 5.0 if t > 0 else 0 for t, val in zip(rainfall_time, rainfall_depth_mm)]
    cs = CubicSpline(rainfall_time, rain_rate_mm_min)
    t_minutes = np.arange(0, 61, 1)
    rain_rate_smooth = np.maximum(0, cs(t_minutes))
    rainfall_rate_m_min = rain_rate_smooth / 1000.0

    return df, rainfall_rate_m_min, t_minutes

def simulate_water_depth(params, df_roads, rainfall_rate_m_min):
    # 物理参数：Catchment_Factor(街区面积倍数), Cd(管网老化折减系数)
    Catchment_Factor, Cd = params
    n_roads = len(df_roads)
    n_time = len(rainfall_rate_m_min)

    h_m = np.zeros(n_roads)
    h_cm_history = np.zeros((n_time, n_roads))
    dt = 1.0

    base_area = df_roads['长度(m)'] * df_roads['路宽(m)']
    
    # 物理机制1：基于坡度梯度的街区真实汇水效应放大
    catchment_area = base_area.values * (1.0 + Catchment_Factor * df_roads['坡度(‰)'].values / 10.0)
    
    psi = df_roads['径流系数'].values
    drain_capacity = df_roads['排水能力(m³/min)'].values
    is_cement = (df_roads['路面类型'] == '水泥').values
    Z_surf = df_roads['最低标高(m)'].values
    
    # 硬性锁死物理常识：水泥地的下渗率极低 (约 3mm/h)
    I_cement = 0.00005 

    for t in range(n_time):
        # 推理公式法 Q = psi * q * F
        R_volume = rainfall_rate_m_min[t] * catchment_area * psi * dt
        infiltration_volume = np.where(is_cement, I_cement * base_area.values * dt, 0.0)
        
        current_water_volume = h_m * base_area.values
        
        # 物理机制2：引入地下管网的“顶托效应”(Backwater Effect)
        # 当积水加深(例如向40cm逼近时)，管网内水压同样增大，导致内外压差减小，排水效率呈线性崩溃
        backwater_factor = np.maximum(0.1, 1.0 - (h_m / 0.4))
        dynamic_drain = drain_capacity * Cd * backwater_factor * dt 
        
        Q_volume = np.minimum(dynamic_drain, current_water_volume + R_volume)
        new_water_volume = current_water_volume + R_volume - Q_volume - infiltration_volume
        
        h_m = np.maximum(0, new_water_volume / base_area.values)
        
        # 物理机制3：城市低洼漫溢模型
        for i in range(n_roads):
            if h_m[i] > 0.15: # 积水越过15cm路缘石
                overflow_vol = (h_m[i] - 0.15) * base_area.values[i] * 0.2 
                lowest_idx = np.argmin(Z_surf)
                
                # 水往低处流
                if Z_surf[lowest_idx] < Z_surf[i]:
                    h_m[i] -= overflow_vol / base_area.values[i]
                    h_m[lowest_idx] += overflow_vol / base_area.values[lowest_idx]
                    
        h_cm_history[t, :] = h_m * 100.0

    return h_cm_history

def objective_function(params, df_roads, rainfall_rate_m_min, obs_data):
    h_cm_history = simulate_water_depth(params, df_roads, rainfall_rate_m_min)
    simulated, observed = [], []
    for t in [15, 30, 45, 60]:
        simulated.extend(h_cm_history[t, :])
        observed.extend(obs_data[t])
    
    rmse = np.sqrt(np.mean((np.array(simulated) - np.array(observed))**2))
    return rmse

def map_rules(h):
    capacity = np.where(h < 5, 1.0,
                np.where(h < 10, 0.8,
                np.where(h < 20, 0.4,
                np.where(h < 30, 0.1, 0.0))))
    safety = np.where(h < 5, 1.0,
             np.where(h < 10, 0.9,
             np.where(h < 20, 0.6,
             np.where(h < 30, 0.2, 0.0))))
    return capacity, safety

if __name__ == "__main__":
    print("🚀 开始执行具有强物理约束的城市内涝演化寻优...")
    clean_old_images(IMG_PREFIX)

    df_roads, rain_rate, t_mins = preprocess_data()

    # 初始参数: [坡度汇水乘数(街区面积倍数), 管网基础折减系数Cd]
    init_params = [20.0, 0.6]  
    bounds = [(5.0, 40.0), (0.1, 1.0)]

    res = minimize(objective_function, init_params, args=(df_roads, rain_rate, observed_data),
                   method='L-BFGS-B', bounds=bounds)

    best_Catchment, best_Cd = res.x
    print(f"✅ 标定完成! 最终损失 RMSE: {res.fun:.3f} cm")
    print(f"👉 物理参数反演结果: 街区汇水乘数 = {best_Catchment:.2f}, 管网基础折减系数 Cd = {best_Cd:.3f}\n")

    final_h_history = simulate_water_depth(res.x, df_roads, rain_rate)

    print("📊 最终时序输出矩阵（部分展示）：")
    cols = ['路段']
    for t in [15, 30, 45, 60]:
        cols.extend([f'{t}m水深', f'{t}m通行', f'{t}m安全'])

    res_df = []
    for i, road in enumerate(df_roads['路段编号']):
        row = [road]
        for t in [15, 30, 45, 60]:
            h = final_h_history[t, i]
            c, s = map_rules(h)
            row.extend([f"{h:.1f}", f"{c:.1f}", f"{s:.1f}"])
        res_df.append(row)

    display_df = pd.DataFrame(res_df, columns=cols)
    print(display_df.to_string(index=False))

    # ---------------- 绘图可视化部分 ----------------
    colors = sns.color_palette("Set2", 8)

    # 图一：动态演化折线图
    plt.figure(figsize=(12, 6))
    for i, road in enumerate(df_roads['路段编号']):
        if road == 'L5':
            plt.plot(t_mins, final_h_history[:, i], color='#E63946', linewidth=4,
                     zorder=10, label='L5 (最脆弱低洼积水点)')
        else:
            plt.plot(t_mins, final_h_history[:, i], color=colors[i], linewidth=2,
                     alpha=0.8, label=road)

    plt.title('城市极端降雨下路网积水深度时序动态演化曲线', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('降雨历时 (分钟)', fontsize=12)
    plt.ylabel('地面积水深度 (cm)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'{IMG_PREFIX}一_积水深度动态演化.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📁 已保存: {IMG_PREFIX}一_积水深度动态演化.png")

    # 图二：拟合散点图 (验证物理模型的威力)
    obs_list, sim_list = [], []
    for t in [15, 30, 45, 60]:
        sim_list.extend(final_h_history[t, :])
        obs_list.extend(observed_data[t])
    
    obs_array = np.array(obs_list)
    sim_array = np.array(sim_list)
    
    rmse = np.sqrt(np.mean((sim_array - obs_array)**2))
    mae = np.mean(np.abs(sim_array - obs_array))
    ss_tot = np.sum((obs_array - np.mean(obs_array))**2)
    ss_res = np.sum((obs_array - sim_array)**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    plt.figure(figsize=(7, 7))
    plt.scatter(obs_array, sim_array, c='#457B9D', alpha=0.8, s=80, edgecolors='white', linewidths=1.5)
    max_val = max(np.max(obs_array), np.max(sim_array))
    plt.plot([0, max_val+5], [0, max_val+5], color='#E63946', linestyle='--', linewidth=2, label='1:1 完美拟合基准线')

    bbox_props = dict(boxstyle="round,pad=0.6", fc="#F1FAEE", ec="#1D3557", lw=1.5)
    textstr = f"动力学模型评估指标:\nRMSE = {rmse:.2f} cm\nMAE = {mae:.2f} cm\n$R^2$ = {r2:.3f}"
    plt.text(2, max_val-2, textstr, fontsize=12, fontweight='bold', color='#1D3557', bbox=bbox_props)

    plt.title('水动力物理模型实测验证：预测值 vs 真实观测值', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('真实观测积水深度 (cm)', fontsize=12)
    plt.ylabel('模型推演积水深度 (cm)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{IMG_PREFIX}二_拟合散点图.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📁 已保存: {IMG_PREFIX}二_拟合散点图.png")

    # 图三：热力图
    key_times = [15, 30, 45, 60]
    time_labels = ['15 min', '30 min', '45 min', '60 min']
    road_names = df_roads['路段编号'].values
    capacity_matrix = np.zeros((len(road_names), len(key_times)))

    for i, t in enumerate(key_times):
        capacity, _ = map_rules(final_h_history[t, :])
        capacity_matrix[:, i] = capacity

    plt.figure(figsize=(9, 6))
    ax = sns.heatmap(capacity_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                     xticklabels=time_labels, yticklabels=road_names,
                     vmin=0, vmax=1, cbar_kws={'label': '交通通行折减系数'},
                     linewidths=1, linecolor='white', annot_kws={"size": 12, "weight": "bold"})

    plt.title('内涝灾害下路网关键节点时空交通阻抗演变热力图', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('降雨推进阶段', fontsize=12)
    plt.ylabel('主干/次干路段编号', fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{IMG_PREFIX}三_通行能力热力图.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📁 已保存: {IMG_PREFIX}三_通行能力热力图.png")

    print("\n🎉 全部运行结束！如果觉得这次跑出的图给力，可以放心粘贴进论文了！")

    #问题二
    import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import truncnorm

# ==========================================
# 0. 环境配置、路径管理与旧图清理
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

IMG_PREFIX = "answer2图"
SAVE_DIR = "a2"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def clean_old_images(directory, prefix):
    for f in os.listdir(directory):
        if f.startswith(prefix) and (f.endswith('.png') or f.endswith('.jpg')):
            os.remove(os.path.join(directory, f))

clean_old_images(SAVE_DIR, IMG_PREFIX)

# ==========================================
# 1. 修正后的数据定义 (修正 L7 数据)
# ==========================================

# 1.1 系统韧性指标 (附表5)
indicator_data = {
    'Indicator': ['路网连通度', '排水能力达标率', '低洼路段占比', '积水消退半衰期',
                  '应急响应时间', '抢修覆盖效率', '绿色调蓄容积', '预警覆盖率', '疏散点位密度'],
    'Dimension': ['抵御能力', '抵御能力', '抵御能力', '恢复能力', '恢复能力', '恢复能力', '适应能力', '适应能力', '适应能力'],
    'Direction': ['+', '+', '-', '-', '-', '+', '+', '+', '+'],
    'Actual': [72.0, 68.0, 24.0, 45.0, 12.0, 65.0, 1200.0, 75.0, 2.1],
    'Ideal': [100.0, 100.0, 0.0, 20.0, 5.0, 100.0, 3500.0, 100.0, 5.0],
    'Worst': [0.0, 0.0, 100.0, 120.0, 60.0, 0.0, 0.0, 0.0, 0.0]
}
df_idx = pd.DataFrame(indicator_data)

# 1.2 物理路段多维数据 (修正 L7 排水能力为 380, 并加入坡度和管径)
road_physics = {
    'Road': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'],
    'Elevation': [201.2, 200.7, 199.3, 202.5, 198.9, 200.1, 203.4, 201.1],
    'Drain_Cap': [850, 620, 850, 380, 620, 850, 380, 550],
    'Slope': [1.8, 2.5, 1.2, 3.1, 2.0, 1.5, 2.8, 2.2],
    'Pipe_Dia': [1200, 1000, 1200, 800, 1000, 1200, 800, 1000],
    'Max_Depth': [18.0, 24.0, 12.0, 17.0, 32.0, 15.0, 9.0, 22.0]
}
df_roads = pd.DataFrame(road_physics)

# ==========================================
# 2. 核心算法：MC-EWM-TOPSIS (系统层)
# ==========================================

def get_truncated_normal(mean, low, high, sd):
    return truncnorm((low - mean) / (sd + 1e-7), (high - mean) / (sd + 1e-7), loc=mean, scale=sd)

def run_system_topsis(df, n_samples=500):
    np.random.seed(2026)
    sim_matrix = np.zeros((n_samples, len(df)))
    for i in range(len(df)):
        low, high = min(df['Worst'][i], df['Ideal'][i]), max(df['Worst'][i], df['Ideal'][i])
        gen = get_truncated_normal(df['Actual'][i], low, high, abs(high-low)*0.15)
        sim_matrix[:, i] = gen.rvs(n_samples)

    # 极向标准化
    norm_sim = np.zeros_like(sim_matrix)
    for i in range(len(df)):
        v = sim_matrix[:, i]
        if df['Direction'][i] == '+':
            norm_sim[:, i] = (v - df['Worst'][i]) / (df['Ideal'][i] - df['Worst'][i] + 1e-9)
        else:
            norm_sim[:, i] = (df['Worst'][i] - v) / (df['Worst'][i] - df['Ideal'][i] + 1e-9)

    # EWM (数值稳定版本)
    norm_sim_safe = np.clip(norm_sim, 1e-10, 1)  # 先clip防止log(0)
    col_sums = np.sum(norm_sim_safe, axis=0) + 1e-10
    p = norm_sim_safe / col_sums  # 再归一化
    p = np.clip(p, 1e-10, 1)  # 确保概率不为0
    e = - (1 / np.log(n_samples)) * np.sum(p * np.log(p), axis=0)
    e = np.clip(e, 1e-10, 1)  # 防止熵为0导致权重计算问题
    weights = (1 - e) / np.sum(1 - e)

    # TOPSIS 现状打分
    norm_act = np.array([(df['Actual'][i]-df['Worst'][i])/(df['Ideal'][i]-df['Worst'][i]) if df['Direction'][i]=='+'
                         else (df['Worst'][i]-df['Actual'][i])/(df['Worst'][i]-df['Ideal'][i]) for i in range(len(df))])
    z_act = norm_act * weights
    d_pos = np.sqrt(np.sum((z_act - 1.0*weights)**2))
    d_neg = np.sqrt(np.sum((z_act - 0.0*weights)**2))
    return d_neg / (d_pos + d_neg), weights, norm_act

sys_score, sys_weights, sys_norm_act = run_system_topsis(df_idx)

# ==========================================
# 3. 创新算法：多准则路段脆弱性诊断 (路段层)
# ==========================================

def run_road_vulnerability(df):
    v_matrix = pd.DataFrame()
    v_matrix['Elevation'] = (df['Elevation'].max() - df['Elevation']) / (df['Elevation'].max() - df['Elevation'].min())
    v_matrix['Drain_Cap'] = (df['Drain_Cap'].max() - df['Drain_Cap']) / (df['Drain_Cap'].max() - df['Drain_Cap'].min())
    v_matrix['Slope'] = (df['Slope'] - df['Slope'].min()) / (df['Slope'].max() - df['Slope'].min())
    v_matrix['Pipe_Dia'] = (df['Pipe_Dia'].max() - df['Pipe_Dia']) / (df['Pipe_Dia'].max() - df['Pipe_Dia'].min())
    v_matrix['Max_Depth'] = (df['Max_Depth'] - df['Max_Depth'].min()) / (df['Max_Depth'].max() - df['Max_Depth'].min())

    w = np.array([0.15, 0.15, 0.15, 0.15, 0.4])
    z = v_matrix.values * w
    d_pos = np.sqrt(np.sum((z - 1.0*w)**2, axis=1))
    d_neg = np.sqrt(np.sum((z - 0.0*w)**2, axis=1))
    return d_neg / (d_pos + d_neg)

df_roads['V_Index'] = run_road_vulnerability(df_roads)

# ==========================================
# 4. 可视化
# ==========================================

# 4.1 权重对比
plt.figure(figsize=(10, 6))
plt.bar(np.arange(9)-0.2, [0.12,0.13,0.1,0.15,0.12,0.1,0.11,0.09,0.08], 0.4, label='给定权重', color='#A8DADC')
plt.bar(np.arange(9)+0.2, sys_weights, 0.4, label='MC-EWM客观权重', color='#457B9D')
plt.xticks(range(9), df_idx['Indicator'], rotation=25)
plt.title('评价指标权重分配对比 (基于500次MC收敛测试)', fontsize=14)
plt.legend(); plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}一_权重对比.png", dpi=300)

# 4.2 雷达图
angles = np.linspace(0, 2*np.pi, 9, endpoint=False).tolist()
stats = np.concatenate((sys_norm_act, [sys_norm_act[0]]))
angles += angles[:1]
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='#1D3557', alpha=0.3)
ax.plot(angles, stats, color='#1D3557', marker='o', markersize=8, linewidth=2)
ax.fill_between(np.linspace(0, 2*np.pi/3, 50), 0, 1, color='lime', alpha=0.08, label='抵御能力')
ax.fill_between(np.linspace(2*np.pi/3, 4*np.pi/3, 50), 0, 1, color='skyblue', alpha=0.08, label='恢复能力')
ax.fill_between(np.linspace(4*np.pi/3, 2*np.pi, 50), 0, 1, color='salmon', alpha=0.08, label='适应能力')
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], df_idx['Indicator'], fontsize=10)
ax.set_ylim(0, 1)
plt.title(f'系统韧性分维度诊断 (现状得分: {sys_score:.3f})', pad=25, fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), fontsize=12)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}二_韧性诊断雷达图.png", dpi=300, bbox_inches='tight')

# 4.3 敏感度
sensitivities = []
for i in range(9):
    w_new = sys_weights.copy(); w_new[i] *= 1.1; w_new /= w_new.sum()
    norm_act = sys_norm_act
    z = norm_act * w_new
    sc = np.sqrt(np.sum(z**2)) / (np.sqrt(np.sum(z**2)) + np.sqrt(np.sum((z-w_new)**2)))
    sensitivities.append(abs(sc - sys_score)/(sys_score*0.1))

plt.figure(figsize=(10, 5))
diag_idx = pd.DataFrame({'Indicator': df_idx['Indicator'], 'S': sensitivities}).sort_values('S')
bars = plt.barh(diag_idx['Indicator'], diag_idx['S'], color=['#E63946' if x > 0.08 else '#457B9D' for x in diag_idx['S']])
plt.bar_label(bars, fmt='%.3f', padding=3)
plt.title('系统薄弱环节识别 (敏感度分析)'); plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}三_薄弱环节识别.png", dpi=300)

# 4.4 脆弱路段
plt.figure(figsize=(10, 5))
diag_road = df_roads.sort_values('V_Index', ascending=False)
bars = sns.barplot(x='Road', y='V_Index', hue='Road', data=diag_road, palette='Reds_r', legend=False)
plt.bar_label(bars.containers[0], fmt='%.3f')
plt.ylabel('多维物理脆弱性指数 (PVI)')
plt.title('薄弱路段深度诊断 (集成地形/工程/致灾因子)'); plt.savefig(f"{SAVE_DIR}/{IMG_PREFIX}四_薄弱路段识别.png", dpi=300)

# ==========================================
# 5. 打印报告
# ==========================================
print(f"\n【修正后的评估报告】\n1. 系统综合韧性评分: {sys_score:.4f} (状态: 中度脆弱)\n2. L7数据已修正 (Drain_Cap=380)\n3. 薄弱路段(PVI Top3):")
print(diag_road[['Road', 'Elevation', 'Max_Depth', 'V_Index']].head(3))

#问题三
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

    #问题四
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