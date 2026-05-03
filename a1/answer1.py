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

# 数据字典
road_data = {
    '路段编号': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'],
    '长度(m)': [450, 320, 580, 260, 390, 410, 290, 520],
    '优先级': ['主干路', '次干路', '主干路', '支路', '次干路', '主干路', '支路', '次干路'],
    '路面类型': ['沥青', '沥青', '沥青', '水泥', '沥青', '沥青', '水泥', '沥青'],
    '坡度(‰)': [1.8, 2.5, 1.2, 3.1, 2.0, 1.5, 3.5, 2.2],
    '排水能力(L/s)': [850, 620, 850, 380, 620, 850, 380, 620]
}

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

    rain_rate_mm_min = [val / 5.0 if t > 0 else 0 for t, val in zip(rainfall_time, rainfall_depth_mm)]
    cs = CubicSpline(rainfall_time, rain_rate_mm_min)
    t_minutes = np.arange(0, 61, 1)
    rain_rate_smooth = np.maximum(0, cs(t_minutes))
    rainfall_rate_m_min = rain_rate_smooth / 1000.0

    return df, rainfall_rate_m_min, t_minutes

def simulate_water_depth(params, df_roads, rainfall_rate_m_min):
    theta, I_cement = params
    n_roads = len(df_roads)
    n_time = len(rainfall_rate_m_min)

    h_m = np.zeros(n_roads)
    h_cm_history = np.zeros((n_time, n_roads))
    dt = 1.0

    base_area = df_roads['长度(m)'] * df_roads['路宽(m)']
    gamma = 1.0 + theta * df_roads['坡度(‰)'] / 1000.0
    catchment_area = base_area * gamma

    drain_capacity = df_roads['排水能力(m³/min)'].values
    is_cement = (df_roads['路面类型'] == '水泥').values

    for t in range(n_time):
        R_volume = rainfall_rate_m_min[t] * catchment_area.values * dt
        infiltration_volume = np.where(is_cement, I_cement * base_area.values * dt, 0.0)
        current_water_volume = h_m * base_area.values
        max_possible_drain = drain_capacity * dt
        Q_volume = np.minimum(max_possible_drain, current_water_volume + R_volume)
        new_water_volume = current_water_volume + R_volume - Q_volume - infiltration_volume
        h_m = np.maximum(0, new_water_volume / base_area.values)
        h_cm_history[t, :] = h_m * 100.0

    return h_cm_history

def objective_function(params, df_roads, rainfall_rate_m_min, obs_data):
    if params[0] < 0 or params[1] < 0:
        return 99999.0
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
    print("开始模型参数标定...")
    clean_old_images(IMG_PREFIX)

    df_roads, rain_rate, t_mins = preprocess_data()

    init_params = [2000.0, 0.001]
    bounds = [(0, 10000), (0, 0.05)]

    res = minimize(objective_function, init_params, args=(df_roads, rain_rate, observed_data),
                   method='L-BFGS-B', bounds=bounds)

    best_theta, best_I_cement = res.x
    print(f"标定完成! RMSE: {res.fun:.3f} cm")
    print(f"theta = {best_theta:.2f}, I_cement = {best_I_cement:.6f} m/min\n")

    final_h_history = simulate_water_depth(res.x, df_roads, rain_rate)

    print("最终结果输出矩阵：")
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

    colors = sns.color_palette("Set2", 8)

    # 图一：动态演化折线图
    plt.figure(figsize=(12, 6))
    for i, road in enumerate(df_roads['路段编号']):
        if road == 'L5':
            plt.plot(t_mins, final_h_history[:, i], color='#E63946', linewidth=4,
                     zorder=10, label='L5 (重度积水洼地)')
        else:
            plt.plot(t_mins, final_h_history[:, i], color=colors[i], linewidth=2,
                     alpha=0.8, label=road)

    plt.title('极端降雨下城市路段积水动态演化过程', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('降雨历时 (分钟)', fontsize=12)
    plt.ylabel('积水深度 (cm)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'{IMG_PREFIX}一_积水深度动态演化.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n已保存: {IMG_PREFIX}一_积水深度动态演化.png")

    # 图二：拟合散点图
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
    plt.plot([0, max_val+5], [0, max_val+5], color='#E63946', linestyle='--', linewidth=2, label='1:1 理想拟合线')

    bbox_props = dict(boxstyle="round,pad=0.6", fc="#F1FAEE", ec="#1D3557", lw=1.5)
    textstr = f"模型评估指标:\nRMSE = {rmse:.2f} cm\nMAE = {mae:.2f} cm\nR^2 = {r2:.3f}"
    plt.text(2, max_val-2, textstr, fontsize=12, fontweight='bold', color='#1D3557', bbox=bbox_props)

    plt.title('模型预测值 vs 真实观测值 拟合检验', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('实测积水深度 (cm)', fontsize=12)
    plt.ylabel('模型模拟深度 (cm)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{IMG_PREFIX}二_拟合散点图.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {IMG_PREFIX}二_拟合散点图.png")

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
                     vmin=0, vmax=1, cbar_kws={'label': '通行能力系数'},
                     linewidths=1, linecolor='white', annot_kws={"size": 12, "weight": "bold"})

    plt.title('路网时空通行状态演变热力图', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('推演时间', fontsize=12)
    plt.ylabel('路段编号', fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{IMG_PREFIX}三_通行能力热力图.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {IMG_PREFIX}三_通行能力热力图.png")

    print("\n所有图片已更新完成!")