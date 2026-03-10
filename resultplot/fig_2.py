import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import CRS
from matplotlib.ticker import PercentFormatter

# 添加上级目录到Python路径，以便导入test4plot2和analysis模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mosa4CN3_7', 'TestingResults'))

# 导入所需的模块
from test4plot2 import clearing
from analysis import find_knee, basic_stat


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})


# 定义情景和对应的文件夹
scenarios = {
    'baseline': '../data/output/mosa/251026',
    'EP0.5': '../data/output/mosa/what_if/251026_EP0.5',
    'EP0.75': '../data/output/mosa/what_if/251026_EP0.75',
    'EP1.25': '../data/output/mosa/what_if/251026_EP1.25',
    'EP1.5': '../data/output/mosa/what_if/251026_EP1.5',
    'FCS2': '../data/output/mosa/what_if/251026_FCS2',
    'FCS3': '../data/output/mosa/what_if/251026_FCS3',
    'FCS4': '../data/output/mosa/what_if/251026_FCS4',
    'FCS5': '../data/output/mosa/what_if/251026_FCS5',
    'BC1.25':  '../data/output/mosa/what_if/251026_BC1.25',
    'BC1.5': '../data/output/mosa/what_if/251026_BC1.5',
    'BC1.75': '../data/output/mosa/what_if/251026_BC1.75',
    'BC2': '../data/output/mosa/what_if/251026_BC2',
    'VC0.6':  '../data/output/mosa/what_if/251026_VC0.6',
    'VC0.7': '../data/output/mosa/what_if/251026_VC0.7',
    'VC0.8': '../data/output/mosa/what_if/251026_VC0.8',
    'VC0.9': '../data/output/mosa/what_if/251026_VC0.9'
}

# 情景分组
fcs_scenarios = ['baseline', 'FCS2', 'FCS3', 'FCS4', 'FCS5']
ep_scenarios = ['EP0.5', 'EP0.75', 'baseline', 'EP1.25', 'EP1.5']
bc_scenarios = ['baseline', 'BC1.25', 'BC1.5', 'BC1.75', 'BC2']
vc_scenarios = ['VC0.6','VC0.7','VC0.8','VC0.9','baseline']

def extract_city_name(folder_name, root_name):
    """从文件夹名中提取城市名"""
    if folder_name.endswith(f'_{root_name}'):
        return folder_name[:-len(f'_{root_name}')]
    return None


# 在读取城市指标数据后添加以下代码

# 读取城市指标数据
static_data = pd.read_csv(r'../data/224city_indicators.csv')

# 创建city到network_length_km和vehicle_count的映射
network_length_dict = dict(zip(static_data['city'], static_data['network_length_km']))
vehicle_count_dict = dict(zip(static_data['city'], static_data['vehicle_count']))


# 修改load_scenario_data函数中的相关部分
def load_scenario_data(scenario_path):
    """加载指定情景下的数据"""
    city_stats = []

    if not os.path.exists(scenario_path):
        print(f"路径不存在: {scenario_path}")
        return pd.DataFrame()

    for folder in os.listdir(scenario_path):
        full_path = os.path.join(scenario_path, folder)
        if os.path.isdir(full_path):
            city_name = os.path.basename(full_path)
            if city_name:
                try:
                    # 读取archive_cvs.csv获取built_num、fast_charger、slow_charger
                    cv_file = os.path.join(full_path, 'inf_archive_cvs.csv')
                    if os.path.exists(cv_file):
                        cv_data = pd.read_csv(cv_file)
                        # 获取最后一个solution的数据
                        last_row = cv_data.iloc[-1]

                        # 获取built_num（假设第二列是built_num）
                        built_num = abs(last_row[1])

                        # 获取fast_charger和slow_charger数量
                        # 假设fast_charger在第3列到第(n+2)列，slow_charger在第(n+3)列到第(2n+2)列
                        # 其中n是充电站数量
                        cs_num = built_num  # 简化处理，实际可能需要从其他地方获取
                        if len(last_row) > 2 + cs_num:
                            # 提取fast_charger数量（取负值的绝对值）
                            fast_chargers = [abs(x) for x in last_row[2:2 + int(cs_num)] if pd.notna(x)]
                            fast_charger_total = sum(fast_chargers)

                            # 提取slow_charger数量（取负值的绝对值）
                            slow_chargers = [abs(x) for x in last_row[2 + int(cs_num):2 + 2 * int(cs_num)] if
                                             pd.notna(x)]
                            slow_charger_total = sum(slow_chargers)
                        else:
                            fast_charger_total = 0
                            slow_charger_total = 0

                        # 获取要增加的大中小车数量（最后三列的绝对值）
                        additional_vehicles = 0
                        if len(last_row) >= 3:
                            # 取最后三列的绝对值作为要增加的车辆数
                            additional_vehicles = sum([abs(x) for x in last_row[-3:] if pd.notna(x)])

                        # 读取archive_objs.csv获取cost和emission
                        obj_file = os.path.join(full_path, 'inf_archive_objs.csv')
                        if os.path.exists(obj_file):
                            obj_data = pd.read_csv(obj_file)
                            # 获取cost和emission（假设obj1是cost，obj2是emission）
                            cost = obj_data.iloc[-1, 1]  # obj1列
                            emission = obj_data.iloc[-1, 2]  # obj2列

                            # 获取该城市的network_length_km和vehicle_count
                            network_length = network_length_dict.get(city_name, 1)  # 如果找不到，默认为1
                            static_vehicle_count = vehicle_count_dict.get(city_name, 0)  # 静态车辆数
                            # 实际车辆数 = 静态车辆数 + 增加的车辆数
                            actual_vehicle_count = static_vehicle_count + additional_vehicles

                            city_stats.append({
                                'city': city_name,
                                'cost': -cost / network_length,  # 转换为正值并标准化
                                'emission': -emission / network_length,  # 转换为正值并标准化
                                'built_num': built_num / network_length,  # 标准化
                                'fast_charger': fast_charger_total / network_length,  # 快充桩数标准化
                                'slow_charger': slow_charger_total / network_length,  # 慢充桩数标准化
                                'vehicle_count': actual_vehicle_count / network_length,  # 实际车辆数标准化
                                'network_length_km': network_length
                            })
                except Exception as e:
                    print(f"处理城市 {city_name} 时出错: {e}")

    return pd.DataFrame(city_stats)


# 加载各情景数据
scenario_data = {}
for scenario, path in scenarios.items():
    scenario_data[scenario] = load_scenario_data(path)

# 创建图表
fig, axes = plt.subplots(4, 6, figsize=(15, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# 更新指标列表和单位
metrics = ['cost', 'emission', 'built_num', 'fast_charger', 'slow_charger', 'vehicle_count']
metric_names = ['Cost', 'Emission', 'Charging Stations', 'Fast Chargers', 'Slow Chargers', 'Electric Vehicles']
units = ['M yuan/(year·km)', 'T/(year·km)', r'km$^{-1}$', r'km$^{-1}$', r'km$^{-1}$', r'km$^{-1}$']

# 计算每个城市相对于baseline的变化率
city_relative_data = {}

# 获取baseline数据作为基准
baseline_city_data = scenario_data['baseline']

# 对每个情景计算每个城市相对变化率
for scenario in scenario_data.keys():
    city_relative_data[scenario] = {}

    if scenario == 'baseline':
        # baseline情景的变化率为0%
        for metric in metrics:
            city_relative_data[scenario][metric] = [0.0] * len(baseline_city_data)
    else:
        # 计算相对变化率: (当前值 - 基准值) / 基准值 * 100
        for metric in metrics:
            city_relative_data[scenario][metric] = []
            for i in range(len(scenario_data[scenario])):
                baseline_val = baseline_city_data.iloc[i][metric]
                current_val = scenario_data[scenario].iloc[i][metric]
                if baseline_val != 0:
                    rel_change = (current_val - baseline_val) / baseline_val * 100
                    city_relative_data[scenario][metric].append(rel_change)
                else:
                    city_relative_data[scenario][metric].append(0.0)

fcs5_cost_drop_pct = (pd.Series(city_relative_data['FCS5']['cost']) < -10).mean() * 100
bc2_emission_drop_pct = (pd.Series(city_relative_data['BC2']['emission']) < -10).mean() * 100

print(f"FCS 5x: {fcs5_cost_drop_pct:.1f}% of cities have cost decreases >10%.")
print(f"BC 2x: {bc2_emission_drop_pct:.1f}% of cities have emission decreases >10%.")


# 定义颜色方案（符合Nature风格的配色）
colors = {
    'fcs': '#1f77b4',  # 蓝色
    'ep': '#ff7f0e',  # 橙色
    'bc': '#2ca02c',  # 绿色
    'vc': '#d62728'   # 红色
}

# 定义行标签
row_labels_with_units = [
    'Fast charging speed',
    'Electricity price',
    'Battery capacity',
    'Vehicle cost'
]

# 第一行：FCS情景 (修改绘图部分)
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[0, j]

    # 准备FCS情景的数据用于箱型图
    fcs_box_data = []
    fcs_positions = []
    for i, scenario in enumerate(fcs_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            fcs_box_data.append(city_relative_data[scenario][metric])
            fcs_positions.append(i + 1)

    # 绘制FCS情景箱型图
    if fcs_box_data:
        bp = ax.boxplot(fcs_box_data, positions=fcs_positions, widths=0.6,
                        patch_artist=True, notch=False, showfliers=False,
                        boxprops=dict(facecolor=colors['fcs'], alpha=0.7, linewidth=1),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))

    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(fcs_positions)
    ax.set_xticklabels(['1x\n(baseline)', '2x', '3x', '4x', '5x'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 在最左侧添加行标签和单位
    if j == 0:
        # 第一行：标签名（加粗）
        ax.text(-0.5, 0.5, row_labels_with_units[0].split('\n')[0],
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                rotation=90, verticalalignment='center',
                horizontalalignment='center')

# 第二行：EP情景
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[1, j]

    # 准备EP情景的数据用于箱型图
    ep_box_data = []
    ep_positions = []
    for i, scenario in enumerate(ep_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            ep_box_data.append(city_relative_data[scenario][metric])
            ep_positions.append(i + 1)

    # 绘制EP情景箱型图
    if ep_box_data:
        bp = ax.boxplot(ep_box_data, positions=ep_positions, widths=0.6,
                        patch_artist=True, notch=False, showfliers=False,
                        boxprops=dict(facecolor=colors['ep'], alpha=0.7, linewidth=1),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))

    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(ep_positions)
    ax.set_xticklabels(['0.5x', '0.75x', '1x\n(baseline)', '1.25x', '1.5x'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 在最左侧添加行标签和单位
    if j == 0:
        # 第一行：标签名（加粗）
        ax.text(-0.5, 0.5, row_labels_with_units[1].split('\n')[0],
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                rotation=90, verticalalignment='center',
                horizontalalignment='center')


# 第三行：BC情景
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[2, j]

    # 准备BC情景的数据用于箱型图
    bc_box_data = []
    bc_positions = []
    for i, scenario in enumerate(bc_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            bc_box_data.append(city_relative_data[scenario][metric])
            bc_positions.append(i + 1)

    # 绘制BC情景箱型图
    if bc_box_data:
        bp = ax.boxplot(bc_box_data, positions=bc_positions, widths=0.6,
                        patch_artist=True, notch=False, showfliers=False,
                        boxprops=dict(facecolor=colors['bc'], alpha=0.7, linewidth=1),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))

    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(bc_positions)
    ax.set_xticklabels(['1x\n(baseline)', '1.25x', '1.5x', '1.75x', '2x'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 在最左侧添加行标签和单位
    if j == 0:
        # 第一行：标签名（加粗）
        ax.text(-0.5, 0.5, row_labels_with_units[2].split('\n')[0],
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                rotation=90, verticalalignment='center',
                horizontalalignment='center')

# 第四行：VC情景
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[3, j]

    # 准备VC情景的数据用于箱型图
    vc_box_data = []
    vc_positions = []
    for i, scenario in enumerate(vc_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            vc_box_data.append(city_relative_data[scenario][metric])
            vc_positions.append(i + 1)

    # 绘制VC情景箱型图
    if vc_box_data:
        bp = ax.boxplot(vc_box_data, positions=vc_positions, widths=0.6,
                        patch_artist=True, notch=False, showfliers=False,
                        boxprops=dict(facecolor=colors['vc'], alpha=0.7, linewidth=1),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))

    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(vc_positions)
    ax.set_xticklabels(['0.6x', '0.7x', '0.8x', '0.9x', '1x\n(baseline)'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 在最左侧添加行标签和单位
    if j == 0:
        # 第一行：标签名（加粗）
        ax.text(-0.5, 0.5, row_labels_with_units[3].split('\n')[0],
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                rotation=90, verticalalignment='center',
                horizontalalignment='center')


plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
# plt.savefig('fig_2&3/scenario_comparison.png', dpi=300, bbox_inches='tight',
#             facecolor='white', edgecolor='none')
plt.close()
