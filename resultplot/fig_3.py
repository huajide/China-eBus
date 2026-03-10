import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from matplotlib.ticker import PercentFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mosa4CN3_7'))
from mosa4CN3_7.vehicle_type import VehicleTypes


def plot_elbow_method(data):
    # 提取 network_length_km 列作为聚类特征
    X = data[['network_length_km']].values

    # 定义k值范围
    k_range = range(1, 21)
    inertias = []

    # 计算每个k值对应的簇内平方和(inertia)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # 绘制手肘法图表
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.grid(True, alpha=0.3)

    # 添加垂直线指示可能的拐点
    # 通常在曲线上寻找"肘部"位置，这里以k=3为例示意
    plt.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Suggested elbow point')
    plt.legend()

    plt.tight_layout()
    plt.show(block=True)

    return k_range, inertias


def perform_kmeans_and_stats(data, n_clusters=4):
    # 提取 network_length_km 列作为聚类特征
    X = data[['network_length_km']].values

    # 使用k=4进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # 创建新表格，包含城市、网络长度和聚类组别
    result_df = pd.DataFrame({
        'city': data['city'],
        'network_length_km': data['network_length_km'],
        'group': cluster_labels
    })

    # 根据网络长度均值重新编号组别，使group 1-4按从小到大排列
    group_stats = result_df.groupby('group')['network_length_km'].mean().sort_values()
    group_mapping = {old_group: new_group for new_group, old_group in enumerate(group_stats.index, 1)}
    result_df['group'] = result_df['group'].map(group_mapping)

    # 打印各组的统计信息（以表格形式）
    print("各组统计信息:")
    print("=" * 60)

    # 创建统计表格
    stats_data = []
    for i in range(1, n_clusters+1):
        group_data = result_df[result_df['group'] == i]['network_length_km']
        stats_data.append({
            'Group': i,
            'Min (km)': f"{group_data.min():.2f}",
            'Max (km)': f"{group_data.max():.2f}",
            'Mean (km)': f"{group_data.mean():.2f}",
            'Count': len(group_data)
        })

    # 打印表格
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))

    return result_df


def plot_clustered_indicators_bar(clustered_data, output_data, static_data):
    """
    按照3个分类计算6个每公里指标的平均值，并绘制箱型图（符合Nature期刊风格）

    Parameters:
    clustered_data (pd.DataFrame): 聚类后的数据（已合并为3组）
    output_data (pd.DataFrame): 输出数据
    static_data (pd.DataFrame): 静态数据
    """
    # 合并聚类数据和输出数据
    merged_data = pd.merge(output_data, static_data, left_on='city_name', right_on='city')
    merged_data = pd.merge(merged_data, clustered_data[['city', 'group']], on='city')

    # 计算electricity（参考fig_1中的方法）
    merged_data['electricity'] = 0
    for i in range(len(merged_data)):
        city_name = merged_data['city_name'].iloc[i]
        e_price = static_data[static_data['city'] == city_name]['e_price'].iloc[0]
        try:
            vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')

            # 统计三种车型数量
            unique_vehicles = vs_parking_df.drop_duplicates(subset=['v_name'])
            vehicle_counts = unique_vehicles['vehicle_type'].value_counts()
            counts = vehicle_counts.reindex(['large', 'medium', 'small'], fill_value=0)
            large_count = counts['large'] + merged_data['knee_extra_large'].iloc[i]
            medium_count = counts['medium'] + merged_data['knee_extra_medium'].iloc[i]
            small_count = counts['small'] + merged_data['knee_extra_small'].iloc[i]

            trip_cost = merged_data['knee_cost'].iloc[i] * 1000000
            trip_cost -= large_count * VehicleTypes(merged_data.loc[merged_data.index[i], 'large_model'],
                                                    'large').fix_cost
            trip_cost -= medium_count * VehicleTypes(merged_data.loc[merged_data.index[i], 'medium_model'],
                                                     'medium').fix_cost
            trip_cost -= small_count * VehicleTypes(merged_data.loc[merged_data.index[i], 'small_model'],
                                                    'small').fix_cost
            trip_cost -= merged_data['knee_built_cs'].iloc[i] * 600000
            trip_cost -= (merged_data['knee_built_cs'].iloc[i] *
                          (merged_data['knee_slow_per_cs'].iloc[i] * 2000 +
                           merged_data['knee_fast_per_cs'].iloc[i] * 4000))
            merged_data.loc[merged_data.index[i], 'electricity'] = trip_cost / e_price
        except FileNotFoundError:
            # 如果找不到vs_parking文件，使用默认值0
            merged_data.loc[merged_data.index[i], 'electricity'] = 0

    # 计算6个每公里指标
    merged_data['cost_per_km'] = merged_data['knee_cost'] / merged_data['network_length_km']
    merged_data['emission_per_km'] = merged_data['knee_emission'] / merged_data['network_length_km']
    merged_data['cs_per_km'] = merged_data['knee_built_cs'] / merged_data['network_length_km']
    merged_data['charger_per_km'] = ((merged_data['knee_fast_per_cs'] + merged_data['knee_slow_per_cs']) *
                                     merged_data['knee_built_cs'] / merged_data['network_length_km'])
    merged_data['ev_per_km'] = ((merged_data['vehicle_count'] + merged_data['knee_extra_large'] +
                                 merged_data['knee_extra_medium'] + merged_data['knee_extra_small']) /
                                merged_data['network_length_km'])
    merged_data['electricity_per_km'] = merged_data['electricity'] / merged_data['network_length_km']

    avg = merged_data.groupby('group')['cost_per_km'].mean()
    print(
        f"Medium is {((avg[3] - avg[2]) / avg[3] * 100):.1f}% lower than Large and {((avg[1] - avg[2]) / avg[1] * 100):.1f}% lower than Small.")

    # 定义6个指标名称和显示标题
    indicators = ['cost_per_km', 'emission_per_km', 'cs_per_km',
                  'charger_per_km', 'ev_per_km', 'electricity_per_km']

    indicator_titles = {
        'cost_per_km': 'System Costs\nper km\n(M yuan/km)',
        'emission_per_km': 'GHG Emissions\nper km\n(T/km)',
        'cs_per_km': 'Charging Stations\nper km\n(km$^{-1}$)',
        'charger_per_km': 'Chargers\nper km\n(km$^{-1}$)',
        'ev_per_km': 'Electric Buses\nper km\n(km$^{-1}$)',
        'electricity_per_km': 'Electricity\nConsumption\nper km\n(kWh/km)'
    }

    # 设置Nature期刊风格
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # Nature期刊风格设置
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.linewidth': 1.6,
        'xtick.major.width': 1.6,
        'ytick.major.width': 1.6,
        'xtick.minor.width': 1.2,
        'ytick.minor.width': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
    })

    # 创建画布，6个子图，1行6列布局
    fig, axes = plt.subplots(1, 6, figsize=(22, 5))
    axes = axes.flatten()

    # 使用同一色系，从小到大深变浅
    colors = ['#c6dbef', '#6baed6', '#1f77b4']  # 蓝色系，浅到深

    # 为每个指标创建箱型图
    for idx, (indicator, title) in enumerate(indicator_titles.items()):
        ax = axes[idx]

        # 准备每个组的数据
        group_data = []
        group_labels = []
        for group_id in [1, 2, 3]:
            data = merged_data[merged_data['group'] == group_id][indicator].dropna()
            group_data.append(data)
            if group_id == 1:
                group_labels.append('Small')
            elif group_id == 2:
                group_labels.append('Medium')
            else:
                group_labels.append('Large')

        # 绘制箱型图（使用更常见的箱型图样式）
        box_plot = ax.boxplot(group_data, labels=group_labels, patch_artist=True,
                             widths=0.4,  # 箱子更窄一些
                             boxprops=dict(linewidth=1.5),  # 加粗箱子边框
                             medianprops=dict(linewidth=1.5, color='black'),  # 加粗中位数线
                             whiskerprops=dict(linewidth=1),  # 加粗须线
                             capprops=dict(linewidth=1),  # 加粗顶端线
                             flierprops=dict(marker='o', markersize=3, alpha=0.8))  # 更小的异常值点

        # 设置箱型图颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 设置标题和标签
        ax.set_title(title, fontsize=14, pad=10)

        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        # 设置标签大小
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12, rotation=45)

    # 在整个图形底部添加一次"Bus Network"标签
    fig.text(0.5, 0.02, 'Bus Network', ha='center', fontsize=16)

    # 优化布局
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部标签留出空间

    # 保存图像
    plt.savefig('fig_3/clustered_indicators_boxplot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return merged_data


# 在 fig_3.py 文件末尾添加以下代码

def plot_clustered_cities_map(clustered_data, save_path='fig_3/clustered_cities_map.png'):
    """
    绘制中国地图，显示三类城市的分布情况

    Parameters:
    clustered_data (pd.DataFrame): 包含城市和聚类组别的数据
    save_path (str): 保存路径
    """
    # 导入绘图模块
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from china_cities import load_and_prepare_data, _plot_base_map, _plot_inset_map

    # 加载地图数据
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()

    # 创建图形
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)

    # 使用与柱状图相同的颜色方案
    colors = ['#c6dbef', '#6baed6', '#1f77b4']  # 蓝色系，浅到深

    # 绘制基础地图（更浅的灰色背景）
    _plot_base_map(ax_main, cities, province_boundaries, country_boundary,
                   show_province_colors=False, city_color='#f0f0f0')  # 更浅的灰色

    # 为每个组别绘制城市
    for group_id, color in zip([1, 2, 3], colors):
        # 获取当前组别的城市代码
        group_cities = clustered_data[clustered_data['group'] == group_id]
        if not group_cities.empty:
            # 根据城市名称匹配地图数据中的城市
            matched_cities = cities[cities['市'].isin(group_cities['city'])]
            if not matched_cities.empty:
                matched_cities.plot(ax=ax_main, color=color, edgecolor='white',
                                    linewidth=0.5, alpha=1.0)  # 不透明，白色边框

    # 绘制国家边界和省边界
    country_boundary.plot(ax=ax_main, facecolor='none', edgecolor='black', linewidth=1.2)
    province_boundaries.plot(ax=ax_main, facecolor='none', edgecolor='#333333', linewidth=0.8)

    # 设置坐标轴范围
    ax_main.set_ylim(bottom=1.9e6, top=6.12e6)
    ax_main.set_xlim(right=2.5e6)

    # 移除坐标轴标签和刻度
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # 添加图例（只显示分类标签：小、中、大）
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white',
                                     linewidth=0.5, alpha=1.0, label=label)
                       for color, label in zip(colors, ['Small', 'Medium', 'Large'])]
    # 添加图例标题
    legend_title = 'Bus Network Scale'
    legend = ax_main.legend(handles=legend_elements, title=legend_title, loc='lower left',
                            fontsize=12, frameon=True, title_fontsize=13)
    legend.get_title().set_fontweight('bold')

    # 插入小图 - 南海诸岛
    ax_inset = fig.add_axes([0.78, 0.21, 0.11, 0.19])
    _plot_inset_map(ax_inset, cities, province_boundaries, country_boundary, nine_dash_line,
                    show_province_colors=False, city_color='#f0f0f0')  # 更浅的灰色

    # 在小图中也绘制分类城市
    for group_id, color in zip([1, 2, 3], colors):
        group_cities = clustered_data[clustered_data['group'] == group_id]
        if not group_cities.empty:
            matched_cities = cities[cities['市'].isin(group_cities['city'])]
            if not matched_cities.empty:
                matched_cities.plot(ax=ax_inset, color=color, edgecolor='white',
                                    linewidth=0.3, alpha=1.0)  # 不透明，白色边框

    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)
    plt.close()


def plot_scenario_changes_by_group(clustered_data, output_data, static_data,
                                 save_path='fig_3/scenario_changes_by_group.png'):
    """
    绘制各情景下不同聚类组别的中位数变化率折线图

    Parameters:
    clustered_data (pd.DataFrame): 聚类后的数据
    output_data (pd.DataFrame): 输出数据
    static_data (pd.DataFrame): 静态数据
    save_path (str): 保存路径
    """
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
        'BC1.25': '../data/output/mosa/what_if/251026_BC1.25',
        'BC1.5': '../data/output/mosa/what_if/251026_BC1.5',
        'BC1.75': '../data/output/mosa/what_if/251026_BC1.75',
        'BC2': '../data/output/mosa/what_if/251026_BC2',
        'VC0.6': '../data/output/mosa/what_if/251026_VC0.6',
        'VC0.7': '../data/output/mosa/what_if/251026_VC0.7',
        'VC0.8': '../data/output/mosa/what_if/251026_VC0.8',
        'VC0.9': '../data/output/mosa/what_if/251026_VC0.9'
    }

    # 情景分组
    scenario_groups = {
        'FCS': ['baseline', 'FCS2', 'FCS3', 'FCS4', 'FCS5'],
        'EP': ['EP0.5', 'EP0.75', 'baseline', 'EP1.25', 'EP1.5'],
        'BC': ['baseline', 'BC1.25', 'BC1.5', 'BC1.75', 'BC2'],
        'VC': ['VC0.6', 'VC0.7', 'VC0.8', 'VC0.9', 'baseline']
    }

    # 创建city到network_length_km和vehicle_count的映射
    network_length_dict = dict(zip(static_data['city'], static_data['network_length_km']))
    vehicle_count_dict = dict(zip(static_data['city'], static_data['vehicle_count']))


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
                            cs_num = built_num  # 简化处理
                            if len(last_row) > 2 + cs_num:
                                # 提取fast_charger数量
                                fast_chargers = [abs(x) for x in last_row[2:2 + int(cs_num)] if pd.notna(x)]
                                fast_charger_total = sum(fast_chargers)

                                # 提取slow_charger数量
                                slow_chargers = [abs(x) for x in last_row[2 + int(cs_num):2 + 2 * int(cs_num)] if
                                                 pd.notna(x)]
                                slow_charger_total = sum(slow_chargers)
                            else:
                                fast_charger_total = 0
                                slow_charger_total = 0

                            # 获取要增加的大中小车数量
                            additional_vehicles = 0
                            if len(last_row) >= 3:
                                additional_vehicles = sum([abs(x) for x in last_row[-3:] if pd.notna(x)])

                            # 读取archive_objs.csv获取cost和emission
                            obj_file = os.path.join(full_path, 'inf_archive_objs.csv')
                            if os.path.exists(obj_file):
                                obj_data = pd.read_csv(obj_file)
                                # 获取cost和emission
                                cost = obj_data.iloc[-1, 1]  # obj1列
                                emission = obj_data.iloc[-1, 2]  # obj2列

                                # 获取该城市的network_length_km和vehicle_count
                                network_length = network_length_dict.get(city_name, 1)
                                static_vehicle_count = vehicle_count_dict.get(city_name, 0)
                                actual_vehicle_count = static_vehicle_count + additional_vehicles

                                city_stats.append({
                                    'city': city_name,
                                    'cost': -cost / network_length,
                                    'emission': -emission / network_length,
                                    'built_num': built_num / network_length,
                                    'fast_charger': fast_charger_total / network_length,
                                    'slow_charger': slow_charger_total / network_length,
                                    'vehicle_count': actual_vehicle_count / network_length,
                                    'network_length_km': network_length
                                })
                    except Exception as e:
                        print(f"处理城市 {city_name} 时出错: {e}")

        return pd.DataFrame(city_stats)


    # 加载各情景数据
    scenario_data = {}
    for scenario, path in scenarios.items():
        scenario_data[scenario] = load_scenario_data(path)

    # 确保所有情景数据中的城市顺序一致
    common_cities = None
    for scenario in scenario_data:
        if len(scenario_data[scenario]) > 0:
            if common_cities is None:
                common_cities = set(scenario_data[scenario]['city'])
            else:
                common_cities = common_cities.intersection(set(scenario_data[scenario]['city']))

    if common_cities is None:
        print("没有找到共同的城市数据")
        return None

    common_cities = list(common_cities)
    print(f"找到 {len(common_cities)} 个共同城市")

    # 重新整理数据，确保城市顺序一致
    aligned_scenario_data = {}
    for scenario in scenario_data:
        if len(scenario_data[scenario]) > 0:
            # 按照common_cities的顺序重新排列数据
            aligned_data = []
            for city in common_cities:
                city_data = scenario_data[scenario][scenario_data[scenario]['city'] == city]
                if len(city_data) > 0:
                    aligned_data.append(city_data.iloc[0])
            aligned_scenario_data[scenario] = pd.DataFrame(aligned_data)

    # 计算每个城市相对于baseline的变化率
    city_relative_data = {}
    baseline_city_data = aligned_scenario_data.get('baseline', pd.DataFrame())

    if baseline_city_data.empty:
        print("Baseline数据为空")
        return None

    for scenario in aligned_scenario_data.keys():
        city_relative_data[scenario] = {}

        # 计算相对变化率
        for metric in ['cost', 'emission', 'built_num', 'fast_charger', 'slow_charger', 'vehicle_count']:
            city_relative_data[scenario][metric] = []
            for i in range(len(aligned_scenario_data[scenario])):
                baseline_val = baseline_city_data.iloc[i][metric]
                current_val = aligned_scenario_data[scenario].iloc[i][metric]
                if baseline_val != 0:
                    rel_change = (current_val - baseline_val) / baseline_val * 100
                    city_relative_data[scenario][metric].append(rel_change)
                else:
                    city_relative_data[scenario][metric].append(0.0)

    # 将聚类信息合并到baseline数据中，确保只包含共同城市
    baseline_with_common_cities = baseline_city_data[baseline_city_data['city'].isin(common_cities)]
    baseline_with_group = pd.merge(baseline_with_common_cities, clustered_data[['city', 'group']], on='city')

    # 创建城市到索引的映射，方便快速查找
    city_to_index = {city: idx for idx, city in enumerate(common_cities)}

    # 计算每组在各情景下的中位数变化率（修改部分）
    group_median_changes = {}

    # 为每个情景分组计算数据
    for group_name, scenarios_list in scenario_groups.items():
        group_median_changes[group_name] = {}

        for metric in ['cost', 'emission', 'built_num', 'fast_charger', 'slow_charger', 'vehicle_count']:
            group_median_changes[group_name][metric] = {}

            # 对于每个组（1, 2, 3）
            for group_id in [1, 2, 3]:
                group_median_changes[group_name][metric][group_id] = []

                # 获取该组的城市
                group_cities = set(baseline_with_group[baseline_with_group['group'] == group_id]['city'].tolist())

                # 对于每个情景
                for scenario in scenarios_list:
                    # 获取这些城市在该情景下的变化率
                    changes = []
                    for city in group_cities:
                        if city in city_to_index and scenario in city_relative_data:
                            city_idx = city_to_index[city]
                            if city_idx < len(city_relative_data[scenario][metric]):
                                changes.append(city_relative_data[scenario][metric][city_idx])

                    # 计算中位数变化率（修改部分）
                    median_change = np.median(changes) if changes else 0
                    group_median_changes[group_name][metric][group_id].append(median_change)

    # 设置绘图参数
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # Nature期刊风格设置
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
    })

    # 创建图表 (4行6列)
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    # 指标定义
    metrics = ['cost', 'emission', 'built_num', 'fast_charger', 'slow_charger', 'vehicle_count']
    metric_names = ['Cost', 'Emission', 'Charging Stations', 'Fast Chargers', 'Slow Chargers', 'Electric Vehicles']
    units = ['M yuan/(year·km)', 'T/(year·km)', r'km$^{-1}$', r'km$^{-1}$', r'km$^{-1}$', r'km$^{-1}$']

    # 使用与之前相同的颜色方案
    group_colors = ['#c6dbef', '#6baed6', '#1f77b4']  # 蓝色系，浅到深

    # 行标签
    row_labels = ['Fast charging speed', 'Electricity price', 'Battery capacity', 'Vehicle cost']

    # 为每个情景分组和指标绘制图表
    for row, (group_name, scenarios_list) in enumerate(scenario_groups.items()):
        x_positions = list(range(len(scenarios_list)))

        # 不同的情景分组有不同的x轴标签
        if group_name == 'FCS':
            x_labels = ['1x\n(baseline)', '2x', '3x', '4x', '5x']
        elif group_name == 'EP':
            x_labels = ['0.5x', '0.75x', '1x\n(baseline)', '1.25x', '1.5x']
        elif group_name == 'BC':
            x_labels = ['1x\n(baseline)', '1.25x', '1.5x', '1.75x', '2x']
        elif group_name == 'VC':
            x_labels = ['0.6x', '0.7x', '0.8x', '0.9x', '1x\n(baseline)']

        for col, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
            ax = axes[row, col]

            # 为每个组绘制折线图（使用中位数数据）
            for group_id, color in zip([1, 2, 3], group_colors):
                y_values = group_median_changes[group_name][metric][group_id]
                ax.plot(x_positions, y_values, marker='o', linewidth=1.5, markersize=4,
                        color=color, label=f'Group {group_id}')

            # 设置标题和标签
            ax.set_title(f'{metric_name}', fontsize=14, pad=10)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.yaxis.set_major_formatter(PercentFormatter())
            ax.grid(True, linestyle=':', alpha=0.6, linewidth=1)

            # 示例：为每列设置固定的y轴范围
            y_ranges = {
                'cost': (-30, 30),
                'emission': (-5, 5),
                'built_num': (-15, 15),
                'fast_charger': (-100, 10),
                'slow_charger': (-70, 10),
                'vehicle_count': (-3, 1)
            }

            if metric in y_ranges:
                ax.set_ylim(y_ranges[metric])

            # 设置边框
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # 在第一列添加行标签
            if col == 0:
                ax.text(-0.3, 0.5, row_labels[row], transform=ax.transAxes,
                        fontsize=13, fontweight='bold', rotation=90,
                        verticalalignment='center', horizontalalignment='center')

            # 在第一行第一列添加图例
            if row == 0 and col == 0:
                # 创建自定义图例元素，同时显示线和点
                legend_elements = [plt.Line2D([0], [0], marker='o', color=color, markerfacecolor=color,
                                              markersize=8, label=label, linewidth=2, markeredgecolor='w',
                                              markeredgewidth=0.5)
                                   for color, label in zip(group_colors, ['Small', 'Medium', 'Large'])]

                # 添加图例标题
                legend = ax.legend(handles=legend_elements, title='Bus Network Scale', loc='upper left',
                                   bbox_to_anchor=(-0.5, 1.5), ncol=3, frameon=False, fontsize=12)
                legend.get_title().set_fontweight('bold')
                legend.get_title().set_fontsize(12)

    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show(block=True)
    plt.close()

    return group_median_changes



if __name__ == '__main__':
    static_data = pd.read_csv("../data/224city_indicators.csv")
    output_data = pd.read_csv("../data/224cities_output.csv")

    '''调用函数绘制手肘'''
    # k_values, inertia_values = plot_elbow_method(static_data)

    '''按k=4调用函数执行聚类和统计'''
    clustered_data = perform_kmeans_and_stats(static_data, 4)

    # '''将最大的两个类别（group 4 和 group 3）合并为一个新的类别'''
    # 首先将 group 4 的城市归类到 group 3
    mask_group4 = clustered_data['group'] == 4
    clustered_data.loc[mask_group4, 'group'] = 3
    #
    # '''1. 绘制聚类城市地图'''
    # # plot_clustered_cities_map(clustered_data)
    #
    # '''2. 计算每公里指标并绘制柱状图'''
    # # 调用函数绘制柱状图
    group_averages = plot_clustered_indicators_bar(clustered_data, output_data, static_data)
    #
    '''3. 绘制各情景下不同聚类组别的平均变化率'''
    # plot_scenario_changes_by_group(clustered_data, output_data, static_data)
