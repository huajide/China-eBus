import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats
import statsmodels.api as sm


def plot_budget_vs_ind_electricity(city_data, output_data, static_data, output_dir='fig_1'):
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mosa4CN3_6'))
    from mosa4CN3_7.vehicle_type import VehicleTypes
    """
    绘制budget与ind_electricity的箱型图，采用Nature期刊风格

    Parameters:
    city_data (pd.DataFrame): 包含城市数据的DataFrame (18cities.csv)
    output_data (pd.DataFrame): 包含输出数据的DataFrame (18cities_output.csv)
    static_data (pd.DataFrame): 包含静态数据的DataFrame
    output_dir (str): 图片保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 需要按车型才知道耗电量
    output_data['electricity'] = 0
    for i in range(len(output_data)):
        city_name = output_data['city_name'][i]
        e_price = static_data[static_data['city'] == city_name]['e_price'].iloc[0]
        vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')

        # 统计三种车型数量
        unique_vehicles = vs_parking_df.drop_duplicates(subset=['v_name'])
        vehicle_counts = unique_vehicles['vehicle_type'].value_counts()
        counts = vehicle_counts.reindex(['large', 'medium', 'small'], fill_value=0)
        large_count = counts['large']+output_data['knee_extra_large'][i]
        medium_count = counts['medium']+output_data['knee_extra_medium'][i]
        small_count = counts['small']+output_data['knee_extra_small'][i]

        trip_cost = output_data['knee_cost'][i] * 1000000
        trip_cost -= large_count * VehicleTypes(output_data.loc[i,f'large_model'], 'large').fix_cost
        trip_cost -= medium_count * VehicleTypes(output_data.loc[i,f'medium_model'], 'medium').fix_cost
        trip_cost -= small_count * VehicleTypes(output_data.loc[i,f'small_model'], 'small').fix_cost
        trip_cost -= output_data['knee_built_cs'][i] * 600000
        trip_cost -= (output_data['knee_built_cs'][i] *
                      (output_data['knee_slow_per_cs'][i]*2000+output_data['knee_fast_per_cs'][i]*4000))
        output_data.loc[i,'electricity'] = trip_cost / e_price

    # 合并数据
    merged_df = pd.merge(output_data, city_data, left_on='city_name', right_on='city')

    # 计算指标：公交电气化成本占预算的百分比和电力消耗占比
    merged_df['cost_to_budget_ratio'] = (merged_df['knee_cost'] / merged_df['budget']) / 1e3 * 100  # 转换为百分比
    merged_df['electricity_ratio'] = merged_df['electricity'] / (merged_df['ind_electricity'] * 1e9) * 100  # 转换为百分比

    # 定义2个指标
    target_cols = ['cost_to_budget_ratio', 'electricity_ratio']

    # 指标标题映射
    titles = {
        'cost_to_budget_ratio': 'Cost-to-Budget\nRatio (%)',
        'electricity_ratio': 'Electricity\nRatio (%)'
    }

    # 选择需要的列
    analysis_df = merged_df[['city_name'] + target_cols].copy()

    print("=== Cost-to-Budget Ratio Statistics ===")
    print(f"Mean: {merged_df['cost_to_budget_ratio'].mean():.4f}")
    print(f"Min: {merged_df['cost_to_budget_ratio'].min():.4f}")
    print(f"Max: {merged_df['cost_to_budget_ratio'].max():.4f}")

    print("\n=== Electricity Ratio Statistics ===")
    print(f"Mean: {merged_df['electricity_ratio'].mean():.4f}")
    print(f"Min: {merged_df['electricity_ratio'].min():.4f}")
    print(f"Max: {merged_df['electricity_ratio'].max():.4f}")

    # 转换为长格式
    melted_df = pd.melt(analysis_df,
                        id_vars='city_name',
                        value_vars=target_cols,
                        var_name='Indicator',
                        value_name='Value')

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

    # 创建画布，2个子图，调整为1行2列
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()

    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    # 科研配色方案
    colors = ['#377eb8', '#ff7f00']

    # 创建图例句柄和标签
    legend_handles = []
    legend_labels = ['Median', 'Mean']

    # 绘制每个指标
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        # 获取当前指标数据
        current_data = melted_df[melted_df['Indicator'] == col]

        # 排序数据以便优化标注
        current_data = current_data.sort_values('Value')

        # 绘制箱型图
        box = sns.boxplot(
            y='Value',
            data=current_data,
            ax=ax,
            orient='v',
            width=0.4,
            fliersize=4,
            showmeans=False,
            color=colors[i],
            boxprops=dict(linewidth=2, edgecolor='black', facecolor=colors[i] + '80'),
            medianprops=dict(color='black', linewidth=2.4, linestyle='--'),
            whiskerprops=dict(color='black', linewidth=2),
            capprops=dict(color='black', linewidth=2)
        )

        # 计算并绘制平均值线（灰色虚线，只在箱体内部）
        mean_value = current_data['Value'].mean()

        # 平均值线只在箱子内部显示
        mean_line = ax.hlines(y=mean_value, xmin=-0.2, xmax=0.2, color='gray', linestyle='--', linewidth=2.4)

        # 添加一条隐藏的线，xmin和xmax为0.1级别，但不显示在legend中
        hidden_line = ax.hlines(y=mean_value * 1.1, xmin=-0.4, xmax=0.4, color='none', linewidth=0)

        # 只在第一个子图添加图例句柄
        if i == 0:
            # 创建中位数线条的代理艺术家
            from matplotlib.lines import Line2D
            median_line = Line2D([0], [0], color='black', linewidth=2.4, linestyle='--')
            mean_line_legend = Line2D([0], [0], color='gray', linewidth=2.4, linestyle='--')
            legend_handles = [median_line, mean_line_legend]

        # 设置标题和标签
        ax.set_title(titles[col], fontsize=20, pad=20)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Value', fontsize=18)

        # 优化布局
        ax.tick_params(axis='both', which='major', labelsize=16)

        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

    # 在最后一个子图旁边添加图例
    axes[-1].legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)

    # 保存图像
    plt.savefig(f'{output_dir}/budget_electricity_boxplot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()



def plot_route_count_vs_population_gdp(indicator_data, city_data, output_dir='fig_1'):
    """
    绘制route_count与人口、GDP的散点图，包含线性回归线和置信区间，采用Nature期刊风格

    Parameters:
    indicator_data (pd.DataFrame): 包含指标数据的DataFrame
    city_data (pd.DataFrame): 包含城市数据的DataFrame
    output_dir (str): 图片保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 合并数据
    merged_df = pd.merge(indicator_data, city_data, left_on='city', right_on='city')

    # 准备数据
    # 人口单位是万，GDP单位是trillion yuan
    population = merged_df['district_pop']  # 万人
    gdp = merged_df['district_gdp'] * 1000  # 转换为 billion yuan
    route_count = merged_df['route_count']

    # 设置Nature期刊风格（增大字体以适应多子图布局）
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # Nature期刊风格设置（字体增大2倍）
    plt.rcParams.update({
        'font.size': 18,  # 从9增大到18
        'axes.titlesize': 20,  # 从10增大到20
        'axes.labelsize': 18,  # 从9增大到18
        'xtick.labelsize': 16,  # 从8增大到16
        'ytick.labelsize': 16,  # 从8增大到16
        'legend.fontsize': 16,  # 从8增大到16
        'axes.linewidth': 1.6,  # 从0.8增大到1.6
        'xtick.major.width': 1.6,  # 从0.8增大到1.6
        'ytick.major.width': 1.6,  # 从0.8增大到1.6
        'xtick.minor.width': 1.2,  # 从0.6增大到1.2
        'ytick.minor.width': 1.2,  # 从0.6增大到1.2
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 8,  # 从4增大到8
        'ytick.major.size': 8,  # 从4增大到8
        'xtick.minor.size': 4,  # 从2增大到4
        'ytick.minor.size': 4,  # 从2增大到4
    })

    # 创建画布，1行2列
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 增大画布尺寸
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # 颜色设置
    scatter_color = '#377eb8'
    regression_color = '#e41a1c'

    # 1. route_count vs population
    ax1 = axes[0]

    # 绘制散点图
    ax1.scatter(population, route_count, color=scatter_color, alpha=0.7, s=60, edgecolors='black', linewidth=0.6)  # 增大点尺寸

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(population, route_count)

    # 绘制回归线
    pop_range = np.linspace(population.min(), population.max(), 100)
    regression_line = slope * pop_range + intercept
    ax1.plot(pop_range, regression_line, color=regression_color, linewidth=3, linestyle='-')  # 增大线宽

    # 计算置信区间
    X_with_const = sm.add_constant(population)
    model = sm.OLS(route_count, X_with_const).fit()
    predictions = model.get_prediction(sm.add_constant(pop_range))
    conf_int = predictions.conf_int()

    # 绘制置信区间
    ax1.fill_between(pop_range, conf_int[:, 0], conf_int[:, 1],
                     color=regression_color, alpha=0.2, linewidth=0)

    # 添加R²和p值
    ax1.text(0.05, 0.95, f'R² = {r_value ** 2:.2f}\np = {p_value:.3f}',
             transform=ax1.transAxes, fontsize=14, verticalalignment='top',  # 增大字体
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # 设置标签
    ax1.set_xlabel('Population (10⁴ people)', fontsize=18)
    ax1.set_ylabel('Route Count', fontsize=18)
    ax1.set_title('Route Count vs Population', fontsize=20, pad=20)

    # 优化布局
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, linestyle=':', alpha=0.5, linewidth=1)

    # 设置边框
    for spine in ax1.spines.values():
        spine.set_linewidth(1.6)

    # 2. route_count vs GDP
    ax2 = axes[1]

    # 绘制散点图
    ax2.scatter(gdp, route_count, color=scatter_color, alpha=0.7, s=60, edgecolors='black', linewidth=0.6)  # 增大点尺寸

    # 线性回归
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(gdp, route_count)

    # 绘制回归线
    gdp_range = np.linspace(gdp.min(), gdp.max(), 100)
    regression_line2 = slope2 * gdp_range + intercept2
    ax2.plot(gdp_range, regression_line2, color=regression_color, linewidth=3, linestyle='-')  # 增大线宽

    # 计算置信区间
    X_with_const2 = sm.add_constant(gdp)
    model2 = sm.OLS(route_count, X_with_const2).fit()
    predictions2 = model2.get_prediction(sm.add_constant(gdp_range))
    conf_int2 = predictions2.conf_int()

    # 绘制置信区间
    ax2.fill_between(gdp_range, conf_int2[:, 0], conf_int2[:, 1],
                     color=regression_color, alpha=0.2, linewidth=0)

    # 添加R²和p值
    ax2.text(0.05, 0.95, f'R² = {r_value2 ** 2:.2f}\np = {p_value2:.3f}',
             transform=ax2.transAxes, fontsize=14, verticalalignment='top',  # 增大字体
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # 设置标签
    ax2.set_xlabel('GDP (billion yuan)', fontsize=18)
    ax2.set_ylabel('Route Count', fontsize=18)
    ax2.set_title('Route Count vs GDP', fontsize=20, pad=20)

    # 优化布局
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(True, linestyle=':', alpha=0.5, linewidth=1)

    # 设置边框
    for spine in ax2.spines.values():
        spine.set_linewidth(1.6)

    # 保存图像
    plt.savefig(f'{output_dir}/route_count_population_gdp_scatter.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def plot_operation_indicators_boxplot(indicator_data, output_dir='fig_1'):
    """
    绘制运营指标的箱型图，将3个指标分别放在子图中，每个子图有独立的y轴，采用Nature期刊风格

    Parameters:
    indicator_data (pd.DataFrame): 包含指标数据的DataFrame
    output_dir (str): 图片保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义3个运营指标
    target_cols = ['route_count', 'route_avg_distance', 'route_avg_trip']

    # 指标标题映射
    titles = {
        'route_count': 'Route Count',
        'route_avg_distance': 'Average Route\nDistance (km)',
        'route_avg_trip': 'Average Trips\nper Route (per day)'
    }

    # 创建分析数据集
    analysis_df = indicator_data[['city', 'route_count', 'route_avg_distance', 'trip_number']].copy()

    # 计算 route_avg_trip = trip_number / route_count
    analysis_df['route_avg_trip'] = analysis_df['trip_number'] / analysis_df['route_count']

    # 选择需要的列
    analysis_df = analysis_df[['city'] + target_cols]

    # 转换为长格式
    melted_df = pd.melt(analysis_df,
                        id_vars='city',
                        value_vars=target_cols,
                        var_name='Indicator',
                        value_name='Value')

    # 设置Nature期刊风格（增大字体以适应多子图布局）
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # Nature期刊风格设置（字体增大2倍）
    plt.rcParams.update({
        'font.size': 18,  # 从9增大到18
        'axes.titlesize': 20,  # 从10增大到20
        'axes.labelsize': 18,  # 从9增大到18
        'xtick.labelsize': 16,  # 从8增大到16
        'ytick.labelsize': 16,  # 从8增大到16
        'legend.fontsize': 16,  # 从8增大到16
        'axes.linewidth': 1.6,  # 从0.8增大到1.6
        'xtick.major.width': 1.6,  # 从0.8增大到1.6
        'ytick.major.width': 1.6,  # 从0.8增大到1.6
        'xtick.minor.width': 1.2,  # 从0.6增大到1.2
        'ytick.minor.width': 1.2,  # 从0.6增大到1.2
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 8,  # 从4增大到8
        'ytick.major.size': 8,  # 从4增大到8
        'xtick.minor.size': 4,  # 从2增大到4
        'ytick.minor.size': 4,  # 从2增大到4
    })

    # 创建画布，3个子图，调整为1行3列
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 增大画布尺寸
    axes = axes.flatten()

    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    # 科研配色方案（选取前3个颜色）
    colors = ['#377eb8', '#ff7f00', '#4daf4a']

    # 创建图例句柄和标签
    legend_handles = []
    legend_labels = ['Median', 'Mean']

    # 绘制每个指标
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        # 获取当前指标数据
        current_data = melted_df[melted_df['Indicator'] == col]

        # 排序数据以便优化标注
        current_data = current_data.sort_values('Value')

        # 绘制箱型图 - 使用更窄的箱子
        box = sns.boxplot(
            y='Value',
            data=current_data,
            ax=ax,
            orient='v',
            width=0.2,  # 从0.1增大到0.2
            fliersize=4,  # 从2增大到4
            showmeans=False,
            color=colors[i],
            boxprops=dict(linewidth=2, edgecolor='black', facecolor=colors[i] + '80'),  # 增大线宽
            medianprops=dict(color='black', linewidth=2.4, linestyle='--'),  # 增大线宽
            whiskerprops=dict(color='black', linewidth=2),  # 增大线宽
            capprops=dict(color='black', linewidth=2)  # 增大线宽
        )

        # 计算并绘制平均值线（灰色虚线，只在箱体内部）
        mean_value = current_data['Value'].mean()
        q1 = current_data['Value'].quantile(0.25)
        q3 = current_data['Value'].quantile(0.75)

        # 平均值线只在箱子内部显示（在四分位数之间）
        mean_line = ax.hlines(y=mean_value, xmin=-0.1, xmax=0.1, color='gray', linestyle='--', linewidth=2.4)  # 增大线宽

        # 添加一条隐藏的线，xmin和xmax为0.1级别，但不显示在legend中
        hidden_line = ax.hlines(y=mean_value * 1.1, xmin=-0.16, xmax=0.16, color='none', linewidth=0)

        # 只在第一个子图添加图例句柄
        if i == 0:
            # 创建中位数线条的代理艺术家
            from matplotlib.lines import Line2D
            median_line = Line2D([0], [0], color='black', linewidth=2.4, linestyle='--')
            mean_line_legend = Line2D([0], [0], color='gray', linewidth=2.4, linestyle='--')
            legend_handles = [median_line, mean_line_legend]

        # 设置标题和标签
        ax.set_title(titles[col], fontsize=20, pad=20)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Value', fontsize=18)

        # 优化布局
        ax.tick_params(axis='both', which='major', labelsize=16)

        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

    # 在最后一个子图旁边添加图例
    axes[-1].legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)

    # 保存图像
    plt.savefig(f'{output_dir}/basic_indicators_boxplot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def plot_electrification_indicators_boxplot(indicator_data, output_data, output_dir='fig_1'):
    """
    绘制基于网络距离的关键电动化指标箱型图，包含5个指标，采用Nature期刊风格

    Parameters:
    indicator_data (pd.DataFrame): 包含指标数据的DataFrame
    output_data (pd.DataFrame): 包含输出数据的DataFrame
    output_dir (str): 图片保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 合并数据
    analysis_df = pd.merge(indicator_data, output_data, left_on='city', right_on='city_name')

    # 计算基于网络距离的指标
    analysis_df['knee_cost_per_km'] = analysis_df['knee_cost'] / analysis_df['network_length_km']
    analysis_df['knee_emission_per_km'] = analysis_df['knee_emission'] / analysis_df['network_length_km']
    analysis_df['knee_cs_per_km'] = analysis_df['knee_built_cs'] / analysis_df['network_length_km']
    analysis_df['knee_charger_per_km'] = ((analysis_df['knee_fast_per_cs'] + analysis_df['knee_slow_per_cs']) *
                                          analysis_df['knee_built_cs'] / analysis_df['network_length_km'])
    analysis_df['knee_ev_per_km'] = ((analysis_df['vehicle_count'] + analysis_df['knee_extra_large'] +
                                      analysis_df['knee_extra_medium'] + analysis_df['knee_extra_small'])
                                     / analysis_df['network_length_km'])

    # 定义5个关键电动化指标
    target_cols = ['knee_cost_per_km', 'knee_emission_per_km',
                   'knee_cs_per_km', 'knee_charger_per_km', 'knee_ev_per_km']

    # 指标标题映射
    titles = {
        'knee_cost_per_km': 'System Costs\nper km (M yuan/km)',
        'knee_emission_per_km': 'GHG Emissions\nper km (T/km)',
        'knee_cs_per_km': 'Charging Stations\nper km (km$^{-1}$)',
        'knee_charger_per_km': 'Chargers\nper km (km$^{-1}$)',
        'knee_ev_per_km': 'Electric Buses\nper km (km$^{-1}$)'
    }

    # 选择需要的列
    analysis_df = analysis_df[['city_name'] + target_cols].copy()

    # 转换为长格式
    melted_df = pd.melt(analysis_df,
                        id_vars='city_name',
                        value_vars=target_cols,
                        var_name='Indicator',
                        value_name='Value')

    # 设置Nature期刊风格（增大字体以适应多子图布局）
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # Nature期刊风格设置（字体增大2倍）
    plt.rcParams.update({
        'font.size': 18,  # 从9增大到18
        'axes.titlesize': 20,  # 从10增大到20
        'axes.labelsize': 18,  # 从9增大到18
        'xtick.labelsize': 16,  # 从8增大到16
        'ytick.labelsize': 16,  # 从8增大到16
        'legend.fontsize': 16,  # 从8增大到16
        'axes.linewidth': 1.6,  # 从0.8增大到1.6
        'xtick.major.width': 1.6,  # 从0.8增大到1.6
        'ytick.major.width': 1.6,  # 从0.8增大到1.6
        'xtick.minor.width': 1.2,  # 从0.6增大到1.2
        'ytick.minor.width': 1.2,  # 从0.6增大到1.2
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 8,  # 从4增大到8
        'ytick.major.size': 8,  # 从4增大到8
        'xtick.minor.size': 4,  # 从2增大到4
        'ytick.minor.size': 4,  # 从2增大到4
    })

    # 创建画布，5个子图，调整为1行5列
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # 增大画布尺寸
    axes = axes.flatten()

    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    # 科研配色方案（蓝色系）
    colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c']

    # 创建图例句柄和标签
    legend_handles = []
    legend_labels = ['Median', 'Mean']

    # 绘制每个指标
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        # 获取当前指标数据
        current_data = melted_df[melted_df['Indicator'] == col]

        # 排序数据以便优化标注
        current_data = current_data.sort_values('Value')

        # 绘制箱型图 - 使用蓝色系配色
        box = sns.boxplot(
            y='Value',
            data=current_data,
            ax=ax,
            orient='v',
            width=0.6,  # 从0.3增大到0.6
            fliersize=6,  # 从3增大到6
            showmeans=False,
            color=colors[i],
            boxprops=dict(linewidth=2, edgecolor='black', facecolor=colors[i] + '80'),  # 增大线宽
            medianprops=dict(color='black', linewidth=2.4, linestyle='--'),  # 增大线宽
            whiskerprops=dict(color='black', linewidth=2),  # 增大线宽
            capprops=dict(color='black', linewidth=2)  # 增大线宽
        )

        # 计算并绘制平均值线（灰色虚线，只在箱体内部）
        mean_value = current_data['Value'].mean()

        # 平均值线只在箱子内部显示
        mean_line = ax.hlines(y=mean_value, xmin=-0.3, xmax=0.3, color='gray', linestyle='--', linewidth=2.4)  # 增大线宽

        # 添加一条隐藏的线，xmin和xmax为0.1级别，但不显示在legend中
        hidden_line = ax.hlines(y=mean_value * 1.1, xmin=-0.6, xmax=0.6, color='none', linewidth=0)

        # 只在第一个子图添加图例句柄
        if i == 0:
            # 创建中位数线条的代理艺术家
            from matplotlib.lines import Line2D
            median_line = Line2D([0], [0], color='black', linewidth=2.4, linestyle='--')
            mean_line_legend = Line2D([0], [0], color='gray', linewidth=2.4, linestyle='--')
            legend_handles = [median_line, mean_line_legend]

        # 设置标题和标签
        ax.set_title(titles[col], fontsize=20, pad=20)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Value', fontsize=18)

        # 优化布局
        ax.tick_params(axis='both', which='major', labelsize=16)

        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

    # 在最后一个子图旁边添加图例
    axes[-1].legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)

    # 保存图像
    plt.savefig(f'{output_dir}/electrification_indicators_boxplot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# 使用示例
if __name__ == "__main__":
    # 读取数据
    static_data = pd.read_csv(r'../data/224city_indicators.csv')
    city_data = pd.read_csv(r'../data/224cities.csv')
    output_data = pd.read_csv(r'../data/224cities_output.csv')

    # 调用函数绘制运营指标图表
    # plot_operation_indicators_boxplot(static_data)

    # # 调用函数绘制电动化指标图表
    # plot_electrification_indicators_boxplot(static_data, output_data)
    #
    # 调用函数绘制route_count与人口、GDP的关系图
    # plot_route_count_vs_population_gdp(static_data, city_data)
    #
    # # 调用函数绘制budget与ind_electricity的关系图
    # plot_budget_vs_ind_electricity(city_data, output_data, static_data)
