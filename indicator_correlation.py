import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mosa4CN3_6'))
from mosa4CN3_6.vehicle_type import VehicleTypes

from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

if __name__ == '__main__':
    """0. Initialize"""
    cities = pd.read_csv(rf'../data/18cities.csv')
    province_list = cities['province'].to_list()
    city_list = cities['city'].to_list()
    date = '2025-1-20'

    indicators = pd.read_csv(rf'../data/city_indicators.csv')  # from indicator_calculation.py

    outputs = pd.read_csv(rf'../data/18cities_output.csv')  # from analyse_report.ipynb in mosa

    indicators['trip_cv'] = np.sqrt(indicators['variance']) / indicators['trip_number'] * 24
    indicators['route_avg_operation_distance'] = indicators['operation_distance'] / indicators['route_count']

    """1. correlation metrics"""
    # # 合并 indicators 和 outputs
    # merged_df = pd.merge(indicators, outputs, left_on='city', right_on='city_name')
    # merged_df = merged_df.drop(columns=['city', 'city_name'])
    # merged_df['knee_fast_per_vehicle'] = merged_df['knee_fast_per_cs'] * merged_df['knee_built_cs'] / merged_df['vehicle_count']
    # merged_df['knee_slow_per_vehicle'] = merged_df['knee_slow_per_cs'] * merged_df['knee_built_cs']  / merged_df['vehicle_count']
    # merged_df['knee_cs_per_vehicle'] = merged_df['knee_built_cs'] / merged_df['vehicle_count']
    # # 计算相关性矩阵
    # correlation_matrix = merged_df.corr()
    #
    # # 设置画布大小
    # plt.figure(figsize=(16, 12))
    #
    # # 绘制热力图
    # sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    # # 保存图像
    # plt.title("Correlation Matrix of All Indicators and Outputs")
    # plt.tight_layout()
    # plt.savefig('../data/output/correlation.png', dpi=300)
    # plt.close()

    """2. scatter plot"""
    # one_page = False
    #
    # vehicles = indicators[['city','vehicle_count']]
    # outputs = pd.merge(outputs, vehicles, left_on='city_name', right_on='city')
    # outputs['knee_fast_per_vehicle'] = outputs['knee_fast_per_cs'] * outputs['knee_built_cs'] / outputs['vehicle_count']
    # outputs['knee_slow_per_vehicle'] = outputs['knee_slow_per_cs'] * outputs['knee_built_cs']  / outputs['vehicle_count']
    # outputs['knee_cs_per_vehicle'] = outputs['knee_built_cs'] / outputs['vehicle_count']
    # outputs['avg_built_ratio'] = outputs['avg_built_cs'] / outputs['candidate_num']
    # outputs = outputs.drop(columns=['city', 'vehicle_count'])
    #
    # indicators = indicators[['city', 'route_total_distance', 'network_length_km', 'operation_distance', 'vehicle_count',
    #                          'route_avg_distance', 'route_avg_operation_distance', 'route_ratio_over_20km', 'route_ratio_over_10km',
    #                          'route_connectivity', 'network_connectivity', 'repetition_rate',
    #                          'average_circuity', 'ratio_circuity_over_2', 'ratio_circuity_over_1_5',
    #                          'avg_total_headway', 'avg_peak_headway', 'ratio_peak_over_10min', 'ratio_peak_over_20min',
    #                          'ratio_avg_over_15min', 'ratio_avg_over_30min',
    #                          'min_velocity', 'eprice', 'distance_per_fuel_vehicle','trip_cv']]
    #
    # outputs = outputs[['city_name', 'avg_built_ratio', 'avg_fast_per_cs',
    #                    'avg_slow_per_cs', 'avg_charger_ratio', 'knee_cost', 'knee_emission',
    #                    'knee_built_cs', 'knee_built_ratio', 'knee_fast_per_cs',
    #                    'knee_slow_per_cs', 'knee_charger_ratio', 'knee_extra_large', 'knee_extra_medium',
    #                    'knee_extra_small','knee_fast_per_vehicle',  'knee_slow_per_vehicle', 'knee_cs_per_vehicle',
    #                    'knee_cost_per_1Mkm','knee_emission_per_1Mkm','knee_cs_per_1Mkm',
    #                    'knee_charger_per_1Mkm','knee_ev_per_1Mkm']]
    #
    # # 假设 indicators 和 outputs 已经被正确加载并处理
    # # 合并数据集以确保 city 字段对应
    # merged_df = pd.merge(indicators, outputs, left_on='city', right_on='city_name')
    #
    # # 删除城市字段以避免计算相关性时出错
    # merged_df.drop(columns=['city', 'city_name'], inplace=True)
    #
    # # 计算相关性矩阵
    # correlation_matrix = merged_df.corr()
    #
    # # 提取 indicators 和 outputs 的列名
    # indicator_columns = indicators.columns.drop('city')
    # output_columns = outputs.columns.drop('city_name')
    #
    # if one_page:
    #     plt.rcParams['axes.unicode_minus'] = False
    #
    #     chinese_font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=8)
    #
    #     plt.rcParams.update({
    #         'font.size': 8,
    #         'xtick.labelsize': 6,
    #         'ytick.labelsize': 6,
    #         'legend.fontsize': 6,
    #         'xtick.direction': 'in',
    #         'ytick.direction': 'in'
    #     })
    #
    #     # ===== 创建画布并预留空间 =====
    #     fig, axes = plt.subplots(len(output_columns), len(indicator_columns), figsize=(52, 48))
    #
    #     # ===== 调整布局预留空间 =====
    #     fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.98, wspace=0.4, hspace=0.4)
    #     # 遍历所有 output 和 indicator 组合
    #     for i, output_col in enumerate(output_columns):
    #         for j, indicator_col in enumerate(indicator_columns):
    #             # 获取相关系数
    #             corr_coef = correlation_matrix.loc[indicator_col, output_col]
    #
    #             # 根据相关系数设置颜色
    #             color = plt.cm.coolwarm((corr_coef + 1) / 2)  # 将 [-1, 1] 映射到 [0, 1]
    #
    #             # 绘制散点图
    #             sns.scatterplot(x=merged_df[indicator_col], y=merged_df[output_col], ax=axes[i, j], color=color)
    #
    #             # 添加相关系数标注
    #             axes[i, j].text(0.5, 0.9, f'Corr: {corr_coef:.2f}', ha='center', va='center',
    #                             transform=axes[i, j].transAxes)
    #
    #             # 设置标签
    #             axes[i, j].set_xlabel(indicator_col)
    #             axes[i, j].set_ylabel(output_col)
    #
    #             # 关闭网格
    #             axes[i, j].grid(False)
    #
    #     # 底部中文标签（indicator 的中文名）
    #     chinese_x_labels = [
    #         '线路总长度', '线路网长度', '每日运营里程', '车辆数', '线路平均长度', '线路平均运营里程',
    #         '长度超过20km的线路比例', '长度超过10km的线路比例', '线路网连通度（考虑共线）',
    #         '线路网连通度（不考虑共线）', '线路重复系数', '平均非直线系数',
    #         '非直线系数超过2的线路比例', '非直线系数超过1.5的线路比例', '平均发车间隔',
    #         '平均高峰发车间隔', '高峰发车间隔小于10min的线路比例',
    #         '高峰发车间隔小于20min的线路比例', '全天平均发车间隔小于15min的线路比例',
    #         '全天平均发车间隔小于30min的线路比例', '线网平均车速', '电价', '燃油车车均运营里程',
    #         '24h班次数的变异系数'
    #     ]
    #
    #     # 左侧中文标签（output 的中文名）
    #     chinese_y_labels = [
    #         '所有解的充电站数/候选点数', '所有解的站均快充桩数', '所有解的站均慢充桩数', '所有解的快慢充桩数比例',
    #         '代表解的系统成本', '代表解的排放', '代表解的充电站数', '代表解的充电站数/候选点数',
    #         '代表解的站均快充桩数', '代表解的慢充桩数', '代表解的快慢充桩数比例',
    #         '代表解的增配大型车数', '代表解的增配中型车数', '代表解的增配小型车数','代表解的快充桩数/车辆数',
    #         '代表解的慢充桩数/车辆数','代表解的充电站数/车辆数', '代表解的每百万公里成本', '代表解的每百万公里排放',
    #         '代表解的每百万公里充电站数', '代表解的每百万公里充电桩数', '代表解的每百万公里车辆数'
    #     ]
    #
    #     outputs = outputs[['city_name', 'avg_built_ratio', 'avg_fast_per_cs',
    #                        'avg_slow_per_cs', 'avg_charger_ratio', 'knee_cost', 'knee_emission',
    #                        'knee_built_cs', 'knee_built_ratio', 'knee_fast_per_cs',
    #                        'knee_slow_per_cs', 'knee_charger_ratio', 'knee_extra_large', 'knee_extra_medium',
    #                        'knee_extra_small', 'knee_fast_per_vehicle', 'knee_slow_per_vehicle', 'knee_cs_per_vehicle',
    #                        'knee_cost_per_1Mkm', 'knee_emission_per_1Mkm', 'knee_cs_per_1Mkm',
    #                        'knee_charger_per_1Mkm', 'knee_ev_per_1Mkm']]
    #
    #     # 计算左侧标签纵向位置（基于每一行的第一个子图）
    #     col_centers = [ax.get_position().x0 + (ax.get_position().x1 - ax.get_position().x0) / 2 for ax in axes[0, :]]
    #
    #     # ===== 添加底部中文标签（indicator 的中文名）=====
    #     for idx, label in enumerate(chinese_x_labels):
    #         x_pos = col_centers[idx]
    #         fig.text(x_pos, 0.01, label, ha='center', va='center',
    #                  fontsize=6, fontproperties=chinese_font)
    #
    #     # ===== 获取每一行中间位置（用于左侧中文标签对齐）=====
    #     row_centers = [ax.get_position().y0 + (ax.get_position().y1 - ax.get_position().y0) / 2 for ax in axes[:, 0]]
    #
    #     # ===== 添加左侧中文标签（output 的中文名）=====
    #     for idx, label in enumerate(chinese_y_labels):
    #         if idx >= len(row_centers):
    #             break
    #         fig.text(0.02, row_centers[idx], label, ha='right', va='center',
    #                  fontsize=6, rotation=45, fontproperties=chinese_font)
    #
    #     # ===== 保存图片 =====
    #     plt.savefig('../data/output/scatter_matrix_with_chinese_labels.png', dpi=300)
    #     plt.close()
    #
    # else:
    #     plt.rcParams['axes.unicode_minus'] = False
    #
    #     # 创建基础输出路径
    #     base_output_dir = '../data/output/scatter_plots_separated'
    #     os.makedirs(base_output_dir, exist_ok=True)
    #
    #     # 遍历所有 output 和 indicator 组合
    #     for i, output_col in enumerate(output_columns):
    #         for j, indicator_col in enumerate(indicator_columns):
    #             # 获取相关系数
    #             corr_coef = correlation_matrix.loc[indicator_col, output_col]
    #
    #             # 创建图像和坐标轴
    #             fig, ax = plt.subplots(figsize=(4, 4))  # 正方形比例
    #
    #             # 根据相关系数设置颜色
    #             color = plt.cm.coolwarm((corr_coef + 1) / 2)  # 将 [-1, 1] 映射到 [0, 1]
    #
    #             # 绘制散点图
    #             sns.scatterplot(x=merged_df[indicator_col], y=merged_df[output_col], ax=ax, color=color)
    #
    #             # 添加相关系数标注
    #             ax.text(0.5, 0.9, f'Corr: {corr_coef:.2f}', ha='center', va='center',
    #                     transform=ax.transAxes, fontsize=8)
    #
    #             # 设置标签和标题（统一使用中文字体）
    #             ax.set_xlabel(indicator_col)
    #             ax.set_ylabel(output_col)
    #             ax.set_title(f'{indicator_col} vs {output_col}')
    #
    #             # 关闭网格
    #             ax.grid(False)
    #
    #             # 构建保存路径
    #             output_subdir = os.path.join(base_output_dir, output_col)
    #             os.makedirs(output_subdir, exist_ok=True)
    #
    #             filename = f"{indicator_col}.png"
    #             save_path = os.path.join(output_subdir, filename)
    #
    #             # 保存图片
    #             plt.tight_layout()
    #             plt.savefig(save_path, dpi=150)
    #             plt.close(fig)

    """3. boxplot for 5 key electrification indicators (per operational distance or network distance)"""
    # per_network_distance = False
    # if per_network_distance:
    #     distance_type = 'network'
    #     analysis_df = pd.merge(indicators, outputs, left_on='city', right_on='city_name')
    #     analysis_df['knee_cost_per_km'] = analysis_df['knee_cost'] / analysis_df['network_length_km']
    #     analysis_df['knee_emission_per_km'] = analysis_df['knee_emission'] / analysis_df['network_length_km']
    #     analysis_df['knee_cs_per_km'] = analysis_df['knee_built_cs'] / analysis_df['network_length_km']
    #     analysis_df['knee_charger_per_km'] = ((analysis_df['knee_fast_per_cs']+analysis_df['knee_slow_per_cs']) *
    #                                         analysis_df['knee_built_cs']/ analysis_df['network_length_km'])
    #     analysis_df['knee_ev_per_km'] = ((analysis_df['vehicle_count']+analysis_df['knee_extra_large']+
    #                                    analysis_df['knee_extra_medium']+analysis_df['knee_extra_small'])
    #                                    / analysis_df['network_length_km'])
    #     titles = {
    #         'knee_cost_per_km': 'system costs per km (M yuan/km)',
    #         'knee_emission_per_km': 'GHG emissions per km (T/km)',
    #         'knee_cs_per_km': 'charging stations per km (km$^{-1}$)',
    #         'knee_charger_per_km': 'chargers per km (km$^{-1}$)',
    #         'knee_ev_per_km': 'electric buses (km$^{-1}$)'
    #     }
    #     target_cols = ['knee_cost_per_km', 'knee_emission_per_km',
    #                    'knee_cs_per_km', 'knee_charger_per_km', 'knee_ev_per_km']
    #
    #     analysis_df = analysis_df[['city_name'] + target_cols].copy()
    # else:
    #     distance_type = 'operation'
    #     # 提取目标列
    #     target_cols = ['knee_cost_per_1Mkm', 'knee_emission_per_1Mkm',
    #                    'knee_cs_per_1Mkm', 'knee_charger_per_1Mkm', 'knee_ev_per_1Mkm']
    #
    #     titles = {
    #         'knee_cost_per_1Mkm': 'system costs per km (yuan/km)',
    #         'knee_emission_per_1Mkm': 'GHG emissions per km (T/M km)',
    #         'knee_cs_per_1Mkm': 'charging stations per km (M km$^{-1}$)',
    #         'knee_charger_per_1Mkm': 'chargers per km (M km$^{-1}$)',
    #         'knee_ev_per_1Mkm': 'electric buses (M km$^{-1}$)'
    #     }
    #
    #     # 创建分析数据集
    #     analysis_df = outputs[['city_name'] + target_cols].copy()
    #
    # # 转换为长格式
    # melted_df = pd.melt(analysis_df,
    #                     id_vars='city_name',
    #                     value_vars=target_cols,
    #                     var_name='Indicator',
    #                     value_name='Value')
    #
    # # 设置中英文字体
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']  # 中文用微软雅黑，英文用Arial
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # # 创建画布
    # fig, axes = plt.subplots(1, len(target_cols), figsize=(18, 7))
    # plt.subplots_adjust(wspace=0.4)
    #
    # # 添加主标题
    # if per_network_distance:
    #     fig.suptitle('Key indicators for bus electrification per network distance',
    #                  fontsize=14, fontname='Arial', y=0.95)
    # else:
    #     fig.suptitle('Key indicators for bus electrification per operational distance',
    #                  fontsize=14, fontname='Arial', y=0.95)
    #
    # # 自定义箱型图样式
    # if per_network_distance:
    #     # 网络相关指标使用蓝色系
    #     boxprops = dict(linewidth=1.5, edgecolor='navy', facecolor=(0.7, 0.8, 1.0, 0.7))
    #     palette = 'Blues'
    # else:
    #     # 运营相关指标使用绿色系
    #     boxprops = dict(linewidth=1.5, edgecolor='darkgreen', facecolor=(0.7, 1.0, 0.8, 0.7))
    #     palette = 'Greens'
    #
    # # 绘制每个指标
    # for i, (ax, col) in enumerate(zip(axes, target_cols)):
    #     # 获取当前指标数据
    #     current_data = melted_df[melted_df['Indicator'] == col]
    #
    #     # 排序数据以便优化标注
    #     current_data = current_data.sort_values('Value')
    #
    #     # 绘制箱型图（移除均值标记，使用自定义颜色）
    #     box = sns.boxplot(
    #         y='Value',
    #         data=current_data,
    #         ax=ax,
    #         orient='v',
    #         width=0.3,
    #         fliersize=3,
    #         showmeans=False,
    #         palette=palette,
    #         boxprops=boxprops
    #     )
    #
    #     # 获取箱子右侧位置
    #     box_x_max = box.get_xlim()[1]
    #
    #     for j, (_, row) in enumerate(current_data.iterrows()):
    #         # 左右交替位置
    #         if j % 2 == 0:
    #             x_pos = box_x_max * 0.98  # 右侧
    #             ha_align = 'right'
    #         else:
    #             x_pos = box_x_max * -1  # 左侧
    #             ha_align = 'left'
    #
    #         # 垂直方向的小偏移避免重叠
    #         y_pos = row['Value']
    #
    #         # 添加标注
    #         if per_network_distance:
    #             text_color = 'darkblue'  # 蓝色系使用深蓝色文字
    #         else:
    #             text_color = 'darkgreen'  # 绿色系使用深绿色文字
    #
    #         ax.text(
    #             x_pos,
    #             y_pos,
    #             row['city_name'],
    #             va='center',
    #             ha=ha_align,
    #             fontsize=8,
    #             color=text_color,
    #             rotation=0
    #         )
    #
    #     # 调整x轴范围留出左右标注空间
    #     current_xlim = ax.get_xlim()
    #     ax.set_xlim(current_xlim[0] - current_xlim[1] * 0.15, current_xlim[1] * 1.15)
    #
    #     # 设置标题和标签（使用Arial字体）
    #     ax.set_title(titles[col], fontname='Arial', fontsize=10)
    #     ax.set_xlabel('', fontname='Arial')
    #     ax.set_ylabel('Value', fontname='Arial', fontsize=9)
    #
    #     # 优化布局
    #     ax.tick_params(axis='both', which='major', labelsize=8)
    #     for tick in ax.get_xticklabels():
    #         tick.set_fontname('Arial')
    #     for tick in ax.get_yticklabels():
    #         tick.set_fontname('Arial')
    #
    # # 保存图像
    # plt.savefig(f'../data/output/boxplots_{distance_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    """4. scatter plot for route number and operation_distance"""
    # # 验证route_count字段存在性
    # if 'route_count' not in indicators.columns:
    #     raise ValueError("route_count字段未在indicators数据框中找到，请检查数据源")
    #
    # # 提取绘图所需数据
    # scatter_data = indicators[['city', 'route_count', 'operation_distance']].copy()
    #
    # # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # # 创建画布
    # plt.figure(figsize=(14, 10))
    #
    # # 绘制基础散点图
    # ax = sns.scatterplot(
    #     x='route_count',
    #     y='operation_distance',
    #     data=scatter_data,
    #     s=100,
    #     alpha=0.7
    # )
    #
    # # 添加城市标注
    # texts = []
    # for idx, row in scatter_data.iterrows():
    #     texts.append(
    #         ax.text(
    #             row['route_count'],
    #             row['operation_distance'],
    #             row['city'],
    #             fontsize=9,
    #             ha='right',
    #             va='bottom'
    #         )
    #     )
    #
    # # 设置样式
    # ax.set_xlabel('线路数量 (Route Count)', fontsize=12)
    # ax.set_ylabel('运营里程 (Operation Distance)', fontsize=12)
    # ax.set_title('城市线路数量与运营里程关系', fontsize=14)
    # ax.grid(True, linestyle='--', alpha=0.3)
    #
    # # 保存图像
    # plt.tight_layout()
    # plt.savefig('../data/output/route_operation_scatter.png', dpi=300, bbox_inches='tight')
    # plt.close()

    """5. regression"""
    indicators.rename(columns={'ratio_peak_over_10min': 'ratio_peak_headway_under_10min',
                               'ratio_peak_over_20min': 'ratio_peak_headway_under_20min',
                               'ratio_avg_over_15min': 'ratio_total_headway_under_15min',
                               'ratio_avg_over_30min': 'ratio_total_headway_under_30min'}, inplace=True)
    independent_indicators = indicators[['city', 'network_length_km', 'operation_distance',
                                         'route_avg_operation_distance', 'repetition_rate',
                                         'ratio_peak_headway_under_10min',
                                         'average_circuity', 'min_velocity', 'eprice',
                                         'trip_cv']]

    # 合并数据
    regression_data = pd.merge(
        independent_indicators,
        outputs[['city_name', 'knee_cost_per_1Mkm', 'knee_emission_per_1Mkm',
                 'knee_cs_per_1Mkm', 'knee_charger_per_1Mkm', 'knee_ev_per_1Mkm']],
        left_on='city', right_on='city_name'
    )

    # 保存原始数据用于计算原始系数
    X_raw = regression_data.iloc[:, 2:-6]  # 跳过city列和因变量列
    X_raw_columns = list(X_raw.columns)

    # 标准化自变量
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled = sm.add_constant(X_scaled)  # 添加常数项

    # 定义因变量列表
    target_cols = ['knee_cost_per_1Mkm', 'knee_emission_per_1Mkm',
                   'knee_cs_per_1Mkm', 'knee_charger_per_1Mkm', 'knee_ev_per_1Mkm']

    # 创建系数和统计指标DataFrame
    coeff_df = pd.DataFrame(index=['intercept'] + list(independent_indicators.columns[2:]))
    original_coeff_df = pd.DataFrame(index=['intercept'] + list(independent_indicators.columns[2:]))
    r2_df = pd.DataFrame(index=['metrics'])

    for target in target_cols:
        # 定义因变量
        y = regression_data[target]

        # 计算因变量标准差
        y_std = y.std()

        # 拟合模型
        model = sm.OLS(y, X_scaled).fit()

        # 存储标准化系数
        coeff_df[f'{target}_coef'] = [model.params[0]] + list(model.params[1:])
        coeff_df[f'{target}_pval'] = [model.pvalues[0]] + list(model.pvalues[1:])

        # 计算原始系数
        original_intercept = model.params[0] * y_std  # 常数项的原始系数
        original_coefs = [model.params[i + 1] * y_std / X_raw[X_raw_columns[i]].std() for i in
                          range(len(X_raw_columns))]
        original_coeff_df[f'{target}_coef'] = [original_intercept] + original_coefs

        # 添加原始系数的p值（与标准化系数相同）
        original_coeff_df[f'{target}_pval'] = [model.pvalues[0]] + list(model.pvalues[1:])

        # 存储R方等指标
        r2_df.loc['r_squared', f'{target}_metrics'] = model.rsquared
        r2_df.loc['adj_r_squared', f'{target}_metrics'] = model.rsquared_adj
        r2_df.loc['f_pvalue', f'{target}_metrics'] = model.f_pvalue
        r2_df.loc['mse', f'{target}_metrics'] = model.mse_total

    # 计算VIF（基于标准化后的X）
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    vif["features"] = ['const'] + list(independent_indicators.columns[2:])

    # 打印标准化回归结果
    print("\n=== 标准化回归系数 ===")
    print(coeff_df.to_string(float_format='%.3f'))

    print("\n=== 原始回归系数 ===")
    print(original_coeff_df.to_string(float_format='%.2e'))

    print("\n=== R方等模型指标 ===")
    print(r2_df.to_string(float_format='%.3f'))

    """6. cost/GDP and electricity load ratio"""
    # # 需要按车型才知道耗电量
    # outputs['electricity'] = 0
    # for i in range(len(outputs)):
    #     # if i != 0:
    #     #     continue
    #     city_name = outputs['city_name'][i]
    #     vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')
    #     result = vs_parking_df.groupby('vehicle_type')['distance'].sum()
    #     proportions = result / result.sum()
    #     outputs.loc[i,'electricity'] = outputs.loc[i,'knee_emission'] * 1e6 * sum(
    #         proportions.get(vt, 0) / VehicleTypes(outputs.loc[i,f'{vt}_model'], vt).per_emission /
    #         VehicleTypes(outputs.loc[i,f'{vt}_model'], vt).e2s_ratio for vt in ['large', 'medium', 'small'])
    # merged_df = pd.merge(outputs, cities, left_on='city_name', right_on='city')
    # merged_df['electricity_ratio'] = merged_df['electricity'] / (merged_df['all_electricity']*1e9)
    #
    # merged_df['cost_ratio'] = merged_df['knee_cost'] / merged_df['GDP'] / 1e6
    #
    # plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # fig, ax = plt.subplots(figsize=(12, 10))
    #
    # # 放大圆的缩放参数
    # gdp_size = (merged_df['GDP'] - merged_df['GDP'].min()) / (
    #             merged_df['GDP'].max() - merged_df['GDP'].min()) * 600 + 200
    # elec_size = (merged_df['all_electricity'] - merged_df['all_electricity'].min()) / (
    #             merged_df['all_electricity'].max() - merged_df['all_electricity'].min()) * 300 + 100
    #
    # # 保证外层圆大于内层圆
    # gdp_size = np.maximum(gdp_size, elec_size + 50)
    #
    # # 外层圆（GDP）- 紫色
    # outer_scatter = ax.scatter(
    #     merged_df['cost_ratio'] * 100,
    #     merged_df['electricity_ratio'] * 100,
    #     s=gdp_size,
    #     c='#7B1FA2',
    #     alpha=0.5,
    #     edgecolors='none',
    #     zorder=2
    # )
    #
    # # 内层圆（all_electricity）- 黄色
    # inner_scatter = ax.scatter(
    #     merged_df['cost_ratio'] * 100,
    #     merged_df['electricity_ratio'] * 100,
    #     s=elec_size,
    #     c='#FFD600',
    #     alpha=0.8,
    #     edgecolors='none',
    #     zorder=3
    # )
    #
    # # 中心点（更小）
    # center_points = ax.scatter(
    #     merged_df['cost_ratio'] * 100,
    #     merged_df['electricity_ratio'] * 100,
    #     s=8,
    #     c='black',
    #     marker='o',
    #     alpha=1.0,
    #     zorder=4
    # )
    #
    # # 城市名称标注
    # for i, city in enumerate(merged_df['city_name']):
    #     ax.annotate(
    #         city,
    #         (merged_df['cost_ratio'].iloc[i] * 100, merged_df['electricity_ratio'].iloc[i] * 100),
    #         xytext=(5, 5),
    #         textcoords='offset points',
    #         fontsize=8,
    #         fontfamily='Arial' if all(ord(char) < 128 for char in city) else 'Microsoft YaHei',
    #         zorder=5
    #     )
    #
    # # 轴标签和标题
    # ax.set_xlabel('Cost Ratio (%)', fontfamily='Arial', fontsize=12)
    # ax.set_ylabel('Electricity Ratio (%)', fontfamily='Arial', fontsize=12)
    # ax.set_title('Cost Ratio vs Electricity Ratio by City', fontfamily='Arial', fontsize=14)
    #
    # # 网格
    # ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    #
    # # 图例（用代理对象更美观，center点用小黑圆）
    # outer_proxy = mpatches.Circle((0, 0), radius=5, color='#7B1FA2', alpha=0.5, label='GDP')
    # inner_proxy = mpatches.Circle((0, 0), radius=3, color='#FFD600', alpha=0.8, label='All Electricity')
    # ax.legend(handles=[outer_proxy, inner_proxy], loc='upper right')
    #
    #
    # # 百分号格式
    # def to_percent(x, pos):
    #     return f'{x:.2f}%'
    #
    #
    # ax.xaxis.set_major_formatter(FuncFormatter(to_percent))
    # ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    #
    # plt.tight_layout()
    # plt.savefig('../data/output/cost_electricity_ratio.png', dpi=300, bbox_inches='tight')
    # plt.close()
