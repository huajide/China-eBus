import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy import stats  # 需要导入scipy.stats

# 添加字体支持
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_correlation_matrix(indicators, outputs, output_dir='fig_1'):
    """绘制指标变量的相关系数矩阵热力图"""

    # 计算额外指标
    indicators = indicators.copy()  # 避免修改原始数据
    indicators['trip_cv'] = np.sqrt(indicators['variance']) / indicators['trip_number'] * 24
    indicators['route_avg_operation_distance'] = indicators['operation_distance'] / indicators['route_count']

    # 选择指标变量
    # indicators_selected = indicators[
    #     ['network_length_km',
    #      'route_avg_distance', 'route_avg_operation_distance', 'route_ratio_over_10km',
    #      'repetition_rate',
    #      'average_circuity',
    #      'avg_total_headway',
    #      'ratio_avg_over_15min',
    #      'e_price', 'distance_per_fuel_vehicle', 'trip_cv', 'pop_coverage',
    #      'operation_total_distance_ratio',
    #      'avg_speed',
    #      'route_terminal_ratio',
    #      'avg_service_hour'
    #      ]]
    indicators_selected = indicators[['route_avg_distance',
       'route_ratio_over_10km',
       'network_connectivity', 'network_length_km',
       'repetition_rate', 'average_circuity',
       'ratio_circuity_over_2',
       'ratio_circuity_over_1_5', 'average_stop_spacing',
       'trip_cv', 'avg_total_headway',
       'ratio_avg_over_15min', 'min_velocity',
       'distance_per_fuel_vehicle',
       'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio',
       'avg_service_hour', 'e_price']]

    # 计算相关性矩阵
    correlation_matrix = indicators_selected.corr()

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 创建上半部分的mask（只显示下半部分）
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # 绘制热力图（只显示下半部分）
    heatmap = sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='RdBu_r',  # 红蓝对比色
        square=True,
        cbar_kws={"shrink": .8},
        vmin=-1,
        vmax=1,
        annot_kws={"size": 8}
    )

    # 设置标签（使用原始变量名）
    labels = list(indicators_selected.columns)

    # 设置x轴标签
    plt.xticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=45,
        ha='right',
        fontsize=10
    )

    # 设置y轴标签
    plt.yticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=0,
        fontsize=10
    )

    # 调整布局
    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"相关系数矩阵热力图已保存至: {output_dir}/correlation_matrix.png")


def calculate_vif(indicators):
    """计算方差膨胀因子(VIF)以检查共线性"""
    # 计算额外指标
    indicators = indicators.copy()
    indicators['trip_cv'] = np.sqrt(indicators['variance']) / indicators['trip_number'] * 24
    indicators['route_avg_operation_distance'] = indicators['operation_distance'] / indicators['route_count']

    # 选择指标变量
    indicators_selected = indicators[['route_avg_distance',
       'route_ratio_over_10km',
       'network_connectivity', 'network_length_km',
       'repetition_rate', 'average_circuity',
       'ratio_circuity_over_1_5', 'average_stop_spacing',
       'trip_cv', 'avg_total_headway',
       'min_velocity',
       'distance_per_fuel_vehicle',
       'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio', 'e_price']]
    # indicators_selected = indicators[
    #     ['network_length_km',
    #      'pop_coverage',
    #      'route_avg_distance',
    #      'operation_total_distance_ratio',
    #      'repetition_rate',
    #      'average_circuity',
    #      'route_terminal_ratio',
    #      'trip_cv','avg_speed',
    #      'e_price'
    #      ]]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indicators_selected)
    X_scaled = sm.add_constant(X_scaled)  # 添加常数项

    # 计算VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = ['const'] + list(indicators_selected.columns)
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

    return vif_data


def perform_ols_regression(indicators, outputs, standardize=True, format='separate'):
    """执行OLS回归分析
    Parameters:
    - format: 'separate' (默认) - 系数和p值分开显示
              'combined' - 系数和显著性标记合并显示
    """
    # 合并数据
    merged_data = pd.merge(indicators, outputs, left_on='city', right_on='city_name')

    # 计算额外指标
    merged_data['trip_cv'] = np.sqrt(merged_data['variance']) / merged_data['trip_number'] * 24
    merged_data['route_avg_operation_distance'] = merged_data['operation_distance'] / merged_data['route_count']

    # 选择自变量
    # X = merged_data[
    #     ['network_length_km',
    #      'route_avg_distance',
    #      'operation_total_distance_ratio',
    #      'repetition_rate',
    #      'average_circuity',
    #      'route_terminal_ratio',
    #      'trip_cv', 'avg_speed',
    #      'e_price'
    #      ]]
    X = merged_data[
        ['route_avg_distance',
       'network_length_km',
       'repetition_rate','average_stop_spacing','trip_cv',
       'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio', 'e_price']]  # 'network_connectivity',


    # 定义因变量（每公里指标）
    y_vars = {
        'cost_per_km': merged_data['knee_cost'] / merged_data['network_length_km'],
        'emission_per_km': merged_data['knee_emission'] / merged_data['network_length_km'],
        'cs_per_km': merged_data['knee_built_cs'] / merged_data['network_length_km'],
        'fast_chargers_per_km': merged_data['knee_fast_per_cs'] * merged_data['knee_built_cs'] / merged_data[
            'network_length_km'],
        'slow_chargers_per_km': merged_data['knee_slow_per_cs'] * merged_data['knee_built_cs'] / merged_data[
            'network_length_km'],
        'ev_per_km': (merged_data['knee_extra_large'] + merged_data['knee_extra_medium'] +
                      merged_data['knee_extra_small'] + merged_data['vehicle_count']) / merged_data[
                         'network_length_km']
    }

    # 处理缺失值
    full_data = pd.concat([X, pd.DataFrame(y_vars)], axis=1)
    clean_data = full_data.dropna()

    # 分离清理后的数据
    X_clean = clean_data[X.columns]
    y_clean_dict = {y_name: clean_data[y_name] for y_name in y_vars.keys()}

    # 根据选项决定是否标准化
    if standardize:
        # 标准化自变量
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_clean)
        print("使用标准化数据进行回归分析")
    else:
        # 不标准化，直接使用原始数据
        X_processed = X_clean.values
        print("使用原始数据进行回归分析")

    X_processed = sm.add_constant(X_processed)  # 添加常数项

    # 存储结果用于表格展示
    feature_names = ['const'] + list(X.columns)

    if format == 'separate':
        # 系数和p值分开显示
        results_table = pd.DataFrame(index=feature_names)
    else:
        # 系数和显著性标记合并显示，添加R方信息行
        results_table = pd.DataFrame(index=['R²'] + feature_names)

    # 存储模型统计信息
    model_stats = pd.DataFrame(index=['R²', '调整R²', 'F检验p值'])

    # 存储每个模型的R方值（用于combined格式）
    r_squared_values = {}

    # 对每个因变量进行回归
    for y_name, y_values in y_clean_dict.items():
        # 确保使用相同索引的数据
        common_index = X_clean.index.intersection(y_values.index)
        X_reg = X_processed[X_clean.index.isin(common_index), :]
        y_reg = y_values[common_index].values

        # 拟合模型
        model = sm.OLS(y_reg, X_reg).fit()

        # 存储R方值
        r_squared_values[y_name] = model.rsquared

        if format == 'separate':
            # 存储系数和p值，使用实际的变量名
            results_table[f'{y_name}_coef'] = [model.params[0]] + list(model.params[1:])  # 常数项+其他系数
            results_table[f'{y_name}_pval'] = [model.pvalues[0]] + list(model.pvalues[1:])  # 常数项+其他p值
        else:
            # 合并系数和显著性标记
            combined_values = []
            # 第一行添加R方值
            combined_values.append(f"{model.rsquared:.4f}")
            # 添加系数和显著性标记
            params = [model.params[0]] + list(model.params[1:])
            pvalues = [model.pvalues[0]] + list(model.pvalues[1:])

            for param, pval in zip(params, pvalues):
                if pval < 0.001:
                    significance = '***'
                elif pval < 0.01:
                    significance = '**'
                elif pval < 0.05:
                    significance = '*'
                else:
                    significance = ''
                combined_values.append(f"{param:.6f}{significance}")

            results_table[f'{y_name}'] = combined_values

        # 存储模型统计信息
        model_stats[y_name] = [model.rsquared, model.rsquared_adj, model.f_pvalue]

    # 打印结果表格
    if format == 'separate':
        print("\n=== 回归系数和p值 ===")
        print(results_table.to_string(float_format='%.6f'))
    else:
        print("\n=== 回归系数 (显著性: *** p<0.001, ** p<0.01, * p<0.05) ===")
        print(results_table.to_string(float_format='%.6f'))

    print("\n=== 模型统计信息 ===")
    print(model_stats.to_string(float_format='%.4f'))

    return results_table, model_stats


def analyze_vehicle_choice_anova(indicators, outputs):
    """
    使用方差分析(ANOVA)分析车型选择与各因素的关系

    Parameters:
    - indicators: 指标数据DataFrame
    - outputs: 输出数据DataFrame，包含车型选择列

    Returns:
    - anova_results: ANOVA分析结果
    """

    # 合并数据
    merged_data = pd.merge(indicators, outputs, left_on='city', right_on='city_name')

    # 计算额外指标
    merged_data['trip_cv'] = np.sqrt(merged_data['variance']) / merged_data['trip_number'] * 24
    merged_data['route_avg_operation_distance'] = merged_data['operation_distance'] / merged_data['route_count']

    # 选择自变量（与perform_ols_regression中使用的变量一致）
    X_vars = ['route_avg_distance', 'network_length_km', 'repetition_rate',
              'average_stop_spacing', 'trip_cv', 'operation_total_distance_ratio',
              'avg_speed', 'route_terminal_ratio', 'e_price']

    X = merged_data[X_vars]

    # 因变量：三种车型选择（分类变量）
    vehicle_types = {
        'large_model': merged_data['large_model'],
        'medium_model': merged_data['medium_model'],
        'small_model': merged_data['small_model']
    }

    # 存储ANOVA结果
    anova_results = {}

    for vehicle_type, y_values in vehicle_types.items():
        print(f"\n=== {vehicle_type} ANOVA分析 ===")

        # 构建分析数据集
        analysis_data = pd.concat([X, y_values], axis=1)
        analysis_data.columns = X_vars + [vehicle_type]

        # 删除缺失值
        clean_data = analysis_data.dropna()

        if len(clean_data) == 0:
            print(f"警告: {vehicle_type} 数据不足，无法进行分析")
            continue

        y_clean = clean_data[vehicle_type]
        X_clean = clean_data[X_vars]

        # 对每个自变量进行单因素ANOVA分析
        factor_results = {}

        for factor in X_vars:
            # 按照车型选择分组
            groups = [group[factor].values for name, group in clean_data.groupby(vehicle_type)]

            # 执行单因素方差分析
            if len(groups) > 1 and all(len(group) > 0 for group in groups):
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    factor_results[factor] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    factor_results[factor] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }
            else:
                factor_results[factor] = {
                    'f_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': False
                }

        # 转换为DataFrame便于展示
        results_df = pd.DataFrame(factor_results).T
        results_df = results_df.sort_values('p_value')

        # 添加显著性标记
        def significance_mark(p_val):
            if pd.isna(p_val):
                return ''
            elif p_val < 0.001:
                return '***'
            elif p_val < 0.01:
                return '**'
            elif p_val < 0.05:
                return '*'
            else:
                return ''

        results_df['significance'] = results_df['p_value'].apply(significance_mark)

        # 显示显著因素
        significant_factors = results_df[results_df['significant']].copy()
        if not significant_factors.empty:
            print(f"\n对{vehicle_type}选择有显著影响的因素:")
            # 格式化显示显著因素
            significant_display = significant_factors[['f_statistic', 'p_value', 'significance']].copy()
            significant_display['f_statistic'] = significant_display['f_statistic'].map('{:.4f}'.format)
            significant_display['p_value'] = significant_display['p_value'].map('{:.6f}'.format)
            print(significant_display.to_string())

            # 显示这些显著因素在不同车型选择类别下的均值
            print(f"\n{vehicle_type}显著影响因素的各选择类别均值:")
            for factor in significant_factors.index:
                print(f"\n{factor}:")
                group_stats = clean_data.groupby(vehicle_type)[factor].agg(['mean', 'std', 'count'])
                print(group_stats.round(4).to_string())
        else:
            print(f"\n未发现对{vehicle_type}选择有显著影响的因素")

        anova_results[vehicle_type] = results_df

    return anova_results


if __name__ == "__main__":
    # 读取数据
    indicator_data = pd.read_csv(r'../data/224city_indicators.csv')
    output_data = pd.read_csv(r'../data/224cities_output.csv')

    # 绘制相关性矩阵
    # plot_correlation_matrix(indicator_data, output_data)

    # 计算VIF
    # print("\n=== 方差膨胀因子 (VIF) ===")
    # vif_result = calculate_vif(indicator_data)
    # print(vif_result.to_string(index=False))
    #
    # 执行OLS回归
    print("\n=== OLS回归分析 ===")
    regression_results, regression_stats = perform_ols_regression(indicator_data, output_data, standardize=False,
                                                                  format='combined')
    regression_results.to_excel(r'fig_1/ols.xlsx')


    # ANOVA分析车型选择影响因素
    print("\n=== 车型选择影响因素的ANOVA分析 ===")
    anova_results = analyze_vehicle_choice_anova(indicator_data, output_data)


