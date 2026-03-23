import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_correlation_matrix(indicators, outputs, output_dir='fig_1'):
    indicators = indicators.copy()
    indicators['trip_cv'] = np.sqrt(indicators['variance']) / indicators['trip_number'] * 24
    indicators['route_avg_operation_distance'] = indicators['operation_distance'] / indicators['route_count']
    indicators_selected = indicators[['route_avg_distance', 'route_ratio_over_10km', 'network_connectivity', 'network_length_km', 'repetition_rate', 'average_circuity', 'ratio_circuity_over_2', 'ratio_circuity_over_1_5', 'average_stop_spacing', 'trip_cv', 'avg_total_headway', 'ratio_avg_over_15min', 'min_velocity', 'distance_per_fuel_vehicle', 'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio', 'avg_service_hour', 'e_price']]
    correlation_matrix = indicators_selected.corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', square=True, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1, annot_kws={'size': 8})
    labels = list(indicators_selected.columns)
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0, fontsize=10)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Correlation matrix saved to: {output_dir}/correlation_matrix.png')

def calculate_vif(indicators):
    indicators = indicators.copy()
    indicators['trip_cv'] = np.sqrt(indicators['variance']) / indicators['trip_number'] * 24
    indicators['route_avg_operation_distance'] = indicators['operation_distance'] / indicators['route_count']
    indicators_selected = indicators[['route_avg_distance', 'route_ratio_over_10km', 'network_connectivity', 'network_length_km', 'repetition_rate', 'average_circuity', 'ratio_circuity_over_1_5', 'average_stop_spacing', 'trip_cv', 'avg_total_headway', 'min_velocity', 'distance_per_fuel_vehicle', 'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio', 'e_price']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indicators_selected)
    X_scaled = sm.add_constant(X_scaled)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = ['const'] + list(indicators_selected.columns)
    vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    return vif_data

def perform_ols_regression(indicators, outputs, standardize=True, format='separate'):
    merged_data = pd.merge(indicators, outputs, left_on='city', right_on='city_name')
    merged_data['trip_cv'] = np.sqrt(merged_data['variance']) / merged_data['trip_number'] * 24
    merged_data['route_avg_operation_distance'] = merged_data['operation_distance'] / merged_data['route_count']
    X = merged_data[['route_avg_distance', 'network_length_km', 'repetition_rate', 'average_stop_spacing', 'trip_cv', 'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio', 'e_price']]
    y_vars = {'cost_per_km': merged_data['knee_cost'] / merged_data['network_length_km'], 'emission_per_km': merged_data['knee_emission'] / merged_data['network_length_km'], 'cs_per_km': merged_data['knee_built_cs'] / merged_data['network_length_km'], 'fast_chargers_per_km': merged_data['knee_fast_per_cs'] * merged_data['knee_built_cs'] / merged_data['network_length_km'], 'slow_chargers_per_km': merged_data['knee_slow_per_cs'] * merged_data['knee_built_cs'] / merged_data['network_length_km'], 'ev_per_km': (merged_data['knee_extra_large'] + merged_data['knee_extra_medium'] + merged_data['knee_extra_small'] + merged_data['vehicle_count']) / merged_data['network_length_km']}
    full_data = pd.concat([X, pd.DataFrame(y_vars)], axis=1)
    clean_data = full_data.dropna()
    X_clean = clean_data[X.columns]
    y_clean_dict = {y_name: clean_data[y_name] for y_name in y_vars.keys()}
    if standardize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_clean)
        print('Using standardized predictors for regression analysis')
    else:
        X_processed = X_clean.values
        print('Using raw predictors for regression analysis')
    X_processed = sm.add_constant(X_processed)
    feature_names = ['const'] + list(X.columns)
    if format == 'separate':
        results_table = pd.DataFrame(index=feature_names)
    else:
        results_table = pd.DataFrame(index=['R²'] + feature_names)
    model_stats = pd.DataFrame(index=['R²', 'Adj. R²', 'F-test p-value'])
    r_squared_values = {}
    for y_name, y_values in y_clean_dict.items():
        common_index = X_clean.index.intersection(y_values.index)
        X_reg = X_processed[X_clean.index.isin(common_index), :]
        y_reg = y_values[common_index].values
        model = sm.OLS(y_reg, X_reg).fit()
        r_squared_values[y_name] = model.rsquared
        if format == 'separate':
            results_table[f'{y_name}_coef'] = [model.params[0]] + list(model.params[1:])
            results_table[f'{y_name}_pval'] = [model.pvalues[0]] + list(model.pvalues[1:])
        else:
            combined_values = []
            combined_values.append(f'{model.rsquared:.4f}')
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
                combined_values.append(f'{param:.6f}{significance}')
            results_table[f'{y_name}'] = combined_values
        model_stats[y_name] = [model.rsquared, model.rsquared_adj, model.f_pvalue]
    if format == 'separate':
        print('\n=== Regression coefficient and P value ===')
        print(results_table.to_string(float_format='%.6f'))
    else:
        print('\n=== Regression coefficient (*** p<0.001, ** p<0.01, * p<0.05) ===')
        print(results_table.to_string(float_format='%.6f'))
    print('\n=== Model information ===')
    print(model_stats.to_string(float_format='%.4f'))
    return (results_table, model_stats)

def analyze_vehicle_choice_anova(indicators, outputs):
    merged_data = pd.merge(indicators, outputs, left_on='city', right_on='city_name')
    merged_data['trip_cv'] = np.sqrt(merged_data['variance']) / merged_data['trip_number'] * 24
    merged_data['route_avg_operation_distance'] = merged_data['operation_distance'] / merged_data['route_count']
    X_vars = ['route_avg_distance', 'network_length_km', 'repetition_rate', 'average_stop_spacing', 'trip_cv', 'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio', 'e_price']
    X = merged_data[X_vars]
    vehicle_types = {'large_model': merged_data['large_model'], 'medium_model': merged_data['medium_model'], 'small_model': merged_data['small_model']}
    anova_results = {}
    for vehicle_type, y_values in vehicle_types.items():
        print(f'\n=== {vehicle_type} ANOVA ===')
        analysis_data = pd.concat([X, y_values], axis=1)
        analysis_data.columns = X_vars + [vehicle_type]
        clean_data = analysis_data.dropna()
        if len(clean_data) == 0:
            print(f'Warning: {vehicle_type} has insufficient data for analysis')
            continue
        y_clean = clean_data[vehicle_type]
        X_clean = clean_data[X_vars]
        factor_results = {}
        for factor in X_vars:
            groups = [group[factor].values for name, group in clean_data.groupby(vehicle_type)]
            if len(groups) > 1 and all((len(group) > 0 for group in groups)):
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    factor_results[factor] = {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
                except Exception as e:
                    factor_results[factor] = {'f_statistic': np.nan, 'p_value': np.nan, 'significant': False}
            else:
                factor_results[factor] = {'f_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        results_df = pd.DataFrame(factor_results).T
        results_df = results_df.sort_values('p_value')

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
        significant_factors = results_df[results_df['significant']].copy()
        if not significant_factors.empty:
            print(f'\n对{vehicle_type} choice:')
            significant_display = significant_factors[['f_statistic', 'p_value', 'significance']].copy()
            significant_display['f_statistic'] = significant_display['f_statistic'].map('{:.4f}'.format)
            significant_display['p_value'] = significant_display['p_value'].map('{:.6f}'.format)
            print(significant_display.to_string())
            print(f'\n{vehicle_type} significant factors by selection category:')
            for factor in significant_factors.index:
                print(f'\n{factor}:')
                group_stats = clean_data.groupby(vehicle_type)[factor].agg(['mean', 'std', 'count'])
                print(group_stats.round(4).to_string())
        else:
            print(f'\n{vehicle_type} choice is not found')
        anova_results[vehicle_type] = results_df
    return anova_results
if __name__ == '__main__':
    indicator_data = pd.read_csv('../data/224city_indicators.csv')
    output_data = pd.read_csv('../data/224cities_output.csv')

    print('\n=== correlation_matrix ===')
    plot_correlation_matrix(indicator_data, output_data)

    print("\n=== VIF ===")
    vif_result = calculate_vif(indicator_data)
    print(vif_result.to_string(index=False))

    print('\n=== OLS ===')
    regression_results, regression_stats = perform_ols_regression(indicator_data, output_data, standardize=False, format='combined')
    regression_results.to_excel('fig_1/ols.xlsx')

    print('\n=== ANOVA ===')
    anova_results = analyze_vehicle_choice_anova(indicator_data, output_data)
