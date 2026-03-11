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
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'amosa4CN'))
    from amosa4CN.vehicle_type import VehicleTypes
    os.makedirs(output_dir, exist_ok=True)
    output_data['electricity'] = 0
    for i in range(len(output_data)):
        city_name = output_data['city_name'][i]
        e_price = static_data[static_data['city'] == city_name]['e_price'].iloc[0]
        vs_parking_df = pd.read_csv(f'../data/input/vs_parking_nodeid/{city_name}.csv')
        unique_vehicles = vs_parking_df.drop_duplicates(subset=['v_name'])
        vehicle_counts = unique_vehicles['vehicle_type'].value_counts()
        counts = vehicle_counts.reindex(['large', 'medium', 'small'], fill_value=0)
        large_count = counts['large'] + output_data['knee_extra_large'][i]
        medium_count = counts['medium'] + output_data['knee_extra_medium'][i]
        small_count = counts['small'] + output_data['knee_extra_small'][i]
        trip_cost = output_data['knee_cost'][i] * 1000000
        trip_cost -= large_count * VehicleTypes(output_data.loc[i, f'large_model'], 'large').fix_cost
        trip_cost -= medium_count * VehicleTypes(output_data.loc[i, f'medium_model'], 'medium').fix_cost
        trip_cost -= small_count * VehicleTypes(output_data.loc[i, f'small_model'], 'small').fix_cost
        trip_cost -= output_data['knee_built_cs'][i] * 600000
        trip_cost -= output_data['knee_built_cs'][i] * (output_data['knee_slow_per_cs'][i] * 2000 + output_data['knee_fast_per_cs'][i] * 4000)
        output_data.loc[i, 'electricity'] = trip_cost / e_price
    merged_df = pd.merge(output_data, city_data, left_on='city_name', right_on='city')
    merged_df['cost_to_budget_ratio'] = merged_df['knee_cost'] / merged_df['budget'] / 1000.0 * 100
    merged_df['electricity_ratio'] = merged_df['electricity'] / (merged_df['ind_electricity'] * 1000000000.0) * 100
    target_cols = ['cost_to_budget_ratio', 'electricity_ratio']
    titles = {'cost_to_budget_ratio': 'Cost-to-Budget\nRatio (%)', 'electricity_ratio': 'Electricity\nRatio (%)'}
    analysis_df = merged_df[['city_name'] + target_cols].copy()
    print('=== Cost-to-Budget Ratio Statistics ===')
    print(f"Mean: {merged_df['cost_to_budget_ratio'].mean():.4f}")
    print(f"Min: {merged_df['cost_to_budget_ratio'].min():.4f}")
    print(f"Max: {merged_df['cost_to_budget_ratio'].max():.4f}")
    print('\n=== Electricity Ratio Statistics ===')
    print(f"Mean: {merged_df['electricity_ratio'].mean():.4f}")
    print(f"Min: {merged_df['electricity_ratio'].min():.4f}")
    print(f"Max: {merged_df['electricity_ratio'].max():.4f}")
    melted_df = pd.melt(analysis_df, id_vars='city_name', value_vars=target_cols, var_name='Indicator', value_name='Value')
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.linewidth': 1.6, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 8, 'ytick.major.size': 8, 'xtick.minor.size': 4, 'ytick.minor.size': 4})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    colors = ['#377eb8', '#ff7f00']
    legend_handles = []
    legend_labels = ['Median', 'Mean']
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        current_data = melted_df[melted_df['Indicator'] == col]
        current_data = current_data.sort_values('Value')
        box = sns.boxplot(y='Value', data=current_data, ax=ax, orient='v', width=0.4, fliersize=4, showmeans=False, color=colors[i], boxprops=dict(linewidth=2, edgecolor='black', facecolor=colors[i] + '80'), medianprops=dict(color='black', linewidth=2.4, linestyle='--'), whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2))
        mean_value = current_data['Value'].mean()
        mean_line = ax.hlines(y=mean_value, xmin=-0.2, xmax=0.2, color='gray', linestyle='--', linewidth=2.4)
        hidden_line = ax.hlines(y=mean_value * 1.1, xmin=-0.4, xmax=0.4, color='none', linewidth=0)
        if i == 0:
            from matplotlib.lines import Line2D
            median_line = Line2D([0], [0], color='black', linewidth=2.4, linestyle='--')
            mean_line_legend = Line2D([0], [0], color='gray', linewidth=2.4, linestyle='--')
            legend_handles = [median_line, mean_line_legend]
        ax.set_title(titles[col], fontsize=20, pad=20)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Value', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
    axes[-1].legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)
    plt.savefig(f'{output_dir}/budget_electricity_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_route_count_vs_population_gdp(indicator_data, city_data, output_dir='fig_1'):
    os.makedirs(output_dir, exist_ok=True)
    merged_df = pd.merge(indicator_data, city_data, left_on='city', right_on='city')
    population = merged_df['district_pop']
    gdp = merged_df['district_gdp'] * 1000
    route_count = merged_df['route_count']
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.linewidth': 1.6, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 8, 'ytick.major.size': 8, 'xtick.minor.size': 4, 'ytick.minor.size': 4})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    scatter_color = '#377eb8'
    regression_color = '#e41a1c'
    ax1 = axes[0]
    ax1.scatter(population, route_count, color=scatter_color, alpha=0.7, s=60, edgecolors='black', linewidth=0.6)
    slope, intercept, r_value, p_value, std_err = stats.linregress(population, route_count)
    pop_range = np.linspace(population.min(), population.max(), 100)
    regression_line = slope * pop_range + intercept
    ax1.plot(pop_range, regression_line, color=regression_color, linewidth=3, linestyle='-')
    X_with_const = sm.add_constant(population)
    model = sm.OLS(route_count, X_with_const).fit()
    predictions = model.get_prediction(sm.add_constant(pop_range))
    conf_int = predictions.conf_int()
    ax1.fill_between(pop_range, conf_int[:, 0], conf_int[:, 1], color=regression_color, alpha=0.2, linewidth=0)
    ax1.text(0.05, 0.95, f'R² = {r_value ** 2:.2f}\np = {p_value:.3f}', transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.set_xlabel('Population (10⁴ people)', fontsize=18)
    ax1.set_ylabel('Route Count', fontsize=18)
    ax1.set_title('Route Count vs Population', fontsize=20, pad=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, linestyle=':', alpha=0.5, linewidth=1)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.6)
    ax2 = axes[1]
    ax2.scatter(gdp, route_count, color=scatter_color, alpha=0.7, s=60, edgecolors='black', linewidth=0.6)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(gdp, route_count)
    gdp_range = np.linspace(gdp.min(), gdp.max(), 100)
    regression_line2 = slope2 * gdp_range + intercept2
    ax2.plot(gdp_range, regression_line2, color=regression_color, linewidth=3, linestyle='-')
    X_with_const2 = sm.add_constant(gdp)
    model2 = sm.OLS(route_count, X_with_const2).fit()
    predictions2 = model2.get_prediction(sm.add_constant(gdp_range))
    conf_int2 = predictions2.conf_int()
    ax2.fill_between(gdp_range, conf_int2[:, 0], conf_int2[:, 1], color=regression_color, alpha=0.2, linewidth=0)
    ax2.text(0.05, 0.95, f'R² = {r_value2 ** 2:.2f}\np = {p_value2:.3f}', transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax2.set_xlabel('GDP (billion yuan)', fontsize=18)
    ax2.set_ylabel('Route Count', fontsize=18)
    ax2.set_title('Route Count vs GDP', fontsize=20, pad=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(True, linestyle=':', alpha=0.5, linewidth=1)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.6)
    plt.savefig(f'{output_dir}/route_count_population_gdp_scatter.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_operation_indicators_boxplot(indicator_data, output_dir='fig_1'):
    os.makedirs(output_dir, exist_ok=True)
    target_cols = ['route_count', 'route_avg_distance', 'route_avg_trip']
    titles = {'route_count': 'Route Count', 'route_avg_distance': 'Average Route\nDistance (km)', 'route_avg_trip': 'Average Trips\nper Route (per day)'}
    analysis_df = indicator_data[['city', 'route_count', 'route_avg_distance', 'trip_number']].copy()
    analysis_df['route_avg_trip'] = analysis_df['trip_number'] / analysis_df['route_count']
    analysis_df = analysis_df[['city'] + target_cols]
    melted_df = pd.melt(analysis_df, id_vars='city', value_vars=target_cols, var_name='Indicator', value_name='Value')
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.linewidth': 1.6, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 8, 'ytick.major.size': 8, 'xtick.minor.size': 4, 'ytick.minor.size': 4})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    colors = ['#377eb8', '#ff7f00', '#4daf4a']
    legend_handles = []
    legend_labels = ['Median', 'Mean']
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        current_data = melted_df[melted_df['Indicator'] == col]
        current_data = current_data.sort_values('Value')
        box = sns.boxplot(y='Value', data=current_data, ax=ax, orient='v', width=0.2, fliersize=4, showmeans=False, color=colors[i], boxprops=dict(linewidth=2, edgecolor='black', facecolor=colors[i] + '80'), medianprops=dict(color='black', linewidth=2.4, linestyle='--'), whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2))
        mean_value = current_data['Value'].mean()
        q1 = current_data['Value'].quantile(0.25)
        q3 = current_data['Value'].quantile(0.75)
        mean_line = ax.hlines(y=mean_value, xmin=-0.1, xmax=0.1, color='gray', linestyle='--', linewidth=2.4)
        hidden_line = ax.hlines(y=mean_value * 1.1, xmin=-0.16, xmax=0.16, color='none', linewidth=0)
        if i == 0:
            from matplotlib.lines import Line2D
            median_line = Line2D([0], [0], color='black', linewidth=2.4, linestyle='--')
            mean_line_legend = Line2D([0], [0], color='gray', linewidth=2.4, linestyle='--')
            legend_handles = [median_line, mean_line_legend]
        ax.set_title(titles[col], fontsize=20, pad=20)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Value', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
    axes[-1].legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)
    plt.savefig(f'{output_dir}/basic_indicators_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_electrification_indicators_boxplot(indicator_data, output_data, output_dir='fig_1'):
    os.makedirs(output_dir, exist_ok=True)
    analysis_df = pd.merge(indicator_data, output_data, left_on='city', right_on='city_name')
    analysis_df['knee_cost_per_km'] = analysis_df['knee_cost'] / analysis_df['network_length_km']
    analysis_df['knee_emission_per_km'] = analysis_df['knee_emission'] / analysis_df['network_length_km']
    analysis_df['knee_cs_per_km'] = analysis_df['knee_built_cs'] / analysis_df['network_length_km']
    analysis_df['knee_charger_per_km'] = (analysis_df['knee_fast_per_cs'] + analysis_df['knee_slow_per_cs']) * analysis_df['knee_built_cs'] / analysis_df['network_length_km']
    analysis_df['knee_ev_per_km'] = (analysis_df['vehicle_count'] + analysis_df['knee_extra_large'] + analysis_df['knee_extra_medium'] + analysis_df['knee_extra_small']) / analysis_df['network_length_km']
    target_cols = ['knee_cost_per_km', 'knee_emission_per_km', 'knee_cs_per_km', 'knee_charger_per_km', 'knee_ev_per_km']
    titles = {'knee_cost_per_km': 'System Costs\nper km (M yuan/km)', 'knee_emission_per_km': 'GHG Emissions\nper km (T/km)', 'knee_cs_per_km': 'Charging Stations\nper km (km$^{-1}$)', 'knee_charger_per_km': 'Chargers\nper km (km$^{-1}$)', 'knee_ev_per_km': 'Electric Buses\nper km (km$^{-1}$)'}
    analysis_df = analysis_df[['city_name'] + target_cols].copy()
    melted_df = pd.melt(analysis_df, id_vars='city_name', value_vars=target_cols, var_name='Indicator', value_name='Value')
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.linewidth': 1.6, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 8, 'ytick.major.size': 8, 'xtick.minor.size': 4, 'ytick.minor.size': 4})
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes = axes.flatten()
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c']
    legend_handles = []
    legend_labels = ['Median', 'Mean']
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        current_data = melted_df[melted_df['Indicator'] == col]
        current_data = current_data.sort_values('Value')
        box = sns.boxplot(y='Value', data=current_data, ax=ax, orient='v', width=0.6, fliersize=6, showmeans=False, color=colors[i], boxprops=dict(linewidth=2, edgecolor='black', facecolor=colors[i] + '80'), medianprops=dict(color='black', linewidth=2.4, linestyle='--'), whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2))
        mean_value = current_data['Value'].mean()
        mean_line = ax.hlines(y=mean_value, xmin=-0.3, xmax=0.3, color='gray', linestyle='--', linewidth=2.4)
        hidden_line = ax.hlines(y=mean_value * 1.1, xmin=-0.6, xmax=0.6, color='none', linewidth=0)
        if i == 0:
            from matplotlib.lines import Line2D
            median_line = Line2D([0], [0], color='black', linewidth=2.4, linestyle='--')
            mean_line_legend = Line2D([0], [0], color='gray', linewidth=2.4, linestyle='--')
            legend_handles = [median_line, mean_line_legend]
        ax.set_title(titles[col], fontsize=20, pad=20)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Value', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
    axes[-1].legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)
    plt.savefig(f'{output_dir}/electrification_indicators_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
if __name__ == '__main__':
    # Load data
    static_data = pd.read_csv(r'../data/224city_indicators.csv')
    city_data = pd.read_csv(r'../data/224cities.csv')
    output_data = pd.read_csv(r'../data/224cities_output.csv')

    # Plot operation-related indicators
    plot_operation_indicators_boxplot(static_data)

    # Plot electrification-related indicators
    plot_electrification_indicators_boxplot(static_data, output_data)

    # Plot the relationship between route count, population, and GDP
    plot_route_count_vs_population_gdp(static_data, city_data)

    # Plot the relationship between budget and industrial electricity consumption
    plot_budget_vs_ind_electricity(city_data, output_data, static_data)

