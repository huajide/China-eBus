import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from matplotlib.ticker import PercentFormatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'amosa4CN'))
from amosa4CN.vehicle_type import VehicleTypes

def plot_elbow_method(data):
    X = data[['network_length_km']].values
    k_range = range(1, 21)
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Suggested elbow point')
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
    return (k_range, inertias)

def perform_kmeans_and_stats(data, n_clusters=4):
    X = data[['network_length_km']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    result_df = pd.DataFrame({'city': data['city'], 'network_length_km': data['network_length_km'], 'group': cluster_labels})
    group_stats = result_df.groupby('group')['network_length_km'].mean().sort_values()
    group_mapping = {old_group: new_group for new_group, old_group in enumerate(group_stats.index, 1)}
    result_df['group'] = result_df['group'].map(group_mapping)
    print('Group statistics:')
    print('=' * 60)
    stats_data = []
    for i in range(1, n_clusters + 1):
        group_data = result_df[result_df['group'] == i]['network_length_km']
        stats_data.append({'Group': i, 'Min (km)': f'{group_data.min():.2f}', 'Max (km)': f'{group_data.max():.2f}', 'Mean (km)': f'{group_data.mean():.2f}', 'Count': len(group_data)})
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    return result_df

def plot_clustered_indicators_bar(clustered_data, output_data, static_data):
    merged_data = pd.merge(output_data, static_data, left_on='city_name', right_on='city')
    merged_data = pd.merge(merged_data, clustered_data[['city', 'group']], on='city')
    merged_data['electricity'] = 0
    for i in range(len(merged_data)):
        city_name = merged_data['city_name'].iloc[i]
        e_price = static_data[static_data['city'] == city_name]['e_price'].iloc[0]
        try:
            vs_parking_df = pd.read_csv(f'../data/input/vs_parking_nodeid/{city_name}.csv')
            unique_vehicles = vs_parking_df.drop_duplicates(subset=['v_name'])
            vehicle_counts = unique_vehicles['vehicle_type'].value_counts()
            counts = vehicle_counts.reindex(['large', 'medium', 'small'], fill_value=0)
            large_count = counts['large'] + merged_data['knee_extra_large'].iloc[i]
            medium_count = counts['medium'] + merged_data['knee_extra_medium'].iloc[i]
            small_count = counts['small'] + merged_data['knee_extra_small'].iloc[i]
            trip_cost = merged_data['knee_cost'].iloc[i] * 1000000
            trip_cost -= large_count * VehicleTypes(merged_data.loc[merged_data.index[i], 'large_model'], 'large').fix_cost
            trip_cost -= medium_count * VehicleTypes(merged_data.loc[merged_data.index[i], 'medium_model'], 'medium').fix_cost
            trip_cost -= small_count * VehicleTypes(merged_data.loc[merged_data.index[i], 'small_model'], 'small').fix_cost
            trip_cost -= merged_data['knee_built_cs'].iloc[i] * 600000
            trip_cost -= merged_data['knee_built_cs'].iloc[i] * (merged_data['knee_slow_per_cs'].iloc[i] * 2000 + merged_data['knee_fast_per_cs'].iloc[i] * 4000)
            merged_data.loc[merged_data.index[i], 'electricity'] = trip_cost / e_price
        except FileNotFoundError:
            merged_data.loc[merged_data.index[i], 'electricity'] = 0
    merged_data['cost_per_km'] = merged_data['knee_cost'] / merged_data['network_length_km']
    merged_data['emission_per_km'] = merged_data['knee_emission'] / merged_data['network_length_km']
    merged_data['cs_per_km'] = merged_data['knee_built_cs'] / merged_data['network_length_km']
    merged_data['charger_per_km'] = (merged_data['knee_fast_per_cs'] + merged_data['knee_slow_per_cs']) * merged_data['knee_built_cs'] / merged_data['network_length_km']
    merged_data['ev_per_km'] = (merged_data['vehicle_count'] + merged_data['knee_extra_large'] + merged_data['knee_extra_medium'] + merged_data['knee_extra_small']) / merged_data['network_length_km']
    merged_data['electricity_per_km'] = merged_data['electricity'] / merged_data['network_length_km']
    avg = merged_data.groupby('group')['cost_per_km'].mean()
    print(f'Medium is {(avg[3] - avg[2]) / avg[3] * 100:.1f}% lower than Large and {(avg[1] - avg[2]) / avg[1] * 100:.1f}% lower than Small.')
    indicators = ['cost_per_km', 'emission_per_km', 'cs_per_km', 'charger_per_km', 'ev_per_km', 'electricity_per_km']
    indicator_titles = {'cost_per_km': 'System Costs\nper km\n(M yuan/km)', 'emission_per_km': 'GHG Emissions\nper km\n(T/km)', 'cs_per_km': 'Charging Stations\nper km\n(km$^{-1}$)', 'charger_per_km': 'Chargers\nper km\n(km$^{-1}$)', 'ev_per_km': 'Electric Buses\nper km\n(km$^{-1}$)', 'electricity_per_km': 'Electricity\nConsumption\nper km\n(kWh/km)'}
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.linewidth': 1.6, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 8, 'ytick.major.size': 8, 'xtick.minor.size': 4, 'ytick.minor.size': 4})
    fig, axes = plt.subplots(1, 6, figsize=(22, 5))
    axes = axes.flatten()
    colors = ['#c6dbef', '#6baed6', '#1f77b4']
    for idx, (indicator, title) in enumerate(indicator_titles.items()):
        ax = axes[idx]
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
        box_plot = ax.boxplot(group_data, labels=group_labels, patch_artist=True, widths=0.4, boxprops=dict(linewidth=1.5), medianprops=dict(linewidth=1.5, color='black'), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1), flierprops=dict(marker='o', markersize=3, alpha=0.8))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12, rotation=45)
    fig.text(0.5, 0.02, 'Bus Network', ha='center', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('fig_3/clustered_indicators_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return merged_data

def plot_clustered_cities_map(clustered_data, save_path='fig_3/clustered_cities_map.png'):
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from china_cities import load_and_prepare_data, _plot_base_map, _plot_inset_map
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)
    colors = ['#c6dbef', '#6baed6', '#1f77b4']
    _plot_base_map(ax_main, cities, province_boundaries, country_boundary, show_province_colors=False, city_color='#f0f0f0')
    for group_id, color in zip([1, 2, 3], colors):
        group_cities = clustered_data[clustered_data['group'] == group_id]
        if not group_cities.empty:
            matched_cities = cities[cities['city'].isin(group_cities['city'])]
            if not matched_cities.empty:
                matched_cities.plot(ax=ax_main, color=color, edgecolor='white', linewidth=0.5, alpha=1.0)
    country_boundary.plot(ax=ax_main, facecolor='none', edgecolor='black', linewidth=1.2)
    province_boundaries.plot(ax=ax_main, facecolor='none', edgecolor='#333333', linewidth=0.8)
    ax_main.set_ylim(bottom=1900000.0, top=6120000.0)
    ax_main.set_xlim(right=2500000.0)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5, alpha=1.0, label=label) for color, label in zip(colors, ['Small', 'Medium', 'Large'])]
    legend_title = 'Bus Network Scale'
    legend = ax_main.legend(handles=legend_elements, title=legend_title, loc='lower left', fontsize=12, frameon=True, title_fontsize=13)
    legend.get_title().set_fontweight('bold')
    ax_inset = fig.add_axes([0.78, 0.21, 0.11, 0.19])
    _plot_inset_map(ax_inset, cities, province_boundaries, country_boundary, nine_dash_line, show_province_colors=False, city_color='#f0f0f0')
    for group_id, color in zip([1, 2, 3], colors):
        group_cities = clustered_data[clustered_data['group'] == group_id]
        if not group_cities.empty:
            matched_cities = cities[cities['city'].isin(group_cities['city'])]
            if not matched_cities.empty:
                matched_cities.plot(ax=ax_inset, color=color, edgecolor='white', linewidth=0.3, alpha=1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)
    plt.close()

def plot_scenario_changes_by_group(clustered_data, output_data, static_data, save_path='fig_3/scenario_changes_by_group.png'):
    scenarios = {'baseline': '../data/output/mosa/251026', 'EP0.5': '../data/output/mosa/what_if/251026_EP0.5', 'EP0.75': '../data/output/mosa/what_if/251026_EP0.75', 'EP1.25': '../data/output/mosa/what_if/251026_EP1.25', 'EP1.5': '../data/output/mosa/what_if/251026_EP1.5', 'FCS2': '../data/output/mosa/what_if/251026_FCS2', 'FCS3': '../data/output/mosa/what_if/251026_FCS3', 'FCS4': '../data/output/mosa/what_if/251026_FCS4', 'FCS5': '../data/output/mosa/what_if/251026_FCS5', 'BC1.25': '../data/output/mosa/what_if/251026_BC1.25', 'BC1.5': '../data/output/mosa/what_if/251026_BC1.5', 'BC1.75': '../data/output/mosa/what_if/251026_BC1.75', 'BC2': '../data/output/mosa/what_if/251026_BC2', 'VC0.6': '../data/output/mosa/what_if/251026_VC0.6', 'VC0.7': '../data/output/mosa/what_if/251026_VC0.7', 'VC0.8': '../data/output/mosa/what_if/251026_VC0.8', 'VC0.9': '../data/output/mosa/what_if/251026_VC0.9'}
    scenario_groups = {'FCS': ['baseline', 'FCS2', 'FCS3', 'FCS4', 'FCS5'], 'EP': ['EP0.5', 'EP0.75', 'baseline', 'EP1.25', 'EP1.5'], 'BC': ['baseline', 'BC1.25', 'BC1.5', 'BC1.75', 'BC2'], 'VC': ['VC0.6', 'VC0.7', 'VC0.8', 'VC0.9', 'baseline']}
    network_length_dict = dict(zip(static_data['city'], static_data['network_length_km']))
    vehicle_count_dict = dict(zip(static_data['city'], static_data['vehicle_count']))

    def load_scenario_data(scenario_path):
        city_stats = []
        if not os.path.exists(scenario_path):
            print(f'Path does not exist: {scenario_path}')
            return pd.DataFrame()
        for folder in os.listdir(scenario_path):
            full_path = os.path.join(scenario_path, folder)
            if os.path.isdir(full_path):
                city_name = os.path.basename(full_path)
                if city_name:
                    try:
                        cv_file = os.path.join(full_path, 'inf_archive_cvs.csv')
                        if os.path.exists(cv_file):
                            cv_data = pd.read_csv(cv_file)
                            last_row = cv_data.iloc[-1]
                            built_num = abs(last_row[1])
                            cs_num = built_num
                            if len(last_row) > 2 + cs_num:
                                fast_chargers = [abs(x) for x in last_row[2:2 + int(cs_num)] if pd.notna(x)]
                                fast_charger_total = sum(fast_chargers)
                                slow_chargers = [abs(x) for x in last_row[2 + int(cs_num):2 + 2 * int(cs_num)] if pd.notna(x)]
                                slow_charger_total = sum(slow_chargers)
                            else:
                                fast_charger_total = 0
                                slow_charger_total = 0
                            additional_vehicles = 0
                            if len(last_row) >= 3:
                                additional_vehicles = sum([abs(x) for x in last_row[-3:] if pd.notna(x)])
                            obj_file = os.path.join(full_path, 'inf_archive_objs.csv')
                            if os.path.exists(obj_file):
                                obj_data = pd.read_csv(obj_file)
                                cost = obj_data.iloc[-1, 1]
                                emission = obj_data.iloc[-1, 2]
                                network_length = network_length_dict.get(city_name, 1)
                                static_vehicle_count = vehicle_count_dict.get(city_name, 0)
                                actual_vehicle_count = static_vehicle_count + additional_vehicles
                                city_stats.append({'city': city_name, 'cost': -cost / network_length, 'emission': -emission / network_length, 'built_num': built_num / network_length, 'fast_charger': fast_charger_total / network_length, 'slow_charger': slow_charger_total / network_length, 'vehicle_count': actual_vehicle_count / network_length, 'network_length_km': network_length})
                    except Exception as e:
                        print(f'Error when processing {city_name}: {e}')
        return pd.DataFrame(city_stats)
    scenario_data = {}
    for scenario, path in scenarios.items():
        scenario_data[scenario] = load_scenario_data(path)
    common_cities = None
    for scenario in scenario_data:
        if len(scenario_data[scenario]) > 0:
            if common_cities is None:
                common_cities = set(scenario_data[scenario]['city'])
            else:
                common_cities = common_cities.intersection(set(scenario_data[scenario]['city']))
    if common_cities is None:
        print('No common city data found')
        return None
    common_cities = list(common_cities)
    print(f'Find {len(common_cities)} common cities')
    aligned_scenario_data = {}
    for scenario in scenario_data:
        if len(scenario_data[scenario]) > 0:
            aligned_data = []
            for city in common_cities:
                city_data = scenario_data[scenario][scenario_data[scenario]['city'] == city]
                if len(city_data) > 0:
                    aligned_data.append(city_data.iloc[0])
            aligned_scenario_data[scenario] = pd.DataFrame(aligned_data)
    city_relative_data = {}
    baseline_city_data = aligned_scenario_data.get('baseline', pd.DataFrame())
    if baseline_city_data.empty:
        print('Baseline data is empty')
        return None
    for scenario in aligned_scenario_data.keys():
        city_relative_data[scenario] = {}
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
    baseline_with_common_cities = baseline_city_data[baseline_city_data['city'].isin(common_cities)]
    baseline_with_group = pd.merge(baseline_with_common_cities, clustered_data[['city', 'group']], on='city')
    city_to_index = {city: idx for idx, city in enumerate(common_cities)}
    group_median_changes = {}
    for group_name, scenarios_list in scenario_groups.items():
        group_median_changes[group_name] = {}
        for metric in ['cost', 'emission', 'built_num', 'fast_charger', 'slow_charger', 'vehicle_count']:
            group_median_changes[group_name][metric] = {}
            for group_id in [1, 2, 3]:
                group_median_changes[group_name][metric][group_id] = []
                group_cities = set(baseline_with_group[baseline_with_group['group'] == group_id]['city'].tolist())
                for scenario in scenarios_list:
                    changes = []
                    for city in group_cities:
                        if city in city_to_index and scenario in city_relative_data:
                            city_idx = city_to_index[city]
                            if city_idx < len(city_relative_data[scenario][metric]):
                                changes.append(city_relative_data[scenario][metric][city_idx])
                    median_change = np.median(changes) if changes else 0
                    group_median_changes[group_name][metric][group_id].append(median_change)
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5, 'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 3, 'ytick.minor.size': 3})
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    metrics = ['cost', 'emission', 'built_num', 'fast_charger', 'slow_charger', 'vehicle_count']
    metric_names = ['Cost', 'Emission', 'Charging Stations', 'Fast Chargers', 'Slow Chargers', 'Electric Vehicles']
    units = ['M yuan/(year·km)', 'T/(year·km)', 'km$^{-1}$', 'km$^{-1}$', 'km$^{-1}$', 'km$^{-1}$']
    group_colors = ['#c6dbef', '#6baed6', '#1f77b4']
    row_labels = ['Fast charging speed', 'Electricity price', 'Battery capacity', 'Vehicle cost']
    for row, (group_name, scenarios_list) in enumerate(scenario_groups.items()):
        x_positions = list(range(len(scenarios_list)))
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
            for group_id, color in zip([1, 2, 3], group_colors):
                y_values = group_median_changes[group_name][metric][group_id]
                ax.plot(x_positions, y_values, marker='o', linewidth=1.5, markersize=4, color=color, label=f'Group {group_id}')
            ax.set_title(f'{metric_name}', fontsize=14, pad=10)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.yaxis.set_major_formatter(PercentFormatter())
            ax.grid(True, linestyle=':', alpha=0.6, linewidth=1)
            y_ranges = {'cost': (-30, 30), 'emission': (-5, 5), 'built_num': (-15, 15), 'fast_charger': (-100, 10), 'slow_charger': (-70, 10), 'vehicle_count': (-3, 1)}
            if metric in y_ranges:
                ax.set_ylim(y_ranges[metric])
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            if col == 0:
                ax.text(-0.3, 0.5, row_labels[row], transform=ax.transAxes, fontsize=13, fontweight='bold', rotation=90, verticalalignment='center', horizontalalignment='center')
            if row == 0 and col == 0:
                legend_elements = [plt.Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markersize=8, label=label, linewidth=2, markeredgecolor='w', markeredgewidth=0.5) for color, label in zip(group_colors, ['Small', 'Medium', 'Large'])]
                legend = ax.legend(handles=legend_elements, title='Bus Network Scale', loc='upper left', bbox_to_anchor=(-0.5, 1.5), ncol=3, frameon=False, fontsize=12)
                legend.get_title().set_fontweight('bold')
                legend.get_title().set_fontsize(12)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show(block=True)
    plt.close()
    return group_median_changes
if __name__ == '__main__':
    static_data = pd.read_csv("../data/224city_indicators.csv")
    output_data = pd.read_csv("../data/224cities_output.csv")

    '''Call the function to plot the elbow curve'''
    k_values, inertia_values = plot_elbow_method(static_data)

    '''Run clustering and summary statistics with k=4'''
    clustered_data = perform_kmeans_and_stats(static_data, 4)

    '''Merge the two largest groups (group 4 and group 3) into one new group'''
    # Reassign all cities in group 4 to group 3
    mask_group4 = clustered_data['group'] == 4
    clustered_data.loc[mask_group4, 'group'] = 3

    '''1. Plot the clustered city map'''
    plot_clustered_cities_map(clustered_data)

    '''2. Calculate per-km indicators and plot the bar chart'''
    # Call the function to plot the bar chart
    group_averages = plot_clustered_indicators_bar(clustered_data, output_data, static_data)

    '''3. Plot scenario changes by cluster group'''
    plot_scenario_changes_by_group(clustered_data, output_data, static_data)


