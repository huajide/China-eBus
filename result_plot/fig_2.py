import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'amosa4CN', 'TestingResults'))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial', 'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 6, 'ytick.major.size': 6})
scenarios = {'baseline': '../data/output/mosa/251026', 'EP0.5': '../data/output/mosa/what_if/251026_EP0.5', 'EP0.75': '../data/output/mosa/what_if/251026_EP0.75', 'EP1.25': '../data/output/mosa/what_if/251026_EP1.25', 'EP1.5': '../data/output/mosa/what_if/251026_EP1.5', 'FCS2': '../data/output/mosa/what_if/251026_FCS2', 'FCS3': '../data/output/mosa/what_if/251026_FCS3', 'FCS4': '../data/output/mosa/what_if/251026_FCS4', 'FCS5': '../data/output/mosa/what_if/251026_FCS5', 'BC1.25': '../data/output/mosa/what_if/251026_BC1.25', 'BC1.5': '../data/output/mosa/what_if/251026_BC1.5', 'BC1.75': '../data/output/mosa/what_if/251026_BC1.75', 'BC2': '../data/output/mosa/what_if/251026_BC2', 'VC0.6': '../data/output/mosa/what_if/251026_VC0.6', 'VC0.7': '../data/output/mosa/what_if/251026_VC0.7', 'VC0.8': '../data/output/mosa/what_if/251026_VC0.8', 'VC0.9': '../data/output/mosa/what_if/251026_VC0.9'}
fcs_scenarios = ['baseline', 'FCS2', 'FCS3', 'FCS4', 'FCS5']
ep_scenarios = ['EP0.5', 'EP0.75', 'baseline', 'EP1.25', 'EP1.5']
bc_scenarios = ['baseline', 'BC1.25', 'BC1.5', 'BC1.75', 'BC2']
vc_scenarios = ['VC0.6', 'VC0.7', 'VC0.8', 'VC0.9', 'baseline']

def extract_city_name(folder_name, root_name):
    if folder_name.endswith(f'_{root_name}'):
        return folder_name[:-len(f'_{root_name}')]
    return None
static_data = pd.read_csv('../data/224city_indicators.csv')
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
fig, axes = plt.subplots(4, 6, figsize=(15, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
metrics = ['cost', 'emission', 'built_num', 'fast_charger', 'slow_charger', 'vehicle_count']
metric_names = ['Cost', 'Emission', 'Charging Stations', 'Fast Chargers', 'Slow Chargers', 'Electric Vehicles']
units = ['M yuan/(year·km)', 'T/(year·km)', 'km$^{-1}$', 'km$^{-1}$', 'km$^{-1}$', 'km$^{-1}$']
city_relative_data = {}
baseline_city_data = scenario_data['baseline']
for scenario in scenario_data.keys():
    city_relative_data[scenario] = {}
    if scenario == 'baseline':
        for metric in metrics:
            city_relative_data[scenario][metric] = [0.0] * len(baseline_city_data)
    else:
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
colors = {'fcs': '#1f77b4', 'ep': '#ff7f0e', 'bc': '#2ca02c', 'vc': '#d62728'}
row_labels_with_units = ['Fast charging speed', 'Electricity price', 'Battery capacity', 'Vehicle cost']
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[0, j]
    fcs_box_data = []
    fcs_positions = []
    for i, scenario in enumerate(fcs_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            fcs_box_data.append(city_relative_data[scenario][metric])
            fcs_positions.append(i + 1)
    if fcs_box_data:
        bp = ax.boxplot(fcs_box_data, positions=fcs_positions, widths=0.6, patch_artist=True, notch=False, showfliers=False, boxprops=dict(facecolor=colors['fcs'], alpha=0.7, linewidth=1), medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(fcs_positions)
    ax.set_xticklabels(['1x\n(baseline)', '2x', '3x', '4x', '5x'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if j == 0:
        ax.text(-0.5, 0.5, row_labels_with_units[0].split('\n')[0], transform=ax.transAxes, fontsize=12, fontweight='bold', rotation=90, verticalalignment='center', horizontalalignment='center')
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[1, j]
    ep_box_data = []
    ep_positions = []
    for i, scenario in enumerate(ep_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            ep_box_data.append(city_relative_data[scenario][metric])
            ep_positions.append(i + 1)
    if ep_box_data:
        bp = ax.boxplot(ep_box_data, positions=ep_positions, widths=0.6, patch_artist=True, notch=False, showfliers=False, boxprops=dict(facecolor=colors['ep'], alpha=0.7, linewidth=1), medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(ep_positions)
    ax.set_xticklabels(['0.5x', '0.75x', '1x\n(baseline)', '1.25x', '1.5x'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if j == 0:
        ax.text(-0.5, 0.5, row_labels_with_units[1].split('\n')[0], transform=ax.transAxes, fontsize=12, fontweight='bold', rotation=90, verticalalignment='center', horizontalalignment='center')
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[2, j]
    bc_box_data = []
    bc_positions = []
    for i, scenario in enumerate(bc_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            bc_box_data.append(city_relative_data[scenario][metric])
            bc_positions.append(i + 1)
    if bc_box_data:
        bp = ax.boxplot(bc_box_data, positions=bc_positions, widths=0.6, patch_artist=True, notch=False, showfliers=False, boxprops=dict(facecolor=colors['bc'], alpha=0.7, linewidth=1), medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(bc_positions)
    ax.set_xticklabels(['1x\n(baseline)', '1.25x', '1.5x', '1.75x', '2x'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if j == 0:
        ax.text(-0.5, 0.5, row_labels_with_units[2].split('\n')[0], transform=ax.transAxes, fontsize=12, fontweight='bold', rotation=90, verticalalignment='center', horizontalalignment='center')
for j, (metric, metric_name, unit) in enumerate(zip(metrics, metric_names, units)):
    ax = axes[3, j]
    vc_box_data = []
    vc_positions = []
    for i, scenario in enumerate(vc_scenarios):
        if scenario in city_relative_data and metric in city_relative_data[scenario]:
            vc_box_data.append(city_relative_data[scenario][metric])
            vc_positions.append(i + 1)
    if vc_box_data:
        bp = ax.boxplot(vc_box_data, positions=vc_positions, widths=0.6, patch_artist=True, notch=False, showfliers=False, boxprops=dict(facecolor=colors['vc'], alpha=0.7, linewidth=1), medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(vc_positions)
    ax.set_xticklabels(['0.6x', '0.7x', '0.8x', '0.9x', '1x\n(baseline)'], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if j == 0:
        ax.text(-0.5, 0.5, row_labels_with_units[3].split('\n')[0], transform=ax.transAxes, fontsize=12, fontweight='bold', rotation=90, verticalalignment='center', horizontalalignment='center')
plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
plt.savefig('fig_2&3/scenario_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
