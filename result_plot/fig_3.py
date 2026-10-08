from __future__ import annotations

import argparse
from pathlib import Path

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
import fig_2
from fig_2 import find_knee_solution,row_by_solution
from fig_1 import _adjust_readability

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = DATA_DIR / "output" / "results"
FIGURES_ROOT = DATA_DIR / "output" / "figures"

ARCHIVE_ROOT = RESULTS_ROOT / "baseline"
VS_ROOT = DATA_DIR / "input" / "vs_parking_nodeid"
GROUP_FILE = DATA_DIR / "input" / "network_groups.csv"
STATIC_FILE = DATA_DIR / "224city_indicators.csv"
OUTPUT_DIR = FIGURES_ROOT
READABLE_DIR = OUTPUT_DIR
FONT_SCALE = 1.4
CANVAS_SCALE = .96
SELECTED_FONT_SCALE = 1.25
SELECTED_CANVAS_SCALE = 1.
MAP_FONT_SCALE = 1.35
MAP_CANVAS_SCALE = .96
METRICS = (('cost_per_km', 'System costs\nper km\n(M yuan km$^{-1}$ yr$^{-1}$)'), ('emission_per_km', 'GHG emissions\nper km\n(t km$^{-1}$ yr$^{-1}$)'), ('cs_per_km', 'Charging stations\nper km\n(km$^{-1}$)'), ('charger_per_km', 'Chargers\nper km\n(km$^{-1}$)'), ('ev_per_km', 'Electric buses\nper km\n(km$^{-1}$)'), ('electricity_mwh_per_km', 'Electricity\nconsumption\nper km\n(MWh/km)'))
GROUP_LABELS = {1: 'Small network', 2: 'Medium network', 3: 'Large network'}

GROUP_COLORS = {1: '#9ecae1', 2: '#3182bd', 3: '#08519c'}
PANELS = (('Charging speed', 'cost', 'System costs', (-10.2, 1.3), [0, -3, -6, -9]), ('Charging speed', 'built_num', 'Charging stations', (-18, 2.5), [0, -5, -10, -15]), ('Charging speed', 'slow_charger', 'Slow chargers', (-58, 7.5), [0, -20, -40]), ('Battery capacity', 'built_num', 'Charging stations', (-32, 4), [0, -10, -20, -30]))

LABELS = {1: 'Small', 2: 'Medium', 3: 'Large'}
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial', 'axes.linewidth': 1.35, 'xtick.major.width': 1.25, 'ytick.major.width': 1.25, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 5.5, 'ytick.major.size': 5.5})
def build_city_data() -> pd.DataFrame:
    groups = pd.read_csv(GROUP_FILE)[['city', 'network_length_km', 'group']]
    prices = pd.read_csv(STATIC_FILE).set_index('city')['e_price']
    if len(groups) != 224 or groups['city'].duplicated().any():
        raise ValueError('Expected the original 224 unique Fig. 3 city groups')
    rows = []
    for city, length, group in groups.itertuples(index=False, name=None):
        city_dir = ARCHIVE_ROOT / city
        objectives = pd.read_csv(city_dir / 'archive_objs.csv', index_col=0)
        variables = pd.read_csv(city_dir / 'archive_vars.csv', index_col=0)
        diagnostics = pd.read_csv(city_dir / 'archive_cvs.csv', index_col=0)
        solution = find_knee_solution(objectives)
        obj = row_by_solution(objectives, solution)
        var = row_by_solution(variables, solution)
        cv = row_by_solution(diagnostics, solution)
        station_count = len(var) - 3
        if len(cv) != 2 * station_count + 6 or cv.iloc[0] > 0:
            raise ValueError(f'Invalid knee diagnostics: {city}')
        base_vehicles = pd.read_csv(VS_ROOT / f'{city}.csv', usecols=['v_name'])['v_name'].nunique()
        extra_vehicles = -cv.iloc[-3:].sum()
        built_stations = var.iloc[:station_count].sum()
        chargers = -cv.iloc[1:1 + 2 * station_count].sum()
        electricity_mwh = cv.iloc[-5] * 1000 / prices.loc[city]
        rows.append({'city': city, 'cost_per_km': -obj.iloc[0] / length, 'emission_per_km': -obj.iloc[1] / length, 'cs_per_km': built_stations / length, 'charger_per_km': chargers / length, 'ev_per_km': (base_vehicles + extra_vehicles) / length, 'electricity_mwh_per_km': electricity_mwh / length, 'network_length_km': length, 'group': group})
    return pd.DataFrame(rows)

def plot(data: pd.DataFrame, readable: bool=True) -> None:
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'axes.linewidth': 1.35, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ('#c6dbef', '#6baed6', '#1f77b4')
    plot_metrics = METRICS[:5] if readable else METRICS
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(3 * len(plot_metrics), 4.3))
    for ax, (metric, title) in zip(axes, plot_metrics):
        values = [data.loc[data['group'].eq(group), metric].to_numpy() for group in (1, 2, 3)]
        boxes = ax.boxplot(values, tick_labels=('Small', 'Medium', 'Large'), patch_artist=True, widths=0.4, boxprops={'linewidth': 1.2}, medianprops={'color': 'black', 'linewidth': 1.5}, whiskerprops={'linewidth': 1}, capprops={'linewidth': 1}, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.8})
        for sample, median in zip(values, boxes['medians']):
            ax.plot(median.get_xdata(), [np.mean(sample)] * 2, color='gray',
                    linestyle='--', linewidth=median.get_linewidth(), scalex=False, scaley=False)
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=12, pad=11)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.tick_params(axis='x', labelrotation=45, labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
    if readable:
        _adjust_readability(fig, FONT_SCALE, CANVAS_SCALE)
    for ax, (_, title) in zip(axes, plot_metrics):
        ax.title.set_text(title.replace('(M yuan km$^{-1}$ yr$^{-1}$)', '(M yuan/km)').replace('(t km$^{-1}$ yr$^{-1}$)', '(T/km)'))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.3)
    fig.set_dpi(300)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(plt.rcParams['savefig.pad_inches'])
    bounds.y1 += axes[0].title.get_fontsize() / 72
    for ax, (_, title) in zip(axes, plot_metrics):
        ax.title.set_text(title)
    target = READABLE_DIR if readable else OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / 'Figure_3c.png', dpi=300, bbox_inches=bounds)
    plt.close(fig)
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

def plot_clustered_cities_map(clustered_data, save_path=None, font_scale=1.0, canvas_scale=1.0, show=True):
    """Map cities by network-size group."""
    save_path = str(save_path or OUTPUT_DIR / 'Figure_3a.png')
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from china_cities import load_and_prepare_data, _plot_base_map, _plot_inset_map
    from fig_1 import _adjust_readability
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()
    city_codes = pd.read_csv(DATA_DIR / '224cities.csv').set_index('city')['citycode']
    clustered_data = clustered_data.assign(citycode=clustered_data['city'].map(city_codes))
    if clustered_data['citycode'].isna().any():
        raise ValueError('Network groups contain unknown city names')
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)
    colors = ['#c6dbef', '#6baed6', '#1f77b4']
    _plot_base_map(ax_main, cities, province_boundaries, country_boundary, show_province_colors=False, city_color='#f0f0f0')
    for group_id, color in zip([1, 2, 3], colors):
        group_cities = clustered_data[clustered_data['group'] == group_id]
        if not group_cities.empty:
            matched_cities = cities[cities['市代码'].isin(group_cities['citycode'])]
            if not matched_cities.empty:
                matched_cities.plot(ax=ax_main, color=color, edgecolor='white', linewidth=0.5, alpha=1.0)
    country_boundary.plot(ax=ax_main, facecolor='none', edgecolor='black', linewidth=1.2)
    province_boundaries.plot(ax=ax_main, facecolor='none', edgecolor='#333333', linewidth=0.8)
    ax_main.set_ylim(bottom=1900000.0, top=6120000.0)
    ax_main.set_xlim(right=2500000.0)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5, alpha=1.0, label=label) for color, label in zip(colors, ['Small', 'Medium', 'Large'])]
    legend_title = 'Bus network scale'
    legend = ax_main.legend(handles=legend_elements, title=legend_title, loc='lower left', fontsize=12, frameon=True, title_fontsize=13)
    legend.get_title().set_fontweight('bold')
    ax_inset = fig.add_axes([0.78, 0.21, 0.11, 0.19])
    _plot_inset_map(ax_inset, cities, province_boundaries, country_boundary, nine_dash_line, show_province_colors=False, city_color='#f0f0f0')
    for group_id, color in zip([1, 2, 3], colors):
        group_cities = clustered_data[clustered_data['group'] == group_id]
        if not group_cities.empty:
            matched_cities = cities[cities['市代码'].isin(group_cities['citycode'])]
            if not matched_cities.empty:
                matched_cities.plot(ax=ax_inset, color=color, edgecolor='white', linewidth=0.3, alpha=1.0)
    for ax in (ax_main, ax_inset):
        for spine in ax.spines.values():
            spine.set_visible(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _adjust_readability(fig, font_scale, canvas_scale)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=True)
    plt.close()
def cluster_cities(static_data: pd.DataFrame, n_clusters: int=4) -> pd.DataFrame:
    data = static_data[['city', 'network_length_km']].copy()
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(data[['network_length_km']].values)
    data['raw_group'] = labels
    group_means = data.groupby('raw_group')['network_length_km'].mean().sort_values()
    ordered_groups = {old_group: new_group for new_group, old_group in enumerate(group_means.index, start=1)}
    data['group'] = data['raw_group'].map(ordered_groups)
    data.loc[data['group'].eq(4), 'group'] = 3
    return data[['city', 'network_length_km', 'group']]

def subplot_ylim(values: list[float]) -> tuple[float, float]:
    finite_values = [abs(value) for value in values if math.isfinite(value)]
    max_abs = max(finite_values, default=1.0)
    max_abs = max(max_abs, 0.1)
    padded = max_abs * 1.15
    if padded <= 3:
        step = 0.5
    elif padded <= 8:
        step = 1
    elif padded <= 20:
        step = 2
    elif padded <= 60:
        step = 5
    elif padded <= 150:
        step = 10
    else:
        step = 25
    limit = math.ceil(padded / step) * step
    return (-limit, limit)

def selected_axis_panels(changes, groups):
    means = changes.merge(groups,on="city",validate="many_to_one").groupby(["scenario","group"])[list(fig_2.METRICS)].mean().reset_index()
    panels = []
    for label,metric,title,limits,ticks in PANELS:
        names = next(g["scenarios"] for g in fig_2.SENSITIVITY_GROUPS if g["label"] == label)
        values = means.loc[means.scenario.isin(names),metric]
        lo,hi = limits
        step = abs(ticks[1]-ticks[0])
        if values.min() < lo: lo = np.floor(values.min()/step)*step-step*.2
        if values.max() > hi: hi = np.ceil(values.max()/step)*step+step*.2
        expanded = list(np.arange(np.ceil(lo/step)*step,np.floor(hi/step)*step+step*.5,step))
        panels.append((label,metric,title,(lo,hi),expanded if (lo,hi) != limits else ticks))
    panels[0] = (*panels[0][:3],(-4.3,.55),[0,-2,-4])
    panels[3] = (*panels[3][:4],[-20,0,20,40,60,80])
    return panels

def group_mean_changes(scenario_data, baseline, clusters, scenario_name, metric):
    values = scenario_data[scenario_name].merge(clusters[["city","group"]],on="city",validate="many_to_one")
    return values.groupby("group")[metric].mean().to_dict()
def plot_selected_changes(readable: bool=True, changes_path: Path | None=None, output_path: Path | None=None) -> None:
    changes = pd.read_csv(changes_path or fig_2.OUTPUT_DIR / 'plot_data' / 'what_if_percent_changes.csv')
    groups = pd.read_csv(GROUP_FILE)[['city', 'group']]
    if len(groups) != 224 or groups['city'].duplicated().any() or set(groups['city']) != set(changes['city']):
        raise ValueError('Fig. 3 city groups do not match the What-if results')
    merged = changes.merge(groups, on='city', validate='many_to_one')
    if len(merged) != 13 * 224:
        raise ValueError('Fig. 3 requires the baseline and all 12 scenarios for 224 cities')
    plt.rcParams.update({'font.sans-serif': ['Arial', 'DejaVu Sans', 'SimHei'], 'axes.unicode_minus': False, 'font.size': 10, 'font.family': 'Arial', 'axes.linewidth': 1.25, 'xtick.major.width': 1.15, 'ytick.major.width': 1.15, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 4.8, 'ytick.major.size': 4.8})
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.25))
    handles = []
    for ax, (group_label, metric, ylabel, ylim, yticks) in zip(axes, selected_axis_panels(changes, groups)):
        group = next((g for g in fig_2.SENSITIVITY_GROUPS if g['label'] == group_label))
        positions = list(range(5))
        axis_values = []
        for group_id in (1, 2, 3):
            values = []
            for scenario in group['scenarios']:
                rows = merged.loc[merged['scenario'].eq(scenario) & merged['group'].eq(group_id)]
                if rows.empty:
                    raise ValueError(f'Empty Fig. 3 group: {scenario}/{group_id}')
                values.append(rows[metric].mean())
            line = ax.plot(positions, values, marker='o', markersize=3.2, linewidth=1.35, color=GROUP_COLORS[group_id], label=LABELS[group_id])[0]
            if len(handles) < 3:
                handles.append(line)
            axis_values.extend(values)
        ax.set_ylabel(ylabel, fontsize=8.8, labelpad=2)
        ax.set_xlabel(group_label, fontsize=8.8, labelpad=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([label.split('\n', 1)[0] for label in group['xticklabels']], fontsize=7.8)
        ax.set_xlim(-0.2, 4.2)
        if min(axis_values) < ylim[0] or max(axis_values) > ylim[1]:
            raise ValueError(f'Figure 3 values exceed the selected y-axis: {group_label}/{metric}')
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(True, linestyle=':', alpha=0.55, linewidth=0.65)
        ax.set_box_aspect(1)
        ax.tick_params(axis='y', labelsize=7.8)
        ax.tick_params(axis='x', labelsize=7.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for edge in ('left', 'bottom'):
            ax.spines[edge].set_color('black')
            ax.spines[edge].set_linewidth(1.25)
    fig.text(0.012, 0.955, 'Bus network scale', ha='left', va='center', fontsize=8.3, fontweight='bold')
    fig.legend(handles=handles, labels=[LABELS[i] for i in (1, 2, 3)], loc='center left', ncol=3, frameon=False, fontsize=8.3, bbox_to_anchor=(0.18, 0.955), borderaxespad=0.0, handlelength=1.8, columnspacing=1.1)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.82, wspace=0.38)
    if readable:
        _adjust_readability(fig, SELECTED_FONT_SCALE, SELECTED_CANVAS_SCALE)
    target = READABLE_DIR if readable else OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = output_path or target / 'Figure_3d.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved Figure 3: {path}')
def plot_scenario_changes_by_group(changes, groups, output_dir=None) -> Path:
    clusters = groups
    scenario_data = {name: changes.loc[changes.scenario.eq(name)] for name in changes.scenario.unique()}
    baseline = scenario_data['baseline']
    active_groups = [g for g in fig_2.SENSITIVITY_GROUPS if any((s != 'baseline' and s in scenario_data for s in g['scenarios']))]
    FIGURE_DIR = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    FIGURE_STEM = 'What_if_by_network_scale'
    fig, axes = plt.subplots(len(active_groups), len(fig_2.METRIC_LABELS), figsize=(20.0, 3.05 * len(active_groups)), sharey=False)
    if len(active_groups) == 1:
        axes = axes.reshape(1, -1)
    legend_handles = []
    for row_idx, sensitivity_group in enumerate(active_groups):
        for col_idx, (metric, metric_title) in enumerate(fig_2.METRIC_LABELS):
            ax = axes[row_idx, col_idx]
            x_positions = list(range(len(sensitivity_group['scenarios'])))
            y_values_for_axis = []
            for group_id in (1, 2, 3):
                y_values = []
                for scenario in sensitivity_group['scenarios']:
                    medians = group_mean_changes(scenario_data, baseline, clusters, scenario, metric)
                    y_values.append(medians.get(group_id, float('nan')))
                line = ax.plot(x_positions, y_values, marker='o', markersize=4.8, linewidth=2.0, color=GROUP_COLORS[group_id], label=GROUP_LABELS[group_id])[0]
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                y_values_for_axis.extend(y_values)
            ax.set_title(metric_title, fontsize=13.5, fontweight='bold', pad=9)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(sensitivity_group['xticklabels'], fontsize=9.5)
            ax.set_xlim(-0.35, len(x_positions) - 0.65)
            ax.set_ylim(*subplot_ylim(y_values_for_axis))
            ax.yaxis.set_major_formatter(PercentFormatter())
            ax.grid(True, linestyle='--', alpha=0.42, linewidth=0.65)
            ax.set_box_aspect(1)
            ax.tick_params(axis='both', labelsize=10.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.15)
            if col_idx == 0:
                ax.text(-0.52, 0.5, sensitivity_group['label'], transform=ax.transAxes, fontsize=12.5, fontweight='bold', rotation=90, va='center', ha='center')
    fig.legend(handles=legend_handles, labels=[GROUP_LABELS[group_id] for group_id in (1, 2, 3)], loc='upper center', ncol=3, frameon=False, fontsize=12.2, bbox_to_anchor=(0.54, 1.01))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_file = FIGURE_DIR / f'{FIGURE_STEM}.png'
    plt.tight_layout(rect=[0.055, 0.025, 0.995, 0.965], h_pad=2.45, w_pad=1.05)
    fig.savefig(png_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return png_file

def main():
    global ARCHIVE_ROOT,OUTPUT_DIR,READABLE_DIR
    parser = argparse.ArgumentParser(description="Plot network-scale comparisons and grouped What-if results")
    parser.add_argument("--results-root",type=Path,default=RESULTS_ROOT)
    parser.add_argument("--output",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--changes",type=Path)
    parser.add_argument("--baseline-only",action="store_true")
    parser.add_argument("--map",action="store_true")
    args = parser.parse_args()
    ARCHIVE_ROOT = args.results_root / "baseline"
    OUTPUT_DIR = READABLE_DIR = args.output
    args.output.mkdir(parents=True,exist_ok=True)
    plot_data = args.output / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    if args.changes is None:
        args.changes = plot_data / "what_if_percent_changes.csv"
    data = build_city_data()
    data.to_csv(plot_data / "grouped_baseline.csv",index=False)
    plot(data,readable=True)
    groups = pd.read_csv(GROUP_FILE)
    if args.map:
        plot_clustered_cities_map(groups,str(args.output / "Figure_3a.png"),
            font_scale=MAP_FONT_SCALE,canvas_scale=MAP_CANVAS_SCALE,show=False)
    if not args.baseline_only:
        changes = pd.read_csv(args.changes)
        plot_selected_changes(readable=True,changes_path=args.changes,output_path=args.output / "Figure_3d.png")
        means = changes.merge(groups,on="city",validate="many_to_one").groupby(["scenario","group"])[list(fig_2.METRICS)].mean().reset_index()
        means.to_csv(plot_data / "what_if_group_means.csv",index=False)
        plot_scenario_changes_by_group(changes,groups,output_dir=args.output)

if __name__ == "__main__":
    main()
