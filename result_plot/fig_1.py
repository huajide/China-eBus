from __future__ import annotations

import argparse
from pathlib import Path

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = DATA_DIR / "output" / "results"
FIGURES_ROOT = DATA_DIR / "output" / "figures"

OUTPUT_DIR = FIGURES_ROOT
FIG1_STYLE = {"b": (1.08, .96), "c": (1.08, .96), "d": (1.08, .96), "e": (1., .96)}
FIG1D_BOX_WIDTH = .44
def _adjust_readability(fig, font_scale=1.0, canvas_scale=1.0):
    if font_scale == 1.0 and canvas_scale == 1.0:
        return
    for text in fig.findobj(match=plt.Text):
        text.set_fontsize(text.get_fontsize() * font_scale)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width * canvas_scale, height * canvas_scale)

def plot_route_count_vs_population_gdp(indicator_data, city_data, output_dir='fig_1', font_scale=1.0, canvas_scale=1.0):
    """Plot bus-network relationships with population and GDP."""
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
    ax1.text(0.05, 0.95, f'$R^2$ = {r_value ** 2:.2f}\np < 0.001' if p_value < 0.001 else f'$R^2$ = {r_value ** 2:.2f}\np = {p_value:.3f}', transform=ax1.transAxes, fontsize=18, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.set_xlabel('Population ($10^4$ people)', fontsize=18)
    ax1.set_ylabel('Route count', fontsize=18)
    ax1.set_title('Route count vs population', fontsize=20, pad=20)
    ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
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
    ax2.text(0.05, 0.95, f'$R^2$ = {r_value2 ** 2:.2f}\np < 0.001' if p_value2 < 0.001 else f'$R^2$ = {r_value2 ** 2:.2f}\np = {p_value2:.3f}', transform=ax2.transAxes, fontsize=18, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax2.set_xlabel('GDP (billion yuan)', fontsize=18)
    ax2.set_ylabel('Route count', fontsize=18)
    ax2.set_title('Route count vs GDP', fontsize=20, pad=20)
    ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(True, linestyle=':', alpha=0.5, linewidth=1)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.6)
    _adjust_readability(fig, font_scale, canvas_scale)
    plt.savefig(f'{output_dir}/Figure_1c.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_operation_indicators_boxplot(indicator_data, output_dir='fig_1', font_scale=1.0, canvas_scale=1.0):
    """Plot distributions of operating indicators."""
    os.makedirs(output_dir, exist_ok=True)
    target_cols = ['route_count', 'route_avg_distance', 'route_avg_trip']
    titles = {'route_count': 'Route count', 'route_avg_distance': 'Average route\ndistance (km)', 'route_avg_trip': 'Average trips\nper route (per day)'}
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
    plt.subplots_adjust(wspace=0.4, hspace=0.5, top=0.78)
    colors = ['#377eb8', '#ff7f00', '#4daf4a']
    legend_handles = []
    legend_labels = ['Median', 'Mean']
    box_width = 0.35
    half_box_width = box_width / 2
    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        current_data = melted_df[melted_df['Indicator'] == col]
        current_data = current_data.sort_values('Value')
        box = sns.boxplot(y='Value', data=current_data, ax=ax, orient='v', width=box_width, fliersize=4, showmeans=False, color=colors[i], boxprops=dict(linewidth=2, edgecolor='black', facecolor=colors[i] + '80'), medianprops=dict(color='black', linewidth=2.4, linestyle='--'), whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2))
        mean_value = current_data['Value'].mean()
        q1 = current_data['Value'].quantile(0.25)
        q3 = current_data['Value'].quantile(0.75)
        mean_line = ax.hlines(y=mean_value, xmin=-half_box_width, xmax=half_box_width, color='gray', linestyle='--', linewidth=2.4)
        hidden_line = ax.hlines(y=mean_value * 1.1, xmin=-box_width, xmax=box_width, color='none', linewidth=0)
        if i == 0:
            from matplotlib.lines import Line2D
            median_line = Line2D([0], [0], color='black', linewidth=2.4, linestyle='--')
            mean_line_legend = Line2D([0], [0], color='gray', linewidth=2.4, linestyle='--')
            legend_handles = [median_line, mean_line_legend]
        ax.set_title(titles[col], fontsize=20, pad=20)
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Value', fontsize=18)
        if i == 0:
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
    fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=16, ncol=2, columnspacing=1.5, handlelength=2.4)
    _adjust_readability(fig, font_scale, canvas_scale)
    if font_scale != 1.0 or canvas_scale != 1.0:
        fig.legends[0].set_bbox_to_anchor((0.5, 1.2))
    for ax in axes:
        ax.yaxis.label.set_color('white')
    plt.savefig(f'{output_dir}/Figure_1b.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_electrification_indicators_boxplot(indicator_data, output_data, output_dir='fig_1', show_legend=True, font_scale=1.0, canvas_scale=1.0):
    """Plot electrification outcomes per network kilometre."""
    os.makedirs(output_dir, exist_ok=True)
    analysis_df = pd.merge(indicator_data, output_data, left_on='city', right_on='city_name')
    analysis_df['knee_cost_per_km'] = analysis_df['knee_cost'] / analysis_df['network_length_km']
    analysis_df['knee_emission_per_km'] = analysis_df['knee_emission'] / analysis_df['network_length_km']
    analysis_df['knee_cs_per_km'] = analysis_df['knee_built_cs'] / analysis_df['network_length_km']
    analysis_df['knee_charger_per_km'] = (analysis_df['knee_fast_per_cs'] + analysis_df['knee_slow_per_cs']) * analysis_df['knee_built_cs'] / analysis_df['network_length_km']
    analysis_df['knee_ev_per_km'] = (analysis_df['vehicle_count'] + analysis_df['knee_extra_large'] + analysis_df['knee_extra_medium'] + analysis_df['knee_extra_small']) / analysis_df['network_length_km']
    target_cols = ['knee_cost_per_km', 'knee_emission_per_km', 'knee_cs_per_km', 'knee_charger_per_km', 'knee_ev_per_km']
    titles = {'knee_cost_per_km': 'System costs\nper km (M yuan km$^{-1}$ yr$^{-1}$)', 'knee_emission_per_km': 'GHG emissions\nper km (t km$^{-1}$ yr$^{-1}$)', 'knee_cs_per_km': 'Charging stations\nper km (km$^{-1}$)', 'knee_charger_per_km': 'Chargers\nper km (km$^{-1}$)', 'knee_ev_per_km': 'Electric buses\nper km (km$^{-1}$)'}
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
    if show_legend:
        axes[-1].legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=16)
    _adjust_readability(fig, font_scale, canvas_scale)
    for ax in axes:
        ax.yaxis.label.set_color('white')
    plt.savefig(f'{output_dir}/Figure_1e.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
def check_city_data(reference, current):
    if set(reference.city) != set(current.city) or current.city.duplicated().any():
        raise ValueError('City groups do not align')

def gdp_electricity_data(fig3: pd.DataFrame, output: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    data = fig3[['city', 'network_length_km', 'electricity_mwh_per_km']].merge(output[['city_name', 'knee_cost']], left_on='city', right_on='city_name', validate='one_to_one').merge(cities[['city', 'district_gdp', 'ind_electricity']], on='city', validate='one_to_one')
    data['cost_to_gdp_ratio'] = data['knee_cost'] / (data['district_gdp'] * 1000000.0) * 100
    data['electricity_ratio'] = data['electricity_mwh_per_km'] * data['network_length_km'] * 1000.0 / (data['ind_electricity'] * 1000000000.0) * 100
    if len(data) != 224 or not (data[['cost_to_gdp_ratio', 'electricity_ratio']] >= 0).all().all():
        raise ValueError('Invalid Figure 1d values')
    return data

def plot_figure_1d(data: pd.DataFrame, output_dir: Path=OUTPUT_DIR, font_scale: float=1.0, canvas_scale: float=1.0, box_width: float=0.4) -> Path:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 22, 'axes.titlesize': 24, 'axes.labelsize': 24, 'xtick.labelsize': 19, 'ytick.labelsize': 19, 'axes.linewidth': 1.6, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'in', 'ytick.direction': 'in'})
    plt.rcParams.update({'xtick.major.width': 1.25, 'ytick.major.width': 1.25,
                         'xtick.major.size': 5.5, 'ytick.major.size': 5.5,
                         'xtick.minor.width': .6, 'ytick.minor.width': .6,
                         'xtick.minor.size': 2, 'ytick.minor.size': 2})
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    for ax, col, title, color in zip(axes, ('cost_to_gdp_ratio', 'electricity_ratio'), ('Cost-to-GDP\nratio (%)', 'Electricity\nratio (%)'), ('#377eb8', '#ff7f00')):
        sns.boxplot(y=data[col].sort_values(), ax=ax, orient='v', width=box_width, fliersize=4, showmeans=False, color=color, boxprops={'linewidth': 2, 'edgecolor': 'black', 'facecolor': color + '80'}, medianprops={'color': 'black', 'linewidth': 2.4, 'linestyle': '--'}, whiskerprops={'color': 'black', 'linewidth': 2}, capprops={'color': 'black', 'linewidth': 2})
        ax.hlines(data[col].mean(), -box_width / 2, box_width / 2, color='gray', linestyle='--', linewidth=2.4)
        if box_width != 0.4:
            ax.set_xlim(-0.5, 0.5)
        ax.set(title=title, xlabel='', ylabel='Value')
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=1)
    fig.tight_layout()
    _adjust_readability(fig, font_scale, canvas_scale)
    for ax in axes:
        ax.yaxis.label.set_color('white')
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'Figure_1d.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 1 panels")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--map", action="store_true")
    args = parser.parse_args()
    import tab_1
    import fig_3
    tab_1.ARCHIVE_ROOT = fig_3.ARCHIVE_ROOT = args.results_root / "baseline"
    static = pd.read_csv(DATA_DIR / "224city_indicators.csv")
    cities = pd.read_csv(DATA_DIR / "224cities.csv")
    indicators, outputs = tab_1.read_outputs(static)
    grouped = fig_3.build_city_data()
    args.output.mkdir(parents=True, exist_ok=True)
    for panel, function, data in (("b", plot_operation_indicators_boxplot, (static,)),
                                  ("c", plot_route_count_vs_population_gdp, (static,cities))):
        function(*data, output_dir=str(args.output), font_scale=FIG1_STYLE[panel][0], canvas_scale=FIG1_STYLE[panel][1])
    ratios = gdp_electricity_data(grouped,outputs,cities)
    plot_data = args.output / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(plot_data / "Figure_1d_data.csv",index=False)
    plot_figure_1d(ratios,args.output,*FIG1_STYLE["d"],FIG1D_BOX_WIDTH)
    plot_electrification_indicators_boxplot(indicators,outputs,output_dir=str(args.output),show_legend=False,
        font_scale=FIG1_STYLE["e"][0],canvas_scale=FIG1_STYLE["e"][1])
    if args.map:
        from china_cities import plot_highlighted_cities
        plot_highlighted_cities(cities.citycode.tolist(),str(args.output / "Figure_1a.png"),show=False)

if __name__ == "__main__":
    main()
