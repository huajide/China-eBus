from __future__ import annotations

import argparse
from pathlib import Path

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageOps
from sklearn.preprocessing import MinMaxScaler
from fig_1 import _adjust_readability

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = DATA_DIR / "output" / "results"
FIGURES_ROOT = DATA_DIR / "output" / "figures"

STATIC_DATA_FILE = DATA_DIR / "224city_indicators.csv"
CITY_FILE = DATA_DIR / "224cities.csv"
VS_ROOT = DATA_DIR / "input" / "vs_parking_nodeid"
BASELINE_ROOT = RESULTS_ROOT / "baseline"
WHAT_IF_ROOT = RESULTS_ROOT / "what_if"
OUTPUT_DIR = FIGURES_ROOT
READABLE_DIR = OUTPUT_DIR
FONT_SCALE = 1.25
CANVAS_SCALE = 1.
Y_LIMITS = {"emission": (-40,40)}
SENSITIVITY_GROUPS = [{'label': 'Charging speed', 'color': '#1f77b4', 'scenarios': ['baseline', 'FCS2', 'FCS3', 'FCS4', 'FCS5'], 'xticklabels': ['1x\n(baseline)', '2x', '3x', '4x', '5x']}, {'label': 'Battery capacity', 'color': '#2ca02c', 'scenarios': ['baseline', 'BC1.25', 'BC1.5', 'BC1.75', 'BC2'], 'xticklabels': ['1x\n(baseline)', '1.25x', '1.5x', '1.75x', '2x']}, {'label': 'Vehicle cost', 'color': '#d62728', 'scenarios': ['VC0.6', 'VC0.7', 'VC0.8', 'VC0.9', 'baseline'], 'xticklabels': ['0.6x', '0.7x', '0.8x', '0.9x', '1x\n(baseline)']}]
METRIC_LABELS = [('cost', 'Cost'), ('emission', 'Emission'), ('built_num', 'Charging stations'), ('fast_charger', 'Fast chargers'), ('slow_charger', 'Slow chargers'), ('vehicle_count', 'Electric buses')]
METRICS = tuple(key for key, _ in METRIC_LABELS)
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial', 'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 6, 'ytick.major.size': 6})
CASES = {'baseline': (BASELINE_ROOT, None, None), **{f'FCS{v}': (WHAT_IF_ROOT / 'fast_charging_speed' / f'FCS{v}', 'power_factor', v) for v in (2, 3, 4, 5)}, **{f'BC{v:g}': (WHAT_IF_ROOT / 'battery_capacity_growth' / f'BC{v:g}', 'degradation_factor', v) for v in (1.25, 1.5, 1.75, 2.0)}, **{f'VC{v}': (WHAT_IF_ROOT / 'vehicle_purchase_cost' / f'VC{v}', 'vehicle_price_factor', v) for v in (0.6, 0.7, 0.8, 0.9)}}
def read_static_data() -> pd.DataFrame:
    static_data = pd.read_csv(STATIC_DATA_FILE)
    required_columns = {'city', 'network_length_km', 'vehicle_count'}
    missing_columns = required_columns.difference(static_data.columns)
    if missing_columns:
        missing_text = ', '.join(sorted(missing_columns))
        raise ValueError(f'{STATIC_DATA_FILE} is missing columns: {missing_text}')
    return static_data[['city', 'network_length_km', 'vehicle_count']].copy()

def read_archive(csv_file: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_file, index_col=0)
    if data.empty:
        raise ValueError(f'{csv_file} is empty')
    return data

def row_by_solution(data: pd.DataFrame, solution_no: object) -> pd.Series:
    matches = data.loc[data.index.astype(str) == str(solution_no)]
    if matches.empty:
        raise KeyError(f'solution {solution_no} is missing')
    return pd.to_numeric(matches.iloc[-1], errors='coerce')

def pareto_front(obj_data: pd.DataFrame) -> pd.DataFrame:
    objs = obj_data.copy()
    objs.columns = ['obj1', 'obj2']
    objs[['obj1', 'obj2']] = objs[['obj1', 'obj2']].abs()
    objs = objs.reset_index(names='solution_no')
    objs = objs.sort_values(['obj1', 'obj2'], kind='mergesort')
    return objs[objs['obj2'].cummin().eq(objs['obj2'])].copy()

def find_knee_solution(obj_data: pd.DataFrame) -> object:
    front = pareto_front(obj_data)
    if front.empty:
        raise ValueError('empty Pareto front')
    if len(front) == 1:
        return front.iloc[0]['solution_no']
    scaled = MinMaxScaler().fit_transform(front[['obj1', 'obj2']])
    distances = (scaled ** 2).sum(axis=1) ** 0.5
    return front.iloc[distances.argmin()]['solution_no']

def symmetric_ylim(values: list[list[float]]) -> tuple[float, float]:
    max_abs = 1.0
    for data in values:
        if not data:
            continue
        stats = boxplot_stats(data, whis=1.5)[0]
        max_abs = max(max_abs, abs(stats['whislo']), abs(stats['whishi']))
    padded = max_abs * 1.12
    step = 5 if padded <= 60 else 10 if padded <= 150 else 25
    limit = math.ceil(padded / step) * step
    return (-limit, limit)

def set_axis_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

def plot_group(ax: plt.Axes, values: list[list[float]], positions: list[int], color: str, show_mean: bool=False) -> None:
    if not values:
        return
    boxes = ax.boxplot(values, positions=positions, widths=0.6, patch_artist=True, notch=False, showfliers=False, boxprops={'facecolor': color, 'edgecolor': 'black', 'alpha': 0.72, 'linewidth': 0.9}, medianprops={'color': 'black', 'linewidth': 1.7}, whiskerprops={'color': 'black', 'linewidth': 0.9}, capprops={'color': 'black', 'linewidth': 0.9})
    if show_mean:
        for sample, median in zip(values, boxes['medians']):
            ax.plot(median.get_xdata(), [np.mean(sample)] * 2, color='gray',
                    linestyle='--', linewidth=median.get_linewidth(), scalex=False, scaley=False)
def _read_archives(folder: Path):
    objs = pd.read_csv(folder / 'archive_objs.csv', index_col=0)
    cvs = pd.read_csv(folder / 'archive_cvs.csv', index_col=0)
    vars_ = pd.read_csv(folder / 'archive_vars.csv', index_col=0)
    if objs.empty or len(objs.columns) != 2 or (not objs.index.is_unique) or (not objs.index.equals(cvs.index)) or (not objs.index.equals(vars_.index)):
        raise ValueError(f'Archive solution IDs do not align: {folder}')
    stations = len(vars_.columns) - 3
    if stations < 1 or len(cvs.columns) != 2 * stations + 6:
        raise ValueError(f'Archive dimensions do not align: {folder}')
    if not np.isfinite(objs.to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite objectives: {folder}')
    if (cvs.iloc[:, 0].to_numpy(dtype=float) > 1e-09).any():
        raise ValueError(f'Infeasible solution in feasible archive: {folder}')
    return (objs, cvs, vars_, stations)

def build_city_metrics() -> pd.DataFrame:
    static = read_static_data().set_index('city')
    cities = pd.read_csv(CITY_FILE)['city'].tolist()
    if len(cities) != 224 or len(set(cities)) != 224 or set(cities) != set(static.index):
        raise ValueError('City lists do not match the 224-city study sample')
    base_fleet = {city: pd.read_csv(VS_ROOT / f'{city}.csv', usecols=['v_name'])['v_name'].nunique() for city in cities}
    if any((count <= 0 for count in base_fleet.values())):
        raise ValueError('A city has no scheduled baseline buses')
    records = []
    for scenario, (root, parameter, value) in CASES.items():
        folders = {folder.name for folder in root.iterdir() if folder.is_dir()}
        if folders != set(cities):
            raise ValueError(f'Expected exactly 224 cities in {scenario}; missing={set(cities) - folders}')
        for city in cities:
            folder = root / city
            config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
            if config['city'] != city or config['base_seed'] != 42:
                raise ValueError(f'Wrong city or seed: {folder}')
            if scenario != 'baseline' and (config['what_if_parameter'] != parameter or config['what_if_value'] != value):
                raise ValueError(f'Wrong scenario parameter: {folder}')
            objs, cvs, vars_, stations = _read_archives(folder)
            solution = find_knee_solution(objs)
            obj = row_by_solution(objs, solution)
            cv = row_by_solution(cvs, solution)
            var = row_by_solution(vars_, solution)
            if not np.isfinite(np.r_[obj.to_numpy(), cv.to_numpy(), var.to_numpy()]).all():
                raise ValueError(f'Non-finite knee-point values: {folder}')
            if (obj >= 0).any() or cv.iloc[0] > 1e-09:
                raise ValueError(f'Invalid knee-point objective or feasibility: {folder}')
            flags = var.iloc[:stations].to_numpy(dtype=float) > 0.5
            fast = cv.iloc[1:1 + stations].to_numpy(dtype=float)
            slow = cv.iloc[1 + stations:1 + 2 * stations].to_numpy(dtype=float)
            extras = cv.iloc[-3:].to_numpy(dtype=float)
            if (fast > 1e-09).any() or (slow > 1e-09).any() or (extras > 1e-09).any():
                raise ValueError(f'Unexpected charger or additional-bus signs: {folder}')
            length = float(static.loc[city, 'network_length_km'])
            if not math.isfinite(length) or length <= 0:
                raise ValueError(f'Invalid network length: {city}')
            records.append({'scenario': scenario, 'city': city, 'knee_solution': solution, 'network_length_km': length, 'base_vehicles': base_fleet[city], 'additional_vehicles': -extras.sum(), 'cost': -float(obj.iloc[0]) / length, 'emission': -float(obj.iloc[1]) / length, 'built_num': int(flags.sum()) / length, 'fast_charger': -fast[flags].sum() / length, 'slow_charger': -slow[flags].sum() / length, 'vehicle_count': (base_fleet[city] - extras.sum()) / length})
        print(f'Validated {scenario}: {len(cities)} cities', flush=True)
    return pd.DataFrame.from_records(records)

def relative_changes(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = data.loc[data['scenario'].eq('baseline')].set_index('city')
    rows = []
    undefined = []
    for scenario in CASES:
        current = data.loc[data['scenario'].eq(scenario)].set_index('city')
        if set(current.index) != set(baseline.index):
            raise ValueError(f'City mismatch: {scenario}')
        for city in baseline.index:
            row = {'scenario': scenario, 'city': city}
            for metric in METRICS:
                a = float(baseline.loc[city, metric])
                b = float(current.loc[city, metric])
                if a == 0:
                    if b != 0:
                        row[metric] = float('nan')
                        undefined.append({'scenario': scenario, 'city': city, 'metric': metric, 'baseline_value': a, 'scenario_value': b})
                    else:
                        row[metric] = 0.0
                else:
                    row[metric] = (b - a) / a * 100.0
            rows.append(row)
    return (pd.DataFrame.from_records(rows), pd.DataFrame.from_records(undefined, columns=['scenario', 'city', 'metric', 'baseline_value', 'scenario_value']))

def plot_figure2(changes: pd.DataFrame, readable: bool=False, fast_only: bool=False, output_path: Path | None=None, limit_data: pd.DataFrame | None=None, ylim_overrides: dict | None=None) -> Path:
    metric_ylim = {}
    limits = changes if limit_data is None else limit_data
    for metric in METRICS:
        values = [limits.loc[limits['scenario'].eq(case), metric].dropna().tolist() for group in SENSITIVITY_GROUPS for case in group['scenarios']]
        if metric == 'slow_charger':
            stats = [boxplot_stats(v, whis=1.5)[0] for v in values]
            low = min((s['whislo'] for s in stats))
            high = max((s['whishi'] for s in stats))
            span = max(high - low, 1.0)
            metric_ylim[metric] = (math.floor((low - span * 0.12) / 5) * 5, math.ceil((high + span * 0.12) / 5) * 5)
        else:
            low, high = symmetric_ylim(values)
            limit = max(abs(low), abs(high), 5 if metric in {'emission', 'vehicle_count'} else 0)
            metric_ylim[metric] = (-limit, limit)
    metric_ylim.update(Y_LIMITS)
    metric_ylim.update(ylim_overrides or {})
    groups = SENSITIVITY_GROUPS[:1] if fast_only else SENSITIVITY_GROUPS
    fig, axes = plt.subplots(len(groups), 6, figsize=(18.0, 3.0 * len(groups)), sharey=False)
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    for row_idx, group in enumerate(groups):
        for col_idx, (metric, title) in enumerate(METRIC_LABELS):
            ax = axes[row_idx, col_idx]
            values = [changes.loc[changes['scenario'].eq(case), metric].dropna().tolist() for case in group['scenarios']]
            plot_group(ax, values, list(range(1, 6)), group['color'], show_mean=True)
            set_axis_style(ax, {'cost': 'System costs', 'emission': 'GHG emissions'}.get(metric, title))
            ax.title.set_fontweight('normal')
            ax.title.set_fontsize(14)
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.set_ylim(*metric_ylim[metric])
            ax.set_xlim(0.4, 5.6)
            ax.set_xticks(range(1, 6))
            labels = ['1x\n(baseline)', '1.25', '1.5', '1.75', '2'] if row_idx == 1 else ['0.6', '0.7', '0.8', '0.9', '1x\n(baseline)'] if row_idx == 2 else ['1x\n(baseline)', '2', '3', '4', '5']
            ax.set_xticklabels(labels, fontsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            if col_idx == 0:
                ax.text(-0.43, 0.5, group['label'], transform=ax.transAxes, fontsize=13, fontweight='bold', rotation=90, va='center', ha='center')
    if readable:
        _adjust_readability(fig, FONT_SCALE, CANVAS_SCALE)
    if readable:
        fig.tight_layout(rect=[0.01, 0.02, 0.995, 0.985], w_pad=0.0, h_pad=0.6)
    else:
        fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    target = READABLE_DIR if readable else OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = output_path or target / 'Figure_2.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    top_margin = round(axes[0, 0].title.get_fontsize() / 72 * 300)
    with Image.open(path) as image:
        padded = ImageOps.expand(image, border=(0, top_margin, 0, 0), fill='white')
    padded.save(path, dpi=(300, 300))
    plt.close(fig)
    return path

def configure_results(results_root):
    global CASES
    CASES = {name: (results_root / "baseline" if name == "baseline" else
             results_root / "what_if" / folder.parent.name / folder.name, parameter, value)
             for name,(folder,parameter,value) in CASES.items()}

def main():
    parser = argparse.ArgumentParser(description="Plot charging-speed, battery-capacity and vehicle-price scenarios")
    parser.add_argument("--results-root",type=Path,default=RESULTS_ROOT)
    parser.add_argument("--output",type=Path,default=OUTPUT_DIR)
    args = parser.parse_args()
    configure_results(args.results_root)
    metrics = build_city_metrics()
    changes,undefined = relative_changes(metrics)
    args.output.mkdir(parents=True,exist_ok=True)
    plot_data = args.output / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(plot_data / "what_if_city_metrics.csv",index=False)
    changes.to_csv(plot_data / "what_if_percent_changes.csv",index=False)
    undefined.to_csv(plot_data / "what_if_undefined_percent_changes.csv",index=False)
    plot_figure2(changes,readable=True,output_path=args.output / "Figure_2.png")

if __name__ == "__main__":
    main()
