from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import MinMaxScaler


RUN_TAG = "260420"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "output" / "mosa"
WHAT_IF_DIR = OUTPUT_DIR / "what_if"
STATIC_DATA_FILE = DATA_DIR / "224city_indicators.csv"
FIGURE_DIR = SCRIPT_DIR / "fig_2&3"
FIGURE_STEM = "scenario_comparison"
SOLUTION_SELECTION = "knee"

SCENARIOS = {
    "baseline": OUTPUT_DIR / RUN_TAG,
    "FCS2": WHAT_IF_DIR / f"{RUN_TAG}FCS2",
    "FCS3": WHAT_IF_DIR / f"{RUN_TAG}FCS3",
    "FCS4": WHAT_IF_DIR / f"{RUN_TAG}FCS4",
    "FCS5": WHAT_IF_DIR / f"{RUN_TAG}FCS5",
    "BC1.25": WHAT_IF_DIR / f"{RUN_TAG}BC1.25",
    "BC1.5": WHAT_IF_DIR / f"{RUN_TAG}BC1.5",
    "BC1.75": WHAT_IF_DIR / f"{RUN_TAG}BC1.75",
    "BC2": WHAT_IF_DIR / f"{RUN_TAG}BC2",
    "VC0.6": WHAT_IF_DIR / f"{RUN_TAG}VC0.6",
    "VC0.7": WHAT_IF_DIR / f"{RUN_TAG}VC0.7",
    "VC0.8": WHAT_IF_DIR / f"{RUN_TAG}VC0.8",
    "VC0.9": WHAT_IF_DIR / f"{RUN_TAG}VC0.9",
}

SENSITIVITY_GROUPS = [
    {
        "label": "Fast charging speed",
        "color": "#1f77b4",
        "scenarios": ["baseline", "FCS2", "FCS3", "FCS4", "FCS5"],
        "xticklabels": ["1x\n(baseline)", "2x", "3x", "4x", "5x"],
    },
    {
        "label": "Battery capacity",
        "color": "#2ca02c",
        "scenarios": ["baseline", "BC1.25", "BC1.5", "BC1.75", "BC2"],
        "xticklabels": ["1x\n(baseline)", "1.25x", "1.5x", "1.75x", "2x"],
    },
    {
        "label": "Vehicle cost",
        "color": "#d62728",
        "scenarios": ["VC0.6", "VC0.7", "VC0.8", "VC0.9", "baseline"],
        "xticklabels": ["0.6x", "0.7x", "0.8x", "0.9x", "1x\n(baseline)"],
    },
]

METRICS = [
    ("cost", "Cost"),
    ("emission", "Emission"),
    ("built_num", "Charging stations"),
    ("fast_charger", "Fast chargers"),
    ("slow_charger", "Slow chargers"),
    ("vehicle_count", "Electric buses"),
]


plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "Arial",
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    }
)


def read_static_data() -> pd.DataFrame:
    static_data = pd.read_csv(STATIC_DATA_FILE)
    required_columns = {"city", "network_length_km", "vehicle_count"}
    missing_columns = required_columns.difference(static_data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{STATIC_DATA_FILE} is missing columns: {missing_text}")
    return static_data[["city", "network_length_km", "vehicle_count"]].copy()


def read_archive(csv_file: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_file, index_col=0)
    if data.empty:
        raise ValueError(f"{csv_file} is empty")
    return data


def row_by_solution(data: pd.DataFrame, solution_no: object) -> pd.Series:
    matches = data.loc[data.index.astype(str) == str(solution_no)]
    if matches.empty:
        raise KeyError(f"solution {solution_no} is missing")
    return pd.to_numeric(matches.iloc[-1], errors="coerce")


def pareto_front(obj_data: pd.DataFrame) -> pd.DataFrame:
    objs = obj_data.copy()
    objs.columns = ["obj1", "obj2"]
    objs[["obj1", "obj2"]] = objs[["obj1", "obj2"]].abs()
    objs = objs.reset_index(names="solution_no")
    objs = objs.sort_values(["obj1", "obj2"], kind="mergesort")
    return objs[objs["obj2"].cummin().eq(objs["obj2"])].copy()


def find_knee_solution(obj_data: pd.DataFrame) -> object:
    front = pareto_front(obj_data)
    if front.empty:
        raise ValueError("empty Pareto front")
    if len(front) == 1:
        return front.iloc[0]["solution_no"]

    scaled = MinMaxScaler().fit_transform(front[["obj1", "obj2"]])
    distances = (scaled**2).sum(axis=1) ** 0.5
    return front.iloc[distances.argmin()]["solution_no"]


def selected_archive_rows(city_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    obj_data = read_archive(city_dir / "inf_archive_objs.csv")
    cv_data = read_archive(city_dir / "inf_archive_cvs.csv")
    var_data = read_archive(city_dir / "inf_archive_vars.csv")
    if SOLUTION_SELECTION == "last":
        solution_no = obj_data.index[-1]
    elif SOLUTION_SELECTION == "knee":
        solution_no = find_knee_solution(obj_data)
    else:
        raise ValueError(f"Unknown solution selection method: {SOLUTION_SELECTION}")
    return (
        row_by_solution(obj_data, solution_no),
        row_by_solution(cv_data, solution_no),
        row_by_solution(var_data, solution_no),
    )


def infer_candidate_station_count(cv_row: pd.Series, var_row: pd.Series) -> int:
    cv_candidate_count = (len(cv_row) - 6) / 2
    var_candidate_count = len(var_row) - 3
    if not cv_candidate_count.is_integer():
        raise ValueError(f"Unexpected CV column count: {len(cv_row)}")
    if int(cv_candidate_count) != var_candidate_count:
        raise ValueError(
            f"Archive column mismatch: cvs imply {int(cv_candidate_count)} stations, "
            f"vars imply {var_candidate_count} stations"
        )
    return int(cv_candidate_count)


def charger_totals(cv_row: pd.Series, var_row: pd.Series) -> tuple[float, float, float, float]:
    candidate_station_count = infer_candidate_station_count(cv_row, var_row)
    if candidate_station_count <= 0:
        return 0.0, 0.0, 0.0, 0.0

    built_flags = pd.to_numeric(var_row.iloc[:candidate_station_count], errors="coerce").fillna(0).gt(0.5)
    fast_values = cv_row.iloc[1 : 1 + candidate_station_count].abs().astype(float)
    slow_values = cv_row.iloc[1 + candidate_station_count : 1 + 2 * candidate_station_count].abs().astype(float)

    built_num = float(built_flags.sum())
    fast_charger = float(fast_values[built_flags.to_numpy()].sum())
    slow_charger = float(slow_values[built_flags.to_numpy()].sum())
    additional_vehicles = float(cv_row.iloc[-3:].abs().sum())

    return built_num, fast_charger, slow_charger, additional_vehicles


def load_scenario_data(scenario_name: str, scenario_path: Path, static_data: pd.DataFrame) -> pd.DataFrame:
    if not scenario_path.exists():
        print(f"Missing scenario directory: {scenario_name} -> {scenario_path}")
        return pd.DataFrame()

    records = []
    static_lookup = static_data.set_index("city")
    for city_dir in sorted([path for path in scenario_path.iterdir() if path.is_dir()], key=lambda path: path.name):
        city = city_dir.name
        cv_file = city_dir / "inf_archive_cvs.csv"
        obj_file = city_dir / "inf_archive_objs.csv"
        var_file = city_dir / "inf_archive_vars.csv"

        if not cv_file.exists() or not obj_file.exists() or not var_file.exists():
            print(f"Missing archive files for {scenario_name}/{city}")
            continue
        if city not in static_lookup.index:
            print(f"Static indicators missing city: {city}")
            continue

        try:
            obj_row, cv_row, var_row = selected_archive_rows(city_dir)
            built_num, fast_charger, slow_charger, additional_vehicles = charger_totals(cv_row, var_row)
            network_length = float(static_lookup.loc[city, "network_length_km"])
            static_vehicle_count = float(static_lookup.loc[city, "vehicle_count"])

            if not math.isfinite(network_length) or network_length <= 0:
                print(f"Invalid network length for {city}: {network_length}")
                continue

            records.append(
                {
                    "scenario": scenario_name,
                    "city": city,
                    "cost": abs(float(obj_row.iloc[0])) / network_length,
                    "emission": abs(float(obj_row.iloc[1])) / network_length,
                    "built_num": built_num / network_length,
                    "fast_charger": fast_charger / network_length,
                    "slow_charger": slow_charger / network_length,
                    "vehicle_count": (static_vehicle_count + additional_vehicles) / network_length,
                    "network_length_km": network_length,
                }
            )
        except Exception as exc:
            print(f"Failed to process {scenario_name}/{city}: {exc}")

    return pd.DataFrame.from_records(records)


def relative_to_baseline(
    scenario_data: dict[str, pd.DataFrame], baseline: pd.DataFrame, scenario_name: str, metric: str
) -> list[float]:
    if scenario_name == "baseline":
        return [0.0] * len(baseline)
    current = scenario_data.get(scenario_name, pd.DataFrame())
    if current.empty:
        return []

    merged = baseline[["city", metric]].merge(
        current[["city", metric]],
        on="city",
        suffixes=("_baseline", "_scenario"),
        how="inner",
    )
    if len(merged) != len(baseline):
        print(f"{scenario_name}/{metric}: matched {len(merged)} of {len(baseline)} baseline cities")

    baseline_values = merged[f"{metric}_baseline"].astype(float)
    scenario_values = merged[f"{metric}_scenario"].astype(float)
    relative_change = pd.Series(0.0, index=merged.index)
    valid = baseline_values.ne(0)
    relative_change.loc[valid] = (
        (scenario_values.loc[valid] - baseline_values.loc[valid]) / baseline_values.loc[valid] * 100
    )
    return relative_change.astype(float).tolist()


def symmetric_ylim(values: list[list[float]]) -> tuple[float, float]:
    max_abs = 1.0
    for data in values:
        if not data:
            continue
        stats = boxplot_stats(data, whis=1.5)[0]
        max_abs = max(max_abs, abs(stats["whislo"]), abs(stats["whishi"]))

    padded = max_abs * 1.12
    step = 5 if padded <= 60 else 10 if padded <= 150 else 25
    limit = math.ceil(padded / step) * step
    return -limit, limit


def set_axis_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_group(ax: plt.Axes, values: list[list[float]], positions: list[int], color: str) -> None:
    if not values:
        return
    ax.boxplot(
        values,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        notch=False,
        showfliers=False,
        boxprops={"facecolor": color, "edgecolor": "black", "alpha": 0.72, "linewidth": 0.9},
        medianprops={"color": "black", "linewidth": 1.7},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
    )


def main() -> None:
    static_data = read_static_data()
    missing_scenarios = [name for name, path in SCENARIOS.items() if not path.exists()]
    if missing_scenarios:
        print("Missing scenarios:", ", ".join(missing_scenarios))

    scenario_data = {
        scenario: load_scenario_data(scenario, path, static_data)
        for scenario, path in SCENARIOS.items()
        if path.exists()
    }

    baseline = scenario_data.get("baseline", pd.DataFrame())
    if baseline.empty:
        raise RuntimeError(f"Baseline data is missing or empty: {SCENARIOS['baseline']}")
    baseline = baseline.sort_values("city").reset_index(drop=True)

    print(f"Loaded baseline cities: {len(baseline)}")
    for scenario, data in scenario_data.items():
        if scenario != "baseline":
            print(f"Loaded {scenario}: {len(data)} cities")

    active_groups = []
    for group in SENSITIVITY_GROUPS:
        available_non_baseline = [
            scenario for scenario in group["scenarios"] if scenario != "baseline" and scenario in scenario_data
        ]
        if available_non_baseline:
            active_groups.append(group)
        else:
            print(f"Skipped group with no 260420 what-if data: {group['label']}")

    metric_ylim = {}
    for metric, _ in METRICS:
        metric_values = []
        for group in active_groups:
            for scenario in group["scenarios"]:
                values = relative_to_baseline(scenario_data, baseline, scenario, metric)
                if values:
                    metric_values.append(values)
        if metric == "slow_charger":
            finite_values = [value for data in metric_values for value in data if pd.notna(value)]
            if finite_values:
                stats_low = []
                stats_high = []
                for data in metric_values:
                    if data:
                        stats = boxplot_stats(data, whis=1.5)[0]
                        stats_low.append(stats["whislo"])
                        stats_high.append(stats["whishi"])
                low = min(stats_low) if stats_low else min(finite_values)
                high = max(stats_high) if stats_high else max(finite_values)
                span = max(high - low, 1.0)
                metric_ylim[metric] = (math.floor((low - span * 0.12) / 5) * 5, math.ceil((high + span * 0.12) / 5) * 5)
            else:
                metric_ylim[metric] = (-5, 25)
        else:
            low, high = symmetric_ylim(metric_values)
            minimum_limit = 5 if metric in {"emission", "vehicle_count"} else 0
            limit = max(abs(low), abs(high), minimum_limit)
            metric_ylim[metric] = (-limit, limit)

    fig_height = 2.5 * len(active_groups)
    fig_width = 15.0
    fig, axes = plt.subplots(len(active_groups), len(METRICS), figsize=(fig_width, fig_height), sharey=False)
    if len(active_groups) == 1:
        axes = axes.reshape(1, -1)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for row_idx, group in enumerate(active_groups):
        for col_idx, (metric, metric_title) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            box_values = []
            box_positions = []

            for position, scenario in enumerate(group["scenarios"], start=1):
                values = relative_to_baseline(scenario_data, baseline, scenario, metric)
                if values:
                    box_values.append(values)
                    box_positions.append(position)

            plot_group(ax, box_values, box_positions, group["color"])
            set_axis_style(ax, metric_title)
            ax.set_ylim(*metric_ylim[metric])
            ax.set_xlim(0.4, len(group["scenarios"]) + 0.6)
            ax.set_xticks(range(1, len(group["scenarios"]) + 1))
            ax.set_xticklabels(group["xticklabels"], fontsize=9)
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=12)

            if col_idx == 0:
                ax.text(
                    -0.5,
                    0.5,
                    group["label"],
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    va="center",
                    ha="center",
                )

    for ax in axes[-1, :]:
        ax.tick_params(axis="x", labelrotation=0)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_file = FIGURE_DIR / f"{FIGURE_STEM}.png"
    stale_pdf_file = FIGURE_DIR / f"{FIGURE_STEM}.pdf"
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    fig.savefig(png_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    if stale_pdf_file.exists():
        stale_pdf_file.unlink()

    if "FCS5" in scenario_data:
        fcs5_cost = pd.Series(relative_to_baseline(scenario_data, baseline, "FCS5", "cost"))
        print(f"FCS 5x: {(fcs5_cost < -10).mean() * 100:.1f}% of cities have cost decreases >10%.")
    if "BC2" in scenario_data:
        bc2_emission = pd.Series(relative_to_baseline(scenario_data, baseline, "BC2", "emission"))
        print(f"BC 2x: {(bc2_emission < -10).mean() * 100:.1f}% of cities have emission decreases >10%.")

    print(f"Saved figure: {png_file}")


if __name__ == "__main__":
    main()
