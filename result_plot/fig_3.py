from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans

import fig_2 as scenario


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIGURE_DIR = SCRIPT_DIR / "fig_3"

GROUP_LABELS = {
    1: "Small",
    2: "Medium",
    3: "Large",
}

GROUP_COLORS = {
    1: "#9ecae1",
    2: "#3182bd",
    3: "#08519c",
}

SELECTED_PANELS = [
    ("Fast charging speed", "cost", "Cost", 2.0),
    ("Fast charging speed", "built_num", "Charging stations", 1.7),
    ("Fast charging speed", "slow_charger", "Slow chargers", 1.0),
    ("Battery capacity", "built_num", "Charging stations", 1.45),
]

INDICATORS = [
    ("cost", "System costs\nper km\n(M yuan/km)"),
    ("emission", "GHG emissions\nper km\n(t/km)"),
    ("built_num", "Charging stations\nper km\n(km$^{-1}$)"),
    ("fast_charger", "Fast chargers\nper km\n(km$^{-1}$)"),
    ("slow_charger", "Slow chargers\nper km\n(km$^{-1}$)"),
    ("vehicle_count", "Electric buses\nper km\n(km$^{-1}$)"),
]


plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def cluster_cities(static_data: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    data = static_data[["city", "network_length_km"]].copy()
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
        data[["network_length_km"]].values
    )
    data["raw_group"] = labels
    group_means = data.groupby("raw_group")["network_length_km"].mean().sort_values()
    ordered_groups = {old_group: new_group for new_group, old_group in enumerate(group_means.index, start=1)}
    data["group"] = data["raw_group"].map(ordered_groups)
    data.loc[data["group"].eq(4), "group"] = 3
    return data[["city", "network_length_km", "group"]]


def read_baseline_output() -> pd.DataFrame:
    output = pd.read_csv(DATA_DIR / "224cities_output.csv")
    required = {
        "city_name",
        "knee_cost",
        "knee_emission",
        "knee_built_cs",
        "knee_fast_per_cs",
        "knee_slow_per_cs",
        "knee_extra_large",
        "knee_extra_medium",
        "knee_extra_small",
    }
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(f"224cities_output.csv is missing columns: {', '.join(sorted(missing))}")
    return output


def baseline_indicators(static_data: pd.DataFrame, output_data: pd.DataFrame) -> pd.DataFrame:
    merged = output_data.merge(static_data, left_on="city_name", right_on="city", how="inner")
    if len(merged) != len(output_data):
        print(f"Matched {len(merged)} of {len(output_data)} output cities with static indicators.")

    network_length = merged["network_length_km"].astype(float)
    extra_vehicles = (
        merged["knee_extra_large"].astype(float)
        + merged["knee_extra_medium"].astype(float)
        + merged["knee_extra_small"].astype(float)
    )
    return pd.DataFrame(
        {
            "city": merged["city"],
            "cost": merged["knee_cost"].astype(float) / network_length,
            "emission": merged["knee_emission"].astype(float) / network_length,
            "built_num": merged["knee_built_cs"].astype(float) / network_length,
            "fast_charger": (
                merged["knee_fast_per_cs"].astype(float) * merged["knee_built_cs"].astype(float) / network_length
            ),
            "slow_charger": (
                merged["knee_slow_per_cs"].astype(float) * merged["knee_built_cs"].astype(float) / network_length
            ),
            "vehicle_count": (merged["vehicle_count"].astype(float) + extra_vehicles) / network_length,
            "network_length_km": network_length,
        }
    )


def plot_clustered_indicators_boxplot(static_data: pd.DataFrame, output_data: pd.DataFrame) -> Path:
    clusters = cluster_cities(static_data)
    indicators = baseline_indicators(static_data, output_data).merge(clusters[["city", "group"]], on="city")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.6,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
        }
    )
    fig, axes = plt.subplots(1, len(INDICATORS), figsize=(18.5, 4.25))
    colors = [GROUP_COLORS[1], GROUP_COLORS[2], GROUP_COLORS[3]]
    labels = [GROUP_LABELS[1], GROUP_LABELS[2], GROUP_LABELS[3]]
    for ax, (indicator, title) in zip(axes, INDICATORS):
        group_data = [indicators.loc[indicators["group"].eq(group_id), indicator].dropna() for group_id in (1, 2, 3)]
        box = ax.boxplot(
            group_data,
            labels=labels,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            boxprops={"linewidth": 1.45, "edgecolor": "black"},
            medianprops={"linewidth": 1.7, "color": "black"},
            whiskerprops={"linewidth": 1.1, "color": "black"},
            capprops={"linewidth": 1.1, "color": "black"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.74)
        ax.set_title(title, pad=9)
        ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.9)
        ax.tick_params(axis="x", rotation=0)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_file = FIGURE_DIR / "clustered_indicators_boxplot.png"
    plt.tight_layout(rect=[0.01, 0.03, 1.0, 1.0], w_pad=0.7)
    fig.savefig(png_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return png_file


def load_all_scenarios(static_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scenario_data = {}
    for name, path in scenario.SCENARIOS.items():
        if not path.exists():
            print(f"Missing scenario directory: {name} -> {path}")
            continue
        scenario_data[name] = scenario.load_scenario_data(name, path, static_data)
    return scenario_data


def relative_change_frame(
    scenario_data: dict[str, pd.DataFrame],
    baseline: pd.DataFrame,
    scenario_name: str,
    metric: str,
) -> pd.DataFrame:
    if scenario_name == "baseline":
        return baseline[["city"]].assign(change=0.0)
    current = scenario_data.get(scenario_name, pd.DataFrame())
    if current.empty:
        return pd.DataFrame(columns=["city", "change"])
    merged = baseline[["city", metric]].merge(
        current[["city", metric]],
        on="city",
        suffixes=("_baseline", "_scenario"),
        how="inner",
    )
    baseline_values = merged[f"{metric}_baseline"].astype(float)
    scenario_values = merged[f"{metric}_scenario"].astype(float)
    merged["change"] = 0.0
    valid = baseline_values.ne(0)
    merged.loc[valid, "change"] = (
        (scenario_values.loc[valid] - baseline_values.loc[valid]) / baseline_values.loc[valid] * 100
    )
    return merged[["city", "change"]]


def group_mean_changes(
    scenario_data: dict[str, pd.DataFrame],
    baseline: pd.DataFrame,
    clusters: pd.DataFrame,
    scenario_name: str,
    metric: str,
) -> dict[int, float]:
    changes = relative_change_frame(scenario_data, baseline, scenario_name, metric)
    if changes.empty:
        return {}
    merged = changes.merge(clusters[["city", "group"]], on="city", how="inner")
    return merged.groupby("group")["change"].mean().to_dict()


def sensitivity_group_by_label(label: str) -> dict:
    for group in scenario.SENSITIVITY_GROUPS:
        if group["label"] == label:
            return group
    raise KeyError(f"Missing sensitivity group: {label}")


def compact_xticklabels(sensitivity_group: dict) -> list[str]:
    return [label.split("\n", 1)[0] for label in sensitivity_group["xticklabels"]]


def expanded_ylim(values: list[float], scale: float) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return -1.0, 1.0
    max_abs = max(max(abs(min(finite_values)), abs(max(finite_values))) * scale, 0.1)
    if max_abs <= 3:
        step = 0.5
    elif max_abs <= 8:
        step = 1
    elif max_abs <= 20:
        step = 2
    elif max_abs <= 60:
        step = 5
    else:
        step = 10
    limit = math.ceil(max_abs / step) * step
    return -limit, limit


def plot_selected_mean_changes(static_data: pd.DataFrame) -> Path:
    clusters = cluster_cities(static_data)
    scenario_data = load_all_scenarios(static_data)
    baseline = scenario_data.get("baseline", pd.DataFrame())
    if baseline.empty:
        raise RuntimeError(f"Baseline data is missing or empty: {scenario.SCENARIOS['baseline']}")
    baseline = baseline.sort_values("city").reset_index(drop=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.8,
            "ytick.major.size": 4.8,
        }
    )
    fig, axes = plt.subplots(1, len(SELECTED_PANELS), figsize=(8.0, 2.15))
    legend_handles = []
    for ax, (scenario_label, metric, metric_label, ylim_scale) in zip(axes, SELECTED_PANELS):
        sensitivity_group = sensitivity_group_by_label(scenario_label)
        x_positions = list(range(len(sensitivity_group["scenarios"])))
        y_values_for_axis = []
        for group_id in (1, 2, 3):
            y_values = []
            for scenario_name in sensitivity_group["scenarios"]:
                means = group_mean_changes(scenario_data, baseline, clusters, scenario_name, metric)
                y_values.append(means.get(group_id, float("nan")))
            line = ax.plot(
                x_positions,
                y_values,
                marker="o",
                markersize=3.2,
                linewidth=1.35,
                color=GROUP_COLORS[group_id],
                label=GROUP_LABELS[group_id],
            )[0]
            if len(legend_handles) < 3:
                legend_handles.append(line)
            y_values_for_axis.extend(y_values)
        ax.set_ylabel(metric_label, fontsize=8.8, labelpad=2)
        ax.set_xlabel(scenario_label, fontsize=8.8, labelpad=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(compact_xticklabels(sensitivity_group), fontsize=7.8)
        ax.set_xlim(-0.2, len(x_positions) - 0.8)
        ax.set_ylim(*expanded_ylim(y_values_for_axis, ylim_scale))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(True, linestyle=":", alpha=0.55, linewidth=0.65)
        ax.set_box_aspect(1)
        ax.tick_params(axis="y", labelsize=7.8)
        ax.tick_params(axis="x", labelsize=7.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_linewidth(1.25)
        ax.spines["bottom"].set_linewidth(1.25)
    fig.text(0.012, 0.955, "Bus Network Scale", ha="left", va="center", fontsize=8.3, fontweight="bold")
    fig.legend(
        handles=legend_handles,
        labels=[GROUP_LABELS[group_id] for group_id in (1, 2, 3)],
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=8.3,
        bbox_to_anchor=(0.14, 0.985),
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.1,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_file = FIGURE_DIR / f"scenario_changes_selected_mean_alt_{scenario.RUN_TAG}.png"
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.82, wspace=0.18)
    fig.savefig(png_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return png_file


def main() -> None:
    static_data = pd.read_csv(DATA_DIR / "224city_indicators.csv")
    output_data = read_baseline_output()
    clustered_file = plot_clustered_indicators_boxplot(static_data, output_data)
    print(f"Saved clustered indicator figure: {clustered_file}")
    selected_file = plot_selected_mean_changes(static_data)
    print(f"Saved selected scenario figure: {selected_file}")


if __name__ == "__main__":
    main()
