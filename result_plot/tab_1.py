from __future__ import annotations

import argparse
from pathlib import Path

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fig_2 import find_knee_solution,row_by_solution

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = DATA_DIR / "output" / "results"
FIGURES_ROOT = DATA_DIR / "output" / "figures"

ARCHIVE_ROOT = RESULTS_ROOT / "baseline"
VS_ROOT = DATA_DIR / "input" / "vs_parking_nodeid"
OUTPUT_DIR = FIGURES_ROOT
X_VARS = ['route_avg_distance', 'network_length_km', 'repetition_rate', 'average_stop_spacing', 'trip_cv', 'operation_total_distance_ratio', 'avg_speed', 'route_terminal_ratio', 'e_price']

Y_NAMES = ['cost_per_km', 'emission_per_km', 'cs_per_km', 'fast_chargers_per_km', 'slow_chargers_per_km', 'ev_per_km']

_SUPERSCRIPT = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''

def format_scientific(value: float, p_value: float | None=None) -> str:
    """Format coefficients as a × 10ⁿ, with the mantissa kept to two decimals."""
    if pd.isna(value):
        return ''
    if value == 0:
        formatted = '0.00 × 10⁰'
    else:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10 ** exponent
        if abs(round(mantissa, 2)) >= 10:
            mantissa /= 10
            exponent += 1
        formatted = f'{mantissa:.2f} × 10{str(exponent).translate(_SUPERSCRIPT)}'
    if p_value is not None:
        formatted += significance_stars(p_value)
    return formatted

def prepare_regression_data(indicators: pd.DataFrame, outputs: pd.DataFrame):
    merged_data = pd.merge(indicators, outputs, left_on='city', right_on='city_name')
    merged_data['trip_cv'] = np.sqrt(merged_data['variance']) / merged_data['trip_number'] * 24
    merged_data['route_avg_operation_distance'] = merged_data['operation_distance'] / merged_data['route_count']
    x_data = merged_data[X_VARS]
    y_data = pd.DataFrame({'cost_per_km': merged_data['knee_cost'] / merged_data['network_length_km'], 'emission_per_km': merged_data['knee_emission'] / merged_data['network_length_km'], 'cs_per_km': merged_data['knee_built_cs'] / merged_data['network_length_km'], 'fast_chargers_per_km': merged_data['knee_fast_per_cs'] * merged_data['knee_built_cs'] / merged_data['network_length_km'], 'slow_chargers_per_km': merged_data['knee_slow_per_cs'] * merged_data['knee_built_cs'] / merged_data['network_length_km'], 'ev_per_km': (merged_data['knee_extra_large'] + merged_data['knee_extra_medium'] + merged_data['knee_extra_small'] + merged_data['vehicle_count']) / merged_data['network_length_km']})
    clean_data = pd.concat([x_data, y_data], axis=1).dropna()
    return (clean_data[X_VARS], clean_data[Y_NAMES])

def fit_ols_models(x_data: pd.DataFrame, y_data: pd.DataFrame, *, standardize_x: bool):
    if standardize_x:
        x_values = StandardScaler().fit_transform(x_data)
    else:
        x_values = x_data.to_numpy()
    x_values = sm.add_constant(x_values)
    feature_names = ['const'] + X_VARS
    models = {}
    for y_name in Y_NAMES:
        model = sm.OLS(y_data[y_name].to_numpy(), x_values).fit()
        models[y_name] = {'r2': model.rsquared, 'params': pd.Series(model.params, index=feature_names), 'pvalues': pd.Series(model.pvalues, index=feature_names)}
    return models

def build_coef_rank_table(raw_models: dict, rank_models: dict) -> pd.DataFrame:
    table = pd.DataFrame(index=['R2', 'const'] + X_VARS)
    for y_name in Y_NAMES:
        coef_col = f'{y_name}_coef'
        rank_col = f'{y_name}_rank'
        table[coef_col] = ''
        table[rank_col] = '-'
        table.loc['R2', coef_col] = f"{raw_models[y_name]['r2']:.4f}"
        raw_params = raw_models[y_name]['params']
        raw_pvalues = raw_models[y_name]['pvalues']
        for feature in ['const'] + X_VARS:
            table.loc[feature, coef_col] = format_scientific(raw_params.loc[feature], raw_pvalues.loc[feature])
        standardized_abs = rank_models[y_name]['params'].loc[X_VARS].abs()
        ranks = standardized_abs.rank(ascending=False, method='min').astype(int)
        for feature in X_VARS:
            table.loc[feature, rank_col] = int(ranks.loc[feature])
    return table

def write_excel(table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        table.to_excel(writer, sheet_name='table_1_coef_rank')
        worksheet = writer.sheets['table_1_coef_rank']
        worksheet.freeze_panes = 'B2'
        for column_cells in worksheet.columns:
            max_len = max((len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells))
            column_letter = column_cells[0].column_letter
            worksheet.column_dimensions[column_letter].width = min(max(max_len + 2, 10), 24)
def read_outputs(indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    updated_indicators = indicators.copy()
    rows = []
    for city in indicators['city']:
        city_dir = ARCHIVE_ROOT / city
        objectives = pd.read_csv(city_dir / 'archive_objs.csv', index_col=0)
        variables = pd.read_csv(city_dir / 'archive_vars.csv', index_col=0)
        diagnostics = pd.read_csv(city_dir / 'archive_cvs.csv', index_col=0)
        solution = find_knee_solution(objectives)
        obj = row_by_solution(objectives, solution)
        var = row_by_solution(variables, solution)
        cv = row_by_solution(diagnostics, solution)
        n_stations = len(var) - 3
        if len(cv) != 2 * n_stations + 6 or cv.iloc[0] > 0:
            raise ValueError(f'Invalid knee diagnostics: {city}')
        built = var.iloc[:n_stations].sum()
        fast = -cv.iloc[1:1 + n_stations].sum()
        slow = -cv.iloc[1 + n_stations:1 + 2 * n_stations].sum()
        extras = -cv.iloc[-3:]
        updated_indicators.loc[updated_indicators['city'].eq(city), 'vehicle_count'] = pd.read_csv(VS_ROOT / f'{city}.csv', usecols=['v_name'])['v_name'].nunique()
        rows.append({'city_name': city, 'knee_cost': -obj.iloc[0], 'knee_emission': -obj.iloc[1], 'knee_built_cs': built, 'knee_fast_per_cs': fast / built if built else 0, 'knee_slow_per_cs': slow / built if built else 0, 'knee_extra_large': extras.iloc[0], 'knee_extra_medium': extras.iloc[1], 'knee_extra_small': extras.iloc[2]})
    return (updated_indicators, pd.DataFrame(rows))

def model_ranks(models: dict, outcome: str) -> pd.Series:
    return models[outcome]['params'].loc[X_VARS].abs().rank(ascending=False, method='min').astype(int)
def plot_correlation_matrix(indicators, outputs, output_dir='fig_1'):
    """Plot correlation matrix."""
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
    """Calculate variance inflation factors."""
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

def main():
    global ARCHIVE_ROOT
    parser = argparse.ArgumentParser(description="Fit the nine-predictor OLS models and report coefficient rankings")
    parser.add_argument("--results-root",type=Path,default=RESULTS_ROOT)
    parser.add_argument("--output",type=Path,default=OUTPUT_DIR)
    parser.add_argument("--correlation",action="store_true")
    parser.add_argument("--vif",action="store_true")
    args = parser.parse_args()
    ARCHIVE_ROOT = args.results_root / "baseline"
    indicators,outputs = read_outputs(pd.read_csv(DATA_DIR / "224city_indicators.csv"))
    x,y = prepare_regression_data(indicators,outputs)
    raw = fit_ols_models(x,y,standardize_x=False)
    scaled = fit_ols_models(x,y,standardize_x=True)
    table = build_coef_rank_table(raw,scaled)
    args.output.mkdir(parents=True,exist_ok=True)
    plot_data = args.output / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    table.to_csv(plot_data / "Table_1.csv")
    write_excel(table,plot_data / "Table_1.xlsx")
    outputs.to_csv(plot_data / "baseline_knee_points.csv",index=False)
    pd.DataFrame([dict(outcome=k,predictor=f,coefficient=m["params"][f],p=m["pvalues"][f],rank=int(model_ranks(scaled,k)[f]) if f != "const" else None,R2=m["r2"])
        for k,m in raw.items() for f in m["params"].index]).to_csv(plot_data / "OLS_numeric.csv",index=False)
    if args.correlation: plot_correlation_matrix(indicators,outputs,output_dir=str(args.output))
    if args.vif: calculate_vif(indicators).to_csv(plot_data / "VIF.csv",index=False)

if __name__ == "__main__":
    main()
