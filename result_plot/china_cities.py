import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import warnings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['axes.linewidth'] = 0.8

def load_and_prepare_data():
    nine_dash_line = gpd.read_file('E:\\Manufacture\\QGIS\\China\\boundarySea.shp')
    cities = gpd.read_file('E:\\Manufacture\\QGIS\\China\\City.shp')
    projected_crs = 'EPSG:4547'
    nine_dash_line = nine_dash_line.to_crs(projected_crs)
    cities = cities.to_crs(projected_crs)
    province_boundaries = cities.dissolve(by='省').to_crs(projected_crs)
    country_boundary = province_boundaries.dissolve().to_crs(projected_crs)
    return (nine_dash_line, cities, province_boundaries, country_boundary)

def _plot_base_map(ax, cities, province_boundaries, country_boundary, show_province_colors=True, city_color='lightgray'):
    if ax.get_figure().get_size_inches()[0] > 10:
        country_boundary_shadow = country_boundary.copy()
        country_boundary_shadow.geometry = country_boundary.geometry.buffer(30000)
        country_boundary_shadow.plot(ax=ax, facecolor='lightblue', edgecolor='none', alpha=0.3)
    if show_province_colors:
        cities.plot(ax=ax, column='省', cmap='Set3', edgecolor='white', linewidth=0.3, alpha=0.8)
    else:
        cities.plot(ax=ax, color=city_color, edgecolor='white', linewidth=0.3, alpha=0.8)
    country_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2)
    if show_province_colors:
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='darkblue', linewidth=1.5)
    else:
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=0.8)
    ax.set_ylim(bottom=1900000.0, top=6120000.0)
    ax.set_xlim(right=2500000.0)
    ax.set_xticks([])
    ax.set_yticks([])

def _plot_inset_map(ax, cities, province_boundaries, country_boundary, nine_dash_line, show_province_colors=True, highlighted_cities=None, city_color='lightgray'):
    if show_province_colors:
        cities.plot(ax=ax, column='省', cmap='Set3', edgecolor='white', linewidth=0.2, alpha=0.8)
    else:
        cities.plot(ax=ax, color=city_color, edgecolor='white', linewidth=0.2, alpha=0.8)
    if highlighted_cities is not None and (not show_province_colors):
        highlighted_cities.plot(ax=ax, color='cornflowerblue', edgecolor='navy', linewidth=0.3, alpha=0.8)
    country_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)
    if show_province_colors:
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='darkblue', linewidth=1)
    else:
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=0.6)
    nine_dash_line.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)
    ax.set_ylim(top=2400000.0)
    ax.set_xlim(left=-220000.0, right=1370000.0)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_china_map_with_inset(save_path='china_map_normal.png'):
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)
    _plot_base_map(ax_main, cities, province_boundaries, country_boundary, show_province_colors=True)
    ax_inset = fig.add_axes([0.78, 0.21, 0.11, 0.19])
    _plot_inset_map(ax_inset, cities, province_boundaries, country_boundary, nine_dash_line, show_province_colors=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)

def plot_highlighted_cities(city_codes, save_path='selected_cities.png'):
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)
    if city_codes:
        valid_city_codes = cities['市代码'].unique()
        invalid_codes = [code for code in city_codes if code not in valid_city_codes]
        if invalid_codes:
            warnings.warn(f'The following city codes are not found in the dataset: {invalid_codes}', UserWarning)
    _plot_base_map(ax_main, cities, province_boundaries, country_boundary, show_province_colors=False, city_color='lightgray')
    highlighted_cities = None
    if city_codes:
        highlighted_cities = cities[cities['市代码'].isin(city_codes)]
        highlighted_cities.plot(ax=ax_main, color='cornflowerblue', edgecolor='navy', linewidth=0.5, alpha=0.8)
    ax_inset = fig.add_axes([0.78, 0.21, 0.11, 0.19])
    _plot_inset_map(ax_inset, cities, province_boundaries, country_boundary, nine_dash_line, show_province_colors=False, highlighted_cities=highlighted_cities, city_color='lightgray')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)

def plot_graded_cities(city_codes, values, variable_name, cmap='Reds', save_path='graded_cities.png', show=True):
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)
    if city_codes and values and (len(city_codes) == len(values)):
        valid_city_codes = cities['市代码'].unique()
        invalid_codes = [code for code in city_codes if code not in valid_city_codes]
        if invalid_codes:
            warnings.warn(f'The following city codes are not found in the dataset: {invalid_codes}', UserWarning)
    cities.plot(ax=ax_main, color='lightgray', edgecolor='white', linewidth=0.3, alpha=0.8)
    selected_cities = None
    if city_codes and values and (len(city_codes) == len(values)):
        selected_cities = cities[cities['市代码'].isin(city_codes)].copy()
        code_value_map = dict(zip(city_codes, values))
        selected_cities = selected_cities.copy()
        selected_cities['value'] = selected_cities['市代码'].map(code_value_map)
        selected_cities.plot(ax=ax_main, column='value', cmap=cmap, edgecolor='white', linewidth=0.5, alpha=0.8)
    country_boundary.plot(ax=ax_main, facecolor='none', edgecolor='black', linewidth=1.2)
    province_boundaries.plot(ax=ax_main, facecolor='none', edgecolor='#333333', linewidth=0.8)
    if selected_cities is not None:
        vmin = selected_cities['value'].min()
        vmax = selected_cities['value'].max()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_main, orientation='horizontal', fraction=0.046, pad=0.04, shrink=0.4, aspect=30, anchor=(0.0, 1.0), panchor=(0.0, 1.0))
        cbar.set_label(variable_name, fontsize=10)
        pos = ax_main.get_position()
        cbar.ax.set_position([pos.x0 + 0.03, pos.y0 + 0.17, pos.width * 0.4, 0.02])
    ax_main.set_ylim(bottom=1900000.0, top=6120000.0)
    ax_main.set_xlim(right=2500000.0)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_inset = fig.add_axes([0.78, 0.313, 0.11, 0.19])
    cities.plot(ax=ax_inset, color='lightgray', edgecolor='white', linewidth=0.2, alpha=0.8)
    if selected_cities is not None:
        selected_cities.plot(ax=ax_inset, column='value', cmap=cmap, edgecolor='white', linewidth=0.3, alpha=0.8)
    country_boundary.plot(ax=ax_inset, facecolor='none', edgecolor='black', linewidth=0.8)
    province_boundaries.plot(ax=ax_inset, facecolor='none', edgecolor='#333333', linewidth=0.6)
    nine_dash_line.plot(ax=ax_inset, facecolor='none', edgecolor='black', linewidth=0.8)
    ax_inset.set_ylim(top=2400000.0)
    ax_inset.set_xlim(left=-220000.0, right=1370000.0)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=True)
    plt.close()
if __name__ == '__main__':
    cities = pd.read_csv('..\\data\\224cities.csv')
    city_list = cities['citycode'].tolist()
    plot_highlighted_cities(city_list)
