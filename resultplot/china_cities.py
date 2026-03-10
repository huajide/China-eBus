import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import numpy as np
import warnings

# 设置Nature期刊风格
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['axes.linewidth'] = 0.8


def load_and_prepare_data():
    """
    加载并准备数据
    返回处理后的地理数据
    """
    # 读取数据
    nine_dash_line = gpd.read_file(r'E:\Manufacture\QGIS\China\boundarySea.shp')
    cities = gpd.read_file(r'E:\Manufacture\QGIS\China\City.shp')

    # 转换坐标系统
    projected_crs = "EPSG:4547"
    nine_dash_line = nine_dash_line.to_crs(projected_crs)
    cities = cities.to_crs(projected_crs)

    # 创建省边界
    province_boundaries = cities.dissolve(by='省').to_crs(projected_crs)

    # 创建国家边界（所有省边界的并集）
    country_boundary = province_boundaries.dissolve().to_crs(projected_crs)

    return nine_dash_line, cities, province_boundaries, country_boundary


def _plot_base_map(ax, cities, province_boundaries, country_boundary,
                   show_province_colors=True, city_color='lightgray'):
    """
    绘制基础地图元素的公用函数

    参数:
    ax: matplotlib轴对象
    cities: 城市数据
    province_boundaries: 省边界数据
    country_boundary: 国家边界数据
    show_province_colors: 是否按省着色城市
    city_color: 城市统一颜色（当show_province_colors为False时使用）
    """
    # 绘制国家边界（浅蓝色半透明影子，仅在主图中显示）
    if ax.get_figure().get_size_inches()[0] > 10:  # 判断是否为主图
        country_boundary_shadow = country_boundary.copy()
        country_boundary_shadow.geometry = country_boundary.geometry.buffer(30000)
        country_boundary_shadow.plot(ax=ax, facecolor='lightblue', edgecolor='none', alpha=0.3)

    # 绘制城市边界
    if show_province_colors:
        # 绘制城市边界（按省着色）
        cities.plot(ax=ax, column='省', cmap='Set3', edgecolor='white', linewidth=0.3, alpha=0.8)
    else:
        # 绘制城市边界（统一颜色）
        cities.plot(ax=ax, color=city_color, edgecolor='white', linewidth=0.3, alpha=0.8)

    # 绘制国家边界
    country_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2)

    # 绘制省边界
    if show_province_colors:
        # 正常地图使用深蓝色粗线
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='darkblue', linewidth=1.5)
    else:
        # 高亮地图使用深灰色细线
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=0.8)

    # 设置坐标轴范围
    ax.set_ylim(bottom=1.9e6, top=6.12e6)
    ax.set_xlim(right=2.5e6)

    # 移除坐标轴标签和刻度
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_inset_map(ax, cities, province_boundaries, country_boundary, nine_dash_line,
                    show_province_colors=True, highlighted_cities=None, city_color='lightgray'):
    """
    绘制小图的公用函数

    参数:
    ax: matplotlib轴对象
    cities: 城市数据
    province_boundaries: 省边界数据
    country_boundary: 国家边界数据
    nine_dash_line: 九段线数据
    show_province_colors: 是否按省着色城市
    highlighted_cities: 高亮城市数据（可选）
    city_color: 城市统一颜色（当show_province_colors为False时使用）
    """
    # 绘制城市边界
    if show_province_colors:
        # 绘制城市边界（按省着色）
        cities.plot(ax=ax, column='省', cmap='Set3', edgecolor='white', linewidth=0.2, alpha=0.8)
    else:
        # 绘制城市边界（统一颜色）
        cities.plot(ax=ax, color=city_color, edgecolor='white', linewidth=0.2, alpha=0.8)

    # 如果有高亮城市，在小图中也绘制
    if highlighted_cities is not None and not show_province_colors:
        highlighted_cities.plot(ax=ax, color='cornflowerblue', edgecolor='navy', linewidth=0.3, alpha=0.8)

    # 小图中的国家边界
    country_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)

    # 小图中的省边界
    if show_province_colors:
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='darkblue', linewidth=1)
    else:
        province_boundaries.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=0.6)

    # 绘制南海九段线（仅在小图中显示）
    nine_dash_line.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)

    # 设置小图坐标轴范围
    ax.set_ylim(top=2.4e6)
    ax.set_xlim(left=-2.2e5, right=1.37e6)

    # 移除小图的坐标轴标签和刻度
    ax.set_xticks([])
    ax.set_yticks([])


def plot_china_map_with_inset(save_path='china_map_normal.png'):
    """
    绘制中国地图，包含南海小图

    参数:
    save_path: 保存路径
    """
    # 加载数据
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()

    # 创建图形和主图
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)

    # 绘制基础地图
    _plot_base_map(ax_main, cities, province_boundaries, country_boundary,
                   show_province_colors=True)

    # 插入小图 - 南海诸岛
    ax_inset = fig.add_axes([0.78, 0.21, 0.11, 0.19])
    _plot_inset_map(ax_inset, cities, province_boundaries, country_boundary, nine_dash_line,
                    show_province_colors=True)

    # 保存为PNG格式
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)


def plot_highlighted_cities(city_codes, save_path='selected_cities.png'):
    """
    绘制中国地图，高亮显示指定城市代码的城市

    参数:
    city_codes: 要高亮显示的城市代码列表
    save_path: 保存路径
    """
    # 加载数据
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()

    # 创建图形
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)

    # 检查是否存在无效的城市代码
    if city_codes:
        valid_city_codes = cities['市代码'].unique()
        invalid_codes = [code for code in city_codes if code not in valid_city_codes]
        if invalid_codes:
            warnings.warn(f"以下城市代码在数据中不存在: {invalid_codes}", UserWarning)

    # 绘制基础地图（灰色城市） - 这里确保绘制的是灰色
    _plot_base_map(ax_main, cities, province_boundaries, country_boundary,
                   show_province_colors=False, city_color='lightgray')

    # 获取高亮城市数据并在之后绘制
    highlighted_cities = None
    if city_codes:
        highlighted_cities = cities[cities['市代码'].isin(city_codes)]
        # 高亮显示指定城市 - 这里绘制红色
        highlighted_cities.plot(ax=ax_main, color='cornflowerblue', edgecolor='navy', linewidth=0.5, alpha=0.8)

    # 插入小图 - 南海诸岛
    ax_inset = fig.add_axes([0.78, 0.21, 0.11, 0.19])
    _plot_inset_map(ax_inset, cities, province_boundaries, country_boundary, nine_dash_line,
                    show_province_colors=False, highlighted_cities=highlighted_cities,
                    city_color='lightgray')

    # 保存为PNG格式
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)


def plot_graded_cities(city_codes, values, variable_name, cmap='Reds',
                       save_path='graded_cities.png',show=True):
    """
    绘制中国地图，根据值对选中城市进行分级着色

    参数:
    city_codes: 要着色的城市代码列表
    values: 对应的值列表
    variable_name: 变量名称（用于图例标题）
    cmap: 颜色映射
    save_path: 保存路径
    """
    # 加载数据
    nine_dash_line, cities, province_boundaries, country_boundary = load_and_prepare_data()

    # 创建图形
    fig = plt.figure(figsize=(12, 12))
    ax_main = fig.add_subplot(111)

    # 检查是否存在无效的城市代码
    if city_codes and values and len(city_codes) == len(values):
        valid_city_codes = cities['市代码'].unique()
        invalid_codes = [code for code in city_codes if code not in valid_city_codes]
        if invalid_codes:
            warnings.warn(f"以下城市代码在数据中不存在: {invalid_codes}", UserWarning)

    # 绘制所有城市（浅灰色背景）
    cities.plot(ax=ax_main, color='lightgray', edgecolor='white', linewidth=0.3, alpha=0.8)

    # 处理选中城市和值
    selected_cities = None
    if city_codes and values and len(city_codes) == len(values):
        # 创建一个临时的GeoDataFrame只包含选中的城市
        selected_cities = cities[cities['市代码'].isin(city_codes)].copy()

        # 创建代码到值的映射
        code_value_map = dict(zip(city_codes, values))

        # 为选中城市添加值列
        selected_cities = selected_cities.copy()
        selected_cities['value'] = selected_cities['市代码'].map(code_value_map)

        # 绘制分级颜色的选中城市
        selected_cities.plot(ax=ax_main, column='value', cmap=cmap,
                             edgecolor='white', linewidth=0.5, alpha=0.8)

    # 绘制国家边界和省边界（调整绘制顺序，确保边界在上层）
    country_boundary.plot(ax=ax_main, facecolor='none', edgecolor='black', linewidth=1.2)
    province_boundaries.plot(ax=ax_main, facecolor='none', edgecolor='#333333', linewidth=0.8)

    # 添加图例（颜色条）- 放在左下角，被大图包住
    if selected_cities is not None:
        # 获取值的范围
        vmin = selected_cities['value'].min()
        vmax = selected_cities['value'].max()

        # 创建颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])

        # 添加颜色条，位置在左下角内部
        cbar = fig.colorbar(sm, ax=ax_main, orientation='horizontal',
                            fraction=0.046, pad=0.04, shrink=0.4, aspect=30,
                            anchor=(0.0, 1.0), panchor=(0.0, 1.0))
        cbar.set_label(variable_name, fontsize=10)

        # 调整颜色条位置到左下角内部
        pos = ax_main.get_position()
        cbar.ax.set_position([pos.x0 + 0.03, pos.y0 + 0.17, pos.width * 0.4, 0.02])

    # 设置坐标轴范围
    ax_main.set_ylim(bottom=1.9e6, top=6.12e6)
    ax_main.set_xlim(right=2.5e6)

    # 移除坐标轴标签和刻度
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # 插入小图 - 南海诸岛
    ax_inset = fig.add_axes([0.78, 0.313, 0.11, 0.19])

    # 小图中绘制所有城市（浅灰色背景）
    cities.plot(ax=ax_inset, color='lightgray', edgecolor='white', linewidth=0.2, alpha=0.8)

    # 如果有选中城市，在小图中也绘制
    if selected_cities is not None:
        selected_cities.plot(ax=ax_inset, column='value', cmap=cmap,
                             edgecolor='white', linewidth=0.3, alpha=0.8)

    # 小图中的边界
    country_boundary.plot(ax=ax_inset, facecolor='none', edgecolor='black', linewidth=0.8)
    province_boundaries.plot(ax=ax_inset, facecolor='none', edgecolor='#333333', linewidth=0.6)

    # 绘制南海九段线
    nine_dash_line.plot(ax=ax_inset, facecolor='none', edgecolor='black', linewidth=0.8)

    # 设置小图坐标轴范围
    ax_inset.set_ylim(top=2.4e6)
    ax_inset.set_xlim(left=-2.2e5, right=1.37e6)

    # 移除小图的坐标轴标签和刻度
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # 保存为PNG格式
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=True)
    plt.close()


# 使用示例
if __name__ == "__main__":
    # 绘制带南海小图的中国地图
    # plot_china_map_with_inset()

    # 绘制高亮城市地图示例
    # plot_highlighted_cities([110000, 310000, 440100, 500000, 610100, 999999])

    # 绘制分级颜色城市地图示例
    # plot_graded_cities([110000, 310000, 440100, 500000, 610100], [100, 200, 150, 300, 250],
    #                    "Population", cmap='Reds')

    # 绘制224城市
    cities = pd.read_csv(r"..\data\224cities.csv")
    city_list = cities['citycode'].tolist()
    plot_highlighted_cities(city_list)
