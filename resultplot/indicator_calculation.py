import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preparation.duration import circuity_calculation, find_farthest_point, start_end_point

import json
import time
import requests
from requests import RequestException
import pandas as pd
import re
from pyproj import CRS
import ast
import math
from dateutil import parser
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
from shapely.ops import unary_union
from datetime import datetime, timedelta
from centerline.geometry import Centerline

import rasterio
from rasterio.mask import mask
import numpy as np

# 全局变量，用于存储预加载的数据
ZONE_GDF = None
POPULATION_RASTER = None


def initialize_global_data():
    """
    初始化全局数据
    """
    global ZONE_GDF, POPULATION_RASTER

    if ZONE_GDF is None:
        # 加载市辖区数据
        ZONE_GDF = gpd.read_file(r'E:\Manufacture\QGIS\China\2023年初县矢量UTM.shp')
        ZONE_GDF = ZONE_GDF.set_crs(epsg=32650)
        ZONE_GDF = ZONE_GDF.to_crs(epsg=4547)

    if POPULATION_RASTER is None:
        # 打开人口栅格数据
        POPULATION_RASTER = rasterio.open(r"E:\Manufacture\QGIS\China\chn_ppp_2020_UNadj_constrained_4547.tif")


def calculate_population_coverage(city_name, province_name, STOP_GDF):
    """
    计算特定城市公交站点缓冲区在市辖区内的覆盖人口比例

    参数:
    city_name: 城市名称
    province_name: 省份名称
    STOP_GDF: 站点GeoDataFrame

    返回:
    tuple: (覆盖率, 总人口)
    """

    # 确保全局数据已加载
    global ZONE_GDF, POPULATION_RASTER
    if ZONE_GDF is None or POPULATION_RASTER is None:
        initialize_global_data()

    # 步骤1: 使用全局的市辖区数据
    zone_gdf = ZONE_GDF

    # 筛选目标城市的市辖区
    city_level = ['上海市', '北京市', '重庆市', '天津市']
    special = ['香港特别行政区', '澳门特别行政区']
    entity_city = ['中山市', '东莞市', '嘉峪关市', '儋州市']

    if city_name in special:
        sub_zone = zone_gdf[zone_gdf['省级'] == city_name].reset_index(drop=True)
    elif province_name in city_level:
        sub_zone = zone_gdf[(zone_gdf['省级'] == city_name) & (zone_gdf['县级类'] == '市辖区')].reset_index(
            drop=True)
    elif city_name in entity_city:
        sub_zone = zone_gdf[zone_gdf['地级'] == city_name].reset_index(drop=True)
    else:
        sub_zone = zone_gdf[(zone_gdf['地级'] == city_name) & (zone_gdf['县级类'] == '市辖区')].reset_index(
            drop=True)

    if len(sub_zone) == 0:
        print(f"未找到 {city_name} 的市辖区数据")
        return None, None

    # 合并所有市辖区为一个区域
    sub_zone['geometry'] = sub_zone['geometry'].buffer(0)
    merged_zone_polygon = sub_zone.unary_union
    sub_zone_gdf = gpd.GeoDataFrame(geometry=[merged_zone_polygon], crs=zone_gdf.crs)
    sub_zone_gdf = sub_zone_gdf.to_crs(epsg=4547)

    # 统计市辖区总人口
    out_image, out_transform = mask(POPULATION_RASTER, sub_zone_gdf.geometry, crop=True, nodata=0)
    total_population = np.nansum(out_image)

    # 创建500米缓冲区并融合
    # 需要转换到合适的投影坐标系来创建准确的缓冲区
    stop_gdf_projected = STOP_GDF.to_crs(epsg=4547)  # 转换到UTM坐标系
    stop_gdf_projected['geometry'] = stop_gdf_projected['geometry'].buffer(500)  # 500米缓冲区
    stop_buffer_union = stop_gdf_projected.unary_union  # 融合所有缓冲区

    # 转换回地理坐标系
    stop_buffer_gdf = gpd.GeoDataFrame(geometry=[stop_buffer_union], crs=stop_gdf_projected.crs)

    # 步骤3: 保留在市辖区内的缓冲区区域
    # 计算缓冲区与市辖区的交集
    intersection = gpd.overlay(stop_buffer_gdf, sub_zone_gdf, how='intersection')

    # 步骤4: 统计交集区域的人口
    if len(intersection) > 0:
        try:
            out_image, out_transform = mask(POPULATION_RASTER, intersection.geometry, crop=True, nodata=0)
            covered_population = np.nansum(out_image)
        except ValueError as e:
            if "Input shapes do not overlap raster" in str(e):
                print(f"{city_name}: 缓冲区与人口栅格无重叠区域")
                covered_population = 0
            else:
                raise e
    else:
        covered_population = 0

    # 计算覆盖率
    coverage_ratio = covered_population / total_population if total_population > 0 else 0

    print(
        f"{city_name}: {total_population:.0f} people in downtown and {coverage_ratio:.2%} are covered by bus transit")
    return coverage_ratio, total_population


def calculate_distance(p1, p2):
    """ 计算两个点之间的欧氏距离 """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calculate_connectivity(stop_df, allow_shared_stops=True):
    """
    stop_df (gpd.GeoDataFrame): 包含 'stop_id', 'route_name', 'sequence', 'geometry' 的站点数据。
    allow_shared_stops (bool): 是否允许不同线路共享同一站点作为连接边，默认 True。
    float: 连通度 = 总边数 / 唯一节点数
    """
    if allow_shared_stops:
        link_count = stop_df.groupby('route_name').size() - 1
        total_links = link_count.sum()
    else:
        # 新方法：基于 stop_id + route_name 构建唯一边，并去重
        edges_set = set()
        for route_id, group in stop_df.groupby('route_name'):
            sorted_group = group.sort_values(by='sequence').reset_index()
            for i in range(len(sorted_group) - 1):
                stop_id1 = sorted_group.iloc[i]['stop_id']
                stop_id2 = sorted_group.iloc[i + 1]['stop_id']
                # 使用 frozenset 来确保顺序不影响唯一性
                edge = frozenset({stop_id1, stop_id2})
                edges_set.add(edge)
        total_links = len(edges_set)
    unique_stops = stop_df['stop_id'].nunique()
    return total_links / unique_stops if unique_stops > 0 else 0


def parse_timetable(timetables):
    if isinstance(timetables, str):
        return ast.literal_eval(timetables)
    return timetables


def calculate_peak_headway_max_per_hour(times_list):
    # 解析时间字符串为 datetime 对象
    times = []
    for t in times_list:
        try:
            dt = datetime.strptime(t, "%H:%M")
            times.append(dt)
        except Exception:
            continue

    times.sort()
    # 按小时分组统计
    hourly_intervals = {}

    for hour in range(23):
        hour_times = [t for t in times if t.hour == hour]
        if len(hour_times) < 2:
            hourly_intervals[hour] = None
            continue
        intervals = [(hour_times[i + 1] - hour_times[i]).seconds // 60 for i in range(len(hour_times) - 1)]
        max_interval = max(intervals) if intervals else None
        hourly_intervals[hour] = max_interval

    return hourly_intervals


def calculate_headway_stats(df, time_col='s_time', route_col='route_id'):
    """
    根据车辆行程数据计算每条线路的总发车间隔和高峰发车间隔

    参数:
        df (pd.DataFrame): 包含 `route_col` 和 `time_col` 的原始数据
        time_col (str): 时间列名，默认 's_time'
        route_col (str): 线路ID列名，默认 'route_id'

    返回:
        pd.DataFrame: headway_df 包含每条线路的 `total_headway` 和 `peak_hourly_headway`
    """
    results = []

    for route_id, group in df.groupby(route_col):
        times = group[time_col].sort_values().reset_index(drop=True)
        total_trips = len(times)

        if total_trips < 2:
            results.append({
                'route_id': route_id,
                'total_headway': None,
                'peak_hourly_headway': None
            })
            continue

        # 计算所有相邻班次之间的间隔（分钟）
        intervals = [(times.iloc[i + 1] - times.iloc[i]).total_seconds() / 60 for i in range(len(times) - 1)]

        # 过滤掉大于120分钟的异常值
        filtered_intervals = [i for i in intervals if i <= 120]

        if not filtered_intervals:
            total_headway = None
        else:
            total_headway = sum(filtered_intervals) / len(filtered_intervals)

        # 高峰发车间隔计算
        hourly_intervals = []
        for hour in range(24):
            hour_times = times[times.dt.hour == hour]
            count = len(hour_times)
            if count >= 2:
                interval = 60 / count  # 每小时平均发车间隔（分钟/班）
                hourly_intervals.append(interval)

        if hourly_intervals:
            peak_hourly_headway = min(hourly_intervals)  # 取最密的一小时
        elif filtered_intervals:
            peak_hourly_headway = min(filtered_intervals)  # 所有间隔中最短的
        else:
            peak_hourly_headway = None

        results.append({
            'route_id': route_id,
            'total_headway': total_headway,
            'peak_hourly_headway': peak_hourly_headway
        })

    headway_df = pd.DataFrame(results)
    return headway_df


def network_independents(args):
    i, province_list, city_list = args
    province_name, city_name = province_list[i], city_list[i]

    """0. Loading files"""
    route_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
                              crs=CRS.from_epsg(4326))
    stop_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
                             crs=CRS.from_epsg(4326))
    route_shp = route_shp.to_crs("EPSG:4547")
    stop_shp = stop_shp.to_crs("EPSG:4547")

    vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')
    vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
    vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)

    valid_route_ids = vs_parking_df['route_name'].unique()
    route_shp = route_shp[route_shp['route_name'].isin(valid_route_ids)].copy()
    stop_shp = stop_shp[stop_shp['route_name'].isin(valid_route_ids)].copy()

    print(f'Processing {city_name}...')

    """9. stop num and pop coverage"""
    stop_count = stop_shp['stop_id'].nunique()
    pop_coverage, total_population = calculate_population_coverage(city_name, province_name, stop_shp)
    info_slice_stop = [stop_count, pop_coverage, total_population]

    """Return all slices as a dictionary"""
    return {
        'city': city_name,
        'province': province_name,
        'stop_num': info_slice_stop[0],
        'pop_coverage': info_slice_stop[1],
        'total_population': info_slice_stop[2]
    }


# 添加需要的导入
from shapely.geometry import box

if __name__ == '__main__':
    """0. Initialize"""
    # 初始化全局数据
    initialize_global_data()

    cities = pd.read_csv(rf'../data/18cities.csv')
    province_list = cities['province'].to_list()
    city_list = cities['city'].to_list()
    date = '2025-1-20'

    """1. multiprocessing the network-related independent variables"""
    # 准备参数列表
    tasks = [(i, province_list, city_list) for i in range(len(city_list))]
    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:  # 可根据 CPU 核心数调整
        future_to_city = {executor.submit(network_independents, task): task for task in tasks}
        for future in as_completed(future_to_city):
            result = future.result()
            if result:
                results.append(result)
    # 合并结果
    result_df = pd.DataFrame(results)
    # result_df.to_csv(r'../data/output/city_indicators_all.csv', index=False)

    # 关闭全局栅格文件
    if POPULATION_RASTER:
        POPULATION_RASTER.close()
