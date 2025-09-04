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
from datetime import datetime,timedelta
from duration import circuity_calculation, find_farthest_point, start_end_point
from centerline.geometry import Centerline


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

    # """1. route length"""
    # total_distance = route_shp['distance'].sum()
    # average_distance = route_shp['distance'].mean()
    # total_count = len(route_shp)
    # count_over_20km = route_shp[route_shp['distance'] > 20].shape[0]
    # ratio_over_20km = count_over_20km / total_count if total_count > 0 else 0
    # count_over_10km = route_shp[route_shp['distance'] > 10].shape[0]
    # ratio_over_10km = count_over_10km / total_count if total_count > 0 else 0
    # info_slice_route = [total_distance, average_distance, count_over_20km, ratio_over_20km,
    #                     count_over_10km, ratio_over_10km]

    """2. network connectivity and route count"""
    # info_slice_connectivity = [calculate_connectivity(stop_shp, True),
    #                            calculate_connectivity(stop_shp, False), len(route_shp)]

    """3. route network (centerline)"""
    # buf_gdf = gpd.GeoDataFrame(
    #     geometry=route_shp.buffer(distance=45, cap_style='round', join_style='round'),
    #     crs=route_shp.crs)
    # buf_union = unary_union(buf_gdf.geometry)
    # polys = [buf_union] if buf_union.geom_type == 'Polygon' else list(buf_union.geoms)
    # centerlines = []
    # for poly in polys:
    #     cl = Centerline(poly, interpolation_distance=20)
    #     if cl.geometry.geom_type == 'LineString':
    #         centerlines.append(cl.geometry)
    #     elif cl.geometry.geom_type == 'MultiLineString':
    #         centerlines.extend(cl.geometry.geoms)
    # center_gdf = gpd.GeoDataFrame(geometry=centerlines, crs=route_shp.crs)
    # network_length = center_gdf.geometry.length.sum() / 1000
    # repetition = total_distance / network_length if network_length != 0 else None
    # info_slice_network = [network_length, repetition]
    #
    # """4. circuity (avg, number or percentage over x)"""
    # unique_routes = route_shp['route_name'].unique()
    # circuity_results = []
    # for route_name in unique_routes:
    #     route_stops = stop_shp[stop_shp['route_name'] == route_name].copy()  # 提取当前线路的 stops
    #     # 确保至少有两个站点才能计算路径
    #     if len(route_stops) < 2:
    #         continue
    #     # 计算非直线系数
    #     route_distance = route_shp[route_shp['route_name'] == route_name]['distance'].values[0]
    #     circuity = circuity_calculation(route_stops, route_distance)
    #     if circuity > 50:  # 环线
    #         f_point = find_farthest_point(route_stops)
    #         s_point, e_point = start_end_point(route_stops)
    #         distance_sf = s_point.distance(f_point)
    #         distance_fe = f_point.distance(e_point)
    #         circuity = route_distance * 1000 / (distance_sf + distance_fe)
    #     # 存储结果
    #     circuity_results.append({
    #         'route_id': route_name,
    #         'circuity': circuity
    #     })
    # circuity_df = pd.DataFrame(circuity_results)
    # average_circuity = circuity_df['circuity'].mean()
    # total_routes = len(circuity_df)
    # # 超过 2.0 的数量和比例
    # count_over_2 = circuity_df[circuity_df['circuity'] > 2.0].shape[0]
    # ratio_over_2 = count_over_2 / total_routes if total_routes > 0 else 0
    # # 超过 1.5 的数量和比例
    # count_over_1_5 = circuity_df[circuity_df['circuity'] > 1.5].shape[0]
    # ratio_over_1_5 = count_over_1_5 / total_routes if total_routes > 0 else 0
    # info_slice_circuity = [average_circuity, count_over_2, ratio_over_2, count_over_1_5, ratio_over_1_5]
    #
    # """5. stop spacing (avg, number or percentage over x)"""
    # route_shp['stop_count'] = route_shp['route_stop'].str.count(',')
    # route_shp['stop_spacing'] = route_shp['distance'] / (route_shp['stop_count'] - 1).clip(lower=1)
    #
    # total_routes = len(route_shp)
    # average_stop_spacing = route_shp['stop_spacing'].mean()
    # # 超过 1 km 的数量和比例
    # count_over_1km = route_shp[route_shp['stop_spacing'] > 1.0].shape[0]
    # ratio_over_1km = count_over_1km / total_routes if total_routes > 0 else 0
    # # 超过 0.5 km 的数量和比例
    # count_over_half_km = route_shp[route_shp['stop_spacing'] > 0.5].shape[0]
    # ratio_over_half_km = count_over_half_km / total_routes if total_routes > 0 else 0
    #
    # info_slice_spacing = [average_stop_spacing, count_over_1km, ratio_over_1km,
    #                       count_over_half_km, ratio_over_half_km]
    #
    # """6. trip number and variance"""
    # vs_parking_df['hour'] = vs_parking_df['s_time'].dt.hour
    # # 按小时统计发车数量
    # all_hours = pd.Series(0, index=range(24), name='count')
    # hourly_departures = vs_parking_df.groupby('hour').size()
    # all_hours.update(hourly_departures)
    # # 计算每小时发车数量的方差
    # variance = all_hours.var()
    # info_slice_trips  = [len(vs_parking_df), variance]
    #
    # """7. departure interval (avg, number or percentage over x)"""
    # headway_df = calculate_headway_stats(vs_parking_df, time_col='s_time', route_col='route_id')
    #
    # # 统计整体指标
    # valid_total = headway_df.dropna(subset=['total_headway'])
    # avg_total_headway = valid_total['total_headway'].mean() if not valid_total.empty else None
    #
    # valid_peak = headway_df.dropna(subset=['peak_hourly_headway'])
    # avg_peak_headway = valid_peak['peak_hourly_headway'].mean() if not valid_peak.empty else None
    #
    # # 数量和比例统计
    # if not valid_peak.empty:
    #     count_peak_10 = valid_peak[valid_peak['peak_hourly_headway'] <= 10].shape[0]
    #     count_peak_20 = valid_peak[valid_peak['peak_hourly_headway'] <= 20].shape[0]
    #     ratio_peak_10 = count_peak_10 / len(headway_df)
    #     ratio_peak_20 = count_peak_20 / len(headway_df)
    # else:
    #     count_peak_10 = count_peak_20 = ratio_peak_10 = ratio_peak_20 = 0
    #
    # if not valid_total.empty:
    #     count_avg_15 = valid_total[valid_total['total_headway'] <= 15].shape[0]
    #     count_avg_30 = valid_total[valid_total['total_headway'] <= 30].shape[0]
    #     ratio_avg_15 = count_avg_15 / len(headway_df)
    #     ratio_avg_30 = count_avg_30 / len(headway_df)
    # else:
    #     count_avg_15 = count_avg_30 = ratio_avg_15 = ratio_avg_30 = 0
    #
    # info_slice_headway = [avg_total_headway, avg_peak_headway, count_peak_10, ratio_peak_10,
    #                       count_peak_20, ratio_peak_20, count_avg_15, ratio_avg_15,
    #                       count_avg_30, ratio_avg_30]
    #
    # """8. velocity in peak hour"""
    # hourly_avg_velocity = vs_parking_df.groupby('hour')['avg_velocity'].mean().reset_index(name='avg_speed')
    # min_velocity_row = hourly_avg_velocity.loc[hourly_avg_velocity['avg_speed'].idxmin()]
    # info_slice_velocity = [min_velocity_row.values[1]]

    """8. operation distance and vehicles"""
    # 计算车均里程
    total_distance_by_city = vs_parking_df['distance'].sum()
    vehicle_count_by_city = vs_parking_df['v_name'].nunique()

    # 生成车均里程的DataFrame
    average_distance_per_vehicle = total_distance_by_city / vehicle_count_by_city
    info_slice_operation = [total_distance_by_city,vehicle_count_by_city,average_distance_per_vehicle]

    """Return all slices as a dictionary"""
    return {
        'city': city_name,
        'province': province_name,
        # 'route_total_distance': info_slice_route[0],
        # 'route_avg_distance': info_slice_route[1],
        # 'route_count_over_20km': info_slice_route[2],
        # 'route_ratio_over_20km': info_slice_route[3],
        # 'route_count_over_10km': info_slice_route[4],
        # 'route_ratio_over_10km': info_slice_route[5],
        # 'route_connectivity': info_slice_connectivity[0],
        # 'network_connectivity': info_slice_connectivity[1],
        # 'route_count': info_slice_connectivity[2]
        # 'network_length_km': info_slice_network[0],
        # 'repetition_rate': info_slice_network[1],
        # 'average_circuity': info_slice_circuity[0],
        # 'count_circuity_over_2': info_slice_circuity[1],
        # 'ratio_circuity_over_2': info_slice_circuity[2],
        # 'count_circuity_over_1_5':  info_slice_circuity[3],
        # 'ratio_circuity_over_1_5': info_slice_circuity[4],
        # 'average_stop_spacing': info_slice_spacing[0],
        # 'count_spacing_over_1km': info_slice_spacing[1],
        # 'ratio_spacing_over_1km': info_slice_spacing[2],
        # 'count_spacing_over_half_km': info_slice_spacing[3],
        # 'ratio_spacing_over_half_km': info_slice_spacing[4],
        # 'trip_number': info_slice_trips[0],
        # 'variance': info_slice_trips[1],
        # 'avg_total_headway': info_slice_headway[0],
        # 'avg_peak_headway': info_slice_headway[1],
        # 'count_peak_over_10min': info_slice_headway[2],
        # 'ratio_peak_over_10min': info_slice_headway[3],
        # 'count_peak_over_20min': info_slice_headway[4],
        # 'ratio_peak_over_20min': info_slice_headway[5],
        # 'count_avg_over_15min': info_slice_headway[6],
        # 'ratio_avg_over_15min': info_slice_headway[7],
        # 'count_avg_over_30min': info_slice_headway[8],
        # 'ratio_avg_over_30min': info_slice_headway[9],
        # 'min_velocity': info_slice_velocity[0]
        'operation_distance': info_slice_operation[0],
        'vehicle_count': info_slice_operation[1],
        'distance_per_fuel_vehicle': info_slice_operation[2]
    }


if __name__ == '__main__':
    """0. Initialize"""
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
