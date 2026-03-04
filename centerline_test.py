import geopandas as gpd
from shapely.ops import unary_union
from shapely import Polygon, LineString
from shapely.ops import polygonize
import numpy as np
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

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from centerline.geometry import Centerline
from shapely.ops import voronoi_diagram



"""0. Initialize"""
cities = pd.read_csv(rf'../data/18cities.csv')
province_list = cities['province'].to_list()
city_list = cities['city'].to_list()
date = '2025-1-20'

"""X1. generate the centerline of bus network using arcpy"""

for i in range(len(city_list)):
    if i != 17:
        continue

    # 读取线数据
    province_name, city_name = province_list[i], city_list[i]
    route_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
                              crs=CRS.from_epsg(4326))
    stop_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
                              crs=CRS.from_epsg(4326))
    route_shp = route_shp.to_crs("EPSG:4547")
    stop_shp = stop_shp.to_crs("EPSG:4547")
    vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')  # 车辆行程
    vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
    vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)

    valid_route_ids = vs_parking_df['route_name'].unique()
    route_shp = route_shp[route_shp['route_name'].isin(valid_route_ids)].copy()
    stop_shp = stop_shp[stop_shp['route_name'].isin(valid_route_ids)].copy()

    # 1. 生成 45 m 缓冲
    buf_gdf = gpd.GeoDataFrame(
        geometry=route_shp.buffer(
            distance=45,  # 45 m
            cap_style='round',
            join_style='round',  # 1=round, 2=mitre, 3=bevel
        ),
        crs=route_shp.crs)

    # 2. 融合所有缓冲面
    buf_union = unary_union(buf_gdf.geometry)  # shapely Polygon / MultiPolygon

    # 3. 提取中心线
    #    Centerline 只接受单个 Polygon，所以 MultiPolygon 需要拆分
    if buf_union.geom_type == 'Polygon':
        polys = [buf_union]
    else:  # MultiPolygon
        polys = list(buf_union.geoms)

    centerlines = []
    for poly in polys:
        cl = Centerline(poly, interpolation_distance=20)
        # 检查 geometry 的类型
        if cl.geometry.geom_type == 'LineString':
            centerlines.append(cl.geometry)  # 单个 LineString 直接添加
        elif cl.geometry.geom_type == 'MultiLineString':
            centerlines.extend(cl.geometry.geoms)  # MultiLineString 拆分成多个 LineString

    center_gdf = gpd.GeoDataFrame(geometry=centerlines, crs=route_shp.crs)

    # 可视化结果（可选）
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制原始线路
    route_shp.plot(ax=ax, color='blue', linewidth=1, label='原始线路')

    # 绘制缓冲区
    buf_union_gdf = gpd.GeoDataFrame(geometry=[buf_union], crs=route_shp.crs)
    buf_union_gdf.plot(ax=ax, color='lightblue', alpha=0.5, label='45m缓冲区')

    # 绘制中心线
    center_gdf.plot(ax=ax, color='red', linewidth=2, label='中心线')

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend()
    plt.title('线路缓冲区及中心线')
    plt.show(block=True)
