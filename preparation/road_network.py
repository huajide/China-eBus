import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

import sys
import os
from shapely.geometry import Point, LineString
from shapely.ops import split, nearest_points
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'hkbus', 'preprocessing')))
from pyproj import CRS

import geopandas as gpd
import osmnx as ox
import matplotlib
matplotlib.use('TkAgg')

def city_roadnet(gdf,show=False,save_path=''):
    gdf = gdf.to_crs(epsg=4326) if gdf.crs != 4326 else gdf
    minx, miny, maxx, maxy = gdf.total_bounds
    lat_center = (miny + maxy) / 2
    km_per_degree = 111  # 纬度每度约111公里
    delta_lat = 10 / km_per_degree  # 纬度增量
    delta_lon = 10 / (km_per_degree * np.cos(np.radians(lat_center)))  # 经度增量

    expanded_bounds = (
        minx - delta_lon,  # 西经
        miny - delta_lat,  # 南纬
        maxx + delta_lon,  # 东经
        maxy + delta_lat  # 北纬
    )

    G = ox.graph_from_bbox(expanded_bounds[3], expanded_bounds[1], expanded_bounds[2], expanded_bounds[0],
                           network_type='drive')
    if show:
        G_projected = ox.project_graph(G)
        ox.plot_graph(G_projected)
    if len(save_path):
        ox.save_graph_shapefile(G, save_path)




# 使用示例
if __name__ == '__main__':
    # province_name = '四川省'
    # city_name = '绵阳市'
    # route_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
    #                           crs=CRS.from_epsg(4326))
    # stop_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
    #                           crs=CRS.from_epsg(4326))
    # stop_shp_uni = stop_shp.drop_duplicates(subset=['stop_id'])
    # stop_shp_uni = stop_shp_uni.to_crs("EPSG:4547")
    # # city_roadnet(route_shp,save_path=rf'../data/temp/{city_name}')
    # road_shp = gpd.read_file(rf'../data/temp/{city_name}/edges.shp', crs=CRS.from_epsg(4326))
    # road_shp= road_shp.to_crs("EPSG:4547")
    #
    # split_result = split_lines_with_points(road_shp, stop_shp_uni, threshold=50)

    '''准备4：测试打断线功能'''
    point_path = (r"E:\Manufacture\Python\cnbus\data\temp\stops_test.shp")
    line_path = r'E:\Manufacture\Python\cnbus\data\temp\绵阳市\edges.shp'
    export_path = r'E:\Manufacture\Python\cnbus\data\temp\split_arc_test.shp'
    arcpy.management.SplitLineAtPoint(line_path, point_path, export_path, "50 Meters")







