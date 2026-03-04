import pandas as pd
import numpy as np
from osmnx.projection import is_projected
from road_network import city_roadnet
import geopandas as gpd
from pyproj import CRS

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'hkbus', 'preprocessing')))

from timetable import clean_service_hour, read_chelaile, match_chelaile, static_timetable, fill_twin_timetable
from timetable import clean_no_timetables, assume_timetable, extract_avg_intervals
from duration import get_all_pt_duration, update_durations, trajectory_generation
from bus_scheduling import vehicle_scheduling
from route_type import route_type
from shapely.geometry import Point, LineString
import transbigdata as tbd
from scipy.spatial import KDTree
import ast
import logging
import re
import matplotlib
matplotlib.use('TkAgg')


def thin_points(df, min_distance=100):
    coords = df['coords'].apply(lambda x: (x[0], x[1])).tolist()
    tree = KDTree(coords)
    kept = set()
    for idx, row in df.iterrows():
        i = df.index.get_loc(idx)
        p = coords[i]
        indices = tree.query_ball_point(p, r=min_distance)
        neighbors = [j for j in indices if j != i]
        neighbor_labels = [df.index[j] for j in neighbors]
        if all(neighbor_label not in kept for neighbor_label in neighbor_labels):
            kept.add(idx)
    df_filtered = df.loc[list(kept)]
    return df_filtered


"""0. Initialize"""
cities = pd.read_csv(rf'../data/224cities.csv')
province_list = cities['province'].to_list()
city_list = cities['city'].to_list()
date = '2025-1-20'


"""1. Road processing"""

'''1.1 get the osm road network and filter the stops'''
# for i in range(len(city_list)):
#     province_name, city_name = province_list[i], city_list[i]
#     route_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
#                               crs=CRS.from_epsg(4326))
#     city_roadnet(route_shp,save_path=rf'../data/temp/{city_name}')
#
#     stop_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
#                               crs=CRS.from_epsg(4326))
#     stop_shp_uni = stop_shp.drop_duplicates(subset=['stop_id'])
#     stop_shp_uni = stop_shp_uni.reset_index(drop=True)
#     stop_shp_uni = stop_shp_uni.to_crs("EPSG:4547")
#     stop_shp_uni.to_file(rf'../data/temp/{city_name}/uni_stops.shp', index=False)
#
#     road_shp = gpd.read_file(rf'../data/temp/{city_name}/edges.shp', crs=CRS.from_epsg(4326))
#     road_shp= road_shp.to_crs("EPSG:4547")
#     road_shp.to_file(rf'../data/temp/{city_name}/edges.shp', index=False)
#     print(f'{city_name} in s1.1 is done!')

'''1.2 split the road by the stops using arcpy (***plz turn to the arcgispro-py3 environment***)'''
# import arcpy
# proxy_thold = 50 # distance less than 50m will be considered as proxy stop
# for i in range(len(city_list)):
#     province_name, city_name = province_list[i], city_list[i]
#     point_path = rf'E:\Manufacture\Python\cnbus\data\temp\{city_name}/uni_stops.shp'
#     line_path = rf'E:\Manufacture\Python\cnbus\data\temp\{city_name}\edges.shp'
#     export_path = rf'E:\Manufacture\Python\cnbus\data\temp\{city_name}\split_roads.shp'
#     arcpy.management.SplitLineAtPoint(line_path, point_path, export_path, f"{proxy_thold} Meters")
#     print(f'{city_name} in s1.2 is done!')

'''1.3 (***turn back to original environment***)'''
# for i in range(15,len(city_list)):
#     province_name, city_name = province_list[i], city_list[i]
#
#     road_split_shp = gpd.read_file(rf'../data/temp/{city_name}/split_roads.shp', crs=CRS.from_epsg(4326))
#     road_split_shp.drop(columns=['u','v'],inplace=True)
#     nodes_sim, edges_sim = road_preprocessing.nodes_and_edges(road_split_shp, crs='4547', meter_threshold=5)
#     output_dir = rf'../data/road/{city_name}'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     nodes_sim.to_file(output_dir+'/nodes_sim.shp', index=False)
#     edges_sim.to_file(output_dir+'/edges_sim.shp', index=False)
#     print(f'{city_name} in s1.3 is done!')

    
"""2. Timetable"""
all_stat = []
for i in range(len(city_list)):
    if i != 109 and i != 160:
        continue

    province_name, city_name = province_list[i], city_list[i]
    city_only = city_name.replace('市', '')
    logging.info(f"Processing {city_name}'s timetable ({i}/{len(city_list)})")

    """2.1 standardize service hour"""
    route_gdf = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
                              crs=CRS.from_epsg(4326))
    stat = [city_name, len(route_gdf)]
    clean_service_hour(route_gdf)

    """2.2 match chelaile data"""
    chelaile_path = rf'../data/chelaile/{city_only}result.csv'
    chelaile_data = read_chelaile(chelaile_path)

    # all_empty = chelaile_data['timetable'].apply(lambda x: x == [] or (isinstance(x, list) and len(x) == 0)).all()
    # if all_empty:
    #     print(f"城市 {city_name} 的 {city_only}result.csv 文件中 timetable 列都为空，跳过处理")
    # continue  # 跳过当前城市

    route_dnmc = match_chelaile(route_gdf,chelaile_data)

    """2.3 get the raw data to match the completed timetable info"""
    raw_path = f'../data/cnbusdata2024/{province_name}/{city_name}/{city_name}_线路.csv'
    raw_data = pd.read_csv(raw_path, encoding='gbk')
    raw_data.rename(columns={'公交id': 'route_id','运营时刻': 'timetable', '路过的公交站': 'route_stop'}, inplace=True)
    raw_data = raw_data[['route_id', 'timetable', 'route_stop']]

    route_dnmc = route_dnmc.drop(columns=['timetable', 'route_stop'])
    route_dnmc = pd.merge(route_dnmc, raw_data, on='route_id', how='left')
    # print(f"Number of routes with dynamic timetables: {route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0]}")
    stat.append(route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0])

    """2.4 generate static timetables (should be done after matching chelaile data)"""
    static_timetable(route_dnmc)
    # print(f"Number of routes with timetables: {route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0]}")
    stat.append(route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0])
    fill_twin_timetable(route_dnmc)
    # print(f"Number of routes with timetables (adjusted): {route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0]}")
    stat.append(route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0])

    """2.5 delete 5 types of route"""
    route_reserved = clean_no_timetables(route_dnmc)

    """2.6 assume the timetables for those info-lacked routes"""
    route_reserved[['peak_interval', 'offpeak_interval']] = route_reserved['timetables'].apply(
        lambda x: pd.Series(extract_avg_intervals(x))
    )

    # 只统计同时有高峰和平峰间隔数据的线路
    valid_peak_data = route_reserved['peak_interval'].dropna()
    valid_offpeak_data = route_reserved['offpeak_interval'].dropna()

    # 同时有高峰和平峰数据的线路
    valid_both = route_reserved.dropna(subset=['peak_interval', 'offpeak_interval'])

    if len(valid_both) > 0:
        peak_avg = valid_both['peak_interval'].mean()
        offpeak_avg = valid_both['offpeak_interval'].mean()
    else:
        # 如果没有同时有高峰和平峰数据的线路，则分别计算
        peak_avg = valid_peak_data.mean() if len(valid_peak_data) > 0 else None
        offpeak_avg = valid_offpeak_data.mean() if len(valid_offpeak_data) > 0 else None

    peak_avg = round(peak_avg) if not pd.isna(peak_avg) else None
    offpeak_avg = round(offpeak_avg) if not pd.isna(offpeak_avg) else None

    print(f'{city_name} peak-{peak_avg} mins; offpeak-{offpeak_avg} mins')
    assume_timetable(route_reserved,off_peak_interval=offpeak_avg, peak_interval=peak_avg)
    route_reserved.drop(columns=['peak_interval', 'offpeak_interval'], inplace=True)
    stat.extend([len(route_reserved),peak_avg,offpeak_avg])

    route_reserved.to_csv(rf'../data/timetable/{city_name}.csv', index=False)
    all_stat.append(stat)

all_stat = pd.DataFrame(all_stat, columns=['city', 'total_routes', 'with_dynamic_timetable', 'with_all_timetable',
                                           'with_adjusted_timetable', 'reserved_routes','peak_avg_interval',
                                           'offpeak_avg_interval'])


"""3. Duration: data crawling from amap api"""
# key = 'f8199ab65ebb32d41107798f2b3c491b'

# for i in range(len(city_list)):
#     if i > 1:
#         break
#     province_name, city_name = province_list[i], city_list[i]
#
#     stop_gdf = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
#                               crs=CRS.from_epsg(4326))
#     timetables = pd.read_csv(rf'../data/timetable/{city_name}.csv')
#
#     timetables, pt_info_all = get_all_pt_duration(timetables, stop_gdf, city_name, date, key)
#
#     timetables.to_csv(rf'../data/duration/{city_name}.csv', index=False)
#     with open(rf'../data/duration/{city_name}.json', 'w') as json_file:
#         json.dump(pt_info_all, json_file)

"""4. Final processing"""
# for i in range(len(city_list)):
#     # if i != 15:
#     #     continue
#     province_name, city_name = province_list[i], city_list[i]
#
#     '''4.1 processing the duration and generate vs_parking_df'''
#     stop_gdf = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
#                               crs=CRS.from_epsg(4326))
#     route_info = pd.read_csv(rf'../data/duration/{city_name}.csv')
#     route_info = update_durations(route_info)
#     route_info = route_info[route_info['timetables']!='[]'].reset_index(drop=True)
#
#
#     vs_parking_df = trajectory_generation(route_info, stop_gdf, date)
#
#     '''4.2 generate vs_vs_parking_nodeid'''
#     vs_parking_df = vehicle_scheduling(vs_parking_df, minInterval=5, speed=40, line_name='name2crawl',
#                                        s_coords='orientation_coords', e_coords='destination_coords',
#                                        s_time='s_time', e_time='e_time')
#
#     nodes_sim = gpd.read_file(rf'../data/road/{city_name}/nodes_sim.shp',crs=CRS.from_epsg(4547))
#     nodes_sim = nodes_sim[['node_id', 'geometry']].copy()
#
#     if isinstance(vs_parking_df['destination_coords'][0], str):
#         vs_parking_df['destination_coords'] = vs_parking_df.apply(lambda x: ast.literal_eval(x['destination_coords']),
#                                                                   axis=1)
#
#     all_des = vs_parking_df[['destination_coords', 'distance']].copy()
#     all_des = all_des.drop_duplicates(subset='destination_coords').reset_index(drop=True)
#     all_des['geometry'] = all_des.apply(lambda x: Point(x['destination_coords']), axis=1)
#     all_des = all_des.drop(columns=['distance'])
#     all_des = gpd.GeoDataFrame(all_des, geometry='geometry')
#     all_des.set_crs(epsg=4326, inplace=True)
#     all_des.to_crs(epsg=4547, inplace=True)
#
#     all_des = tbd.ckdnearest_point(all_des, nodes_sim)
#     all_des_formatch = all_des[['node_id', 'destination_coords']].copy()
#     vs_parking_df = pd.merge(vs_parking_df, all_des_formatch, on='destination_coords')
#     vs_parking_df.rename(columns={'vehicle_no': 'v_name', 'trip_no': 'trip',
#                                   'node_id': 'destination'}, inplace=True)
#
#     vs_parking_df.to_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv', index=False)
#
#     print(f'{city_name} {len(all_des)} vs {vs_parking_df['destination'].nunique()}')

    # '''4.3 generate charging station'''
    # all_des.rename(columns={'geometry_x': 'geometry'}, inplace=True)
    # all_des = all_des[['node_id', 'destination_coords', 'geometry']].copy()
    # all_des['lon'] = all_des.apply(lambda x: x['destination_coords'][0], axis=1)
    # all_des['lat'] = all_des.apply(lambda x: x['destination_coords'][1], axis=1)
    # all_des = all_des.drop(columns=['destination_coords'])
    #
    # all_des['coords'] = all_des.apply(lambda row: (row.geometry.x, row.geometry.y), axis=1)
    # raw_cs=thin_points(all_des, min_distance=500)
    # raw_cs = raw_cs.drop(columns=['coords'])
    # raw_cs.to_file(f'../data/input/cs_gdf/{city_name}.shp', driver='ESRI Shapefile', encoding='utf-8')


"""5. Supplement the vehicle (route) type"""
# for i in range(len(city_list)):
#     # if i != 15:
#     #     continue
#     province_name, city_name = province_list[i], city_list[i]
#
#     route_gdf = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
#                               crs=CRS.from_epsg(4326))
#     vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')
#     route_gdf = route_type(route_gdf)
#
#     # 将 route_gdf 中的 vehicle_type 按 route_id 匹配到 vs_parking_df
#     vs_parking_df = vs_parking_df.merge(
#         route_gdf[['route_id', 'vehicle_type']].drop_duplicates(),
#         on='route_id',
#         how='left'
#     )
#
#     # 填充缺失值（默认medium）
#     vs_parking_df['vehicle_type'] = vs_parking_df['vehicle_type'].fillna('medium')
#     vs_parking_df.to_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv', index=False)