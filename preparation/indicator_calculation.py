import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
from pyproj import CRS
import ast
import math
from dateutil import parser

from pathlib import Path

GIS_ROOT = Path(os.environ.get("CNBUS_GIS_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "gis")))

import geopandas as gpd
from datetime import datetime

import rasterio
from rasterio.mask import mask
import numpy as np


ZONE_GDF = None
POPULATION_RASTER = None


def initialize_global_data():
    'Initialize global data.'
    global ZONE_GDF, POPULATION_RASTER

    if ZONE_GDF is None:
        
        ZONE_GDF = gpd.read_file(GIS_ROOT / "zones.shp")
        ZONE_GDF = ZONE_GDF.set_crs(epsg=32650)
        ZONE_GDF = ZONE_GDF.to_crs(epsg=4547)

    if POPULATION_RASTER is None:
        
        POPULATION_RASTER = rasterio.open(GIS_ROOT / "population.tif")


def calculate_population_coverage(city_name, province_name, STOP_GDF):
    'Calculate population coverage.'

    
    global ZONE_GDF, POPULATION_RASTER
    if ZONE_GDF is None or POPULATION_RASTER is None:
        initialize_global_data()

    
    zone_gdf = ZONE_GDF

    
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
        print(f"No municipal boundary data found for {city_name}")
        return None, None

    
    sub_zone['geometry'] = sub_zone['geometry'].buffer(0)
    merged_zone_polygon = sub_zone.unary_union
    sub_zone_gdf = gpd.GeoDataFrame(geometry=[merged_zone_polygon], crs=zone_gdf.crs)
    sub_zone_gdf = sub_zone_gdf.to_crs(epsg=4547)

    
    out_image, out_transform = mask(POPULATION_RASTER, sub_zone_gdf.geometry, crop=True, nodata=0)
    total_population = np.nansum(out_image)

    
    
    stop_gdf_projected = STOP_GDF.to_crs(epsg=4547)  
    stop_gdf_projected['geometry'] = stop_gdf_projected['geometry'].buffer(500)  
    stop_buffer_union = stop_gdf_projected.unary_union  

    
    stop_buffer_gdf = gpd.GeoDataFrame(geometry=[stop_buffer_union], crs=stop_gdf_projected.crs)

    
    
    intersection = gpd.overlay(stop_buffer_gdf, sub_zone_gdf, how='intersection')

    
    if len(intersection) > 0:
        try:
            out_image, out_transform = mask(POPULATION_RASTER, intersection.geometry, crop=True, nodata=0)
            covered_population = np.nansum(out_image)
        except ValueError as e:
            if "Input shapes do not overlap raster" in str(e):
                print(f"{city_name}: the buffer does not overlap the population raster")
                covered_population = 0
            else:
                raise e
    else:
        covered_population = 0

    
    coverage_ratio = covered_population / total_population if total_population > 0 else 0

    print(
        f"{city_name}: {total_population:.0f} people in downtown and {coverage_ratio:.2%} are covered by bus transit")
    return coverage_ratio, total_population


def calculate_distance(p1, p2):
    'Calculate distance.'
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calculate_connectivity(stop_df, allow_shared_stops=True):
    'Calculate connectivity.'
    if allow_shared_stops:
        link_count = stop_df.groupby('route_name').size() - 1
        total_links = link_count.sum()
    else:
        
        edges_set = set()
        for route_id, group in stop_df.groupby('route_name'):
            sorted_group = group.sort_values(by='sequence').reset_index()
            for i in range(len(sorted_group) - 1):
                stop_id1 = sorted_group.iloc[i]['stop_id']
                stop_id2 = sorted_group.iloc[i + 1]['stop_id']
                
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
    
    times = []
    for t in times_list:
        try:
            dt = datetime.strptime(t, "%H:%M")
            times.append(dt)
        except Exception:
            continue

    times.sort()
    
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
    'Calculate headway stats.'
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

        
        intervals = [(times.iloc[i + 1] - times.iloc[i]).total_seconds() / 60 for i in range(len(times) - 1)]

        
        filtered_intervals = [i for i in intervals if i <= 120]

        if not filtered_intervals:
            total_headway = None
        else:
            total_headway = sum(filtered_intervals) / len(filtered_intervals)

        
        hourly_intervals = []
        for hour in range(24):
            hour_times = times[times.dt.hour == hour]
            count = len(hour_times)
            if count >= 2:
                interval = 60 / count  
                hourly_intervals.append(interval)

        if hourly_intervals:
            peak_hourly_headway = min(hourly_intervals)  
        elif filtered_intervals:
            peak_hourly_headway = min(filtered_intervals)  
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

    route_shp = gpd.read_file(rf'{Path(__file__).resolve().parents[1]}/data/raw_data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
                              crs=CRS.from_epsg(4326))
    stop_shp = gpd.read_file(rf'{Path(__file__).resolve().parents[1]}/data/raw_data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
                             crs=CRS.from_epsg(4326))
    route_shp = route_shp.to_crs("EPSG:4547")
    stop_shp = stop_shp.to_crs("EPSG:4547")

    vs_parking_df = pd.read_csv(rf'{Path(__file__).resolve().parents[1]}/data/raw_data/input/vs_parking_nodeid/{city_name}.csv')
    vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
    vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)

    valid_route_ids = vs_parking_df['route_name'].unique()
    route_shp = route_shp[route_shp['route_name'].isin(valid_route_ids)].copy()
    stop_shp = stop_shp[stop_shp['route_name'].isin(valid_route_ids)].copy()

    print(f'Processing {city_name}...')

    stop_count = stop_shp['stop_id'].nunique()
    pop_coverage, total_population = calculate_population_coverage(city_name, province_name, stop_shp)
    info_slice_stop = [stop_count, pop_coverage, total_population]

    return {
        'city': city_name,
        'province': province_name,
        'stop_num': info_slice_stop[0],
        'pop_coverage': info_slice_stop[1],
        'total_population': info_slice_stop[2]
    }



