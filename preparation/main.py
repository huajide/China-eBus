import pandas as pd
import numpy as np
from osmnx.projection import is_projected
from road_network import city_roadnet
import geopandas as gpd
from pyproj import CRS
import json
import sys
import os


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
from multiprocessing import Pool, cpu_count
import gradient


def process_city(city_info):
    """Process road network for a single city."""
    i, (province_name, city_name) = city_info

    try:
        route_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
                                  crs=CRS.from_epsg(4326))
        city_roadnet(route_shp, save_path=rf'../data/temp/{city_name}')

        stop_shp = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp',
                                 crs=CRS.from_epsg(4326))
        stop_shp_uni = stop_shp.drop_duplicates(subset=['stop_id'])
        stop_shp_uni = stop_shp_uni.reset_index(drop=True)
        stop_shp_uni = stop_shp_uni.to_crs("EPSG:4547")
        stop_shp_uni.to_file(rf'../data/temp/{city_name}/uni_stops.shp', index=False)

        road_shp = gpd.read_file(rf'../data/temp/{city_name}/edges.shp', crs=CRS.from_epsg(4326))
        road_shp = road_shp.to_crs("EPSG:4547")
        road_shp.to_file(rf'../data/temp/{city_name}/edges.shp', index=False)

        print(f'{city_name} stage 1.1 complete!')
        return f"{city_name} processed successfully"
    except Exception as e:
        return f"Error processing {city_name}: {str(e)}"


def thin_points(df, min_distance=100):
    """Remove points that are too close to each other."""
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


def remove_duplicate_cs_nodes(city_name):
    """
    Handle duplicate node_id issues in charging stations by keeping the most central node.

    Parameters:
    city_name (str): city name
    """
    try:
        cs_gdf_path = rf'../data/input/cs_gdf/{city_name}.shp'

        if not os.path.exists(cs_gdf_path):
            print(f"Warning: charging station file for {city_name} does not exist")
            return False

        cs_gdf = gpd.read_file(cs_gdf_path)
        original_count = len(cs_gdf)

        duplicated_mask = cs_gdf.duplicated(subset=['node_id'], keep=False)
        duplicates = cs_gdf[duplicated_mask]

        if duplicates.empty:
            print(f"{city_name} has no duplicate node_id")
            return True

        print(f"{city_name} found {len(duplicates)} duplicate records")

        duplicate_node_ids = duplicates['node_id'].unique()

        indices_to_remove = []
        for node_id in duplicate_node_ids:
            same_id_records = cs_gdf[cs_gdf['node_id'] == node_id]

            if len(same_id_records) > 1:
                centroid_x = same_id_records.geometry.x.mean()
                centroid_y = same_id_records.geometry.y.mean()

                distances = same_id_records.geometry.apply(
                    lambda geom: ((geom.x - centroid_x) ** 2 + (geom.y - centroid_y) ** 2) ** 0.5
                )

                closest_idx = distances.idxmin()

                for idx in same_id_records.index:
                    if idx != closest_idx:
                        indices_to_remove.append(idx)

        cs_gdf_cleaned = cs_gdf.drop(indices_to_remove)
        final_count = len(cs_gdf_cleaned)

        backup_path = rf'../data/input/cs_gdf/{city_name}_backup.shp'
        if not os.path.exists(backup_path):
            cs_gdf.to_file(backup_path)
            print(f"Original file backed up to: {backup_path}")

        cs_gdf_cleaned.to_file(cs_gdf_path)
        print(f"{city_name} duplicate node_id processed: {original_count} -> {final_count}")

        return True

    except Exception as e:
        print(f"Error processing {city_name}: {str(e)}")
        return False


cities = pd.read_csv(rf'../data/224cities.csv')
province_list = cities['province'].to_list()
city_list = cities['city'].to_list()
city_info_list = list(enumerate(zip(province_list, city_list)))
date = '2025-1-20'


def process_city_slope_energy(province_name, city_name):
    """
    Calculate slope-related energy consumption for all routes in a city.

    Parameters:
    province_name (str): province name
    city_name (str): city name

    Returns:
    pandas.DataFrame: containing route_id and slope_energy_change_kwh
    """
    try:
        route_shp_path = rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp'
        srtm_index_path = r"../../SRTM_v41_China_Tiles/index.shp"
        srtm_tiles_dir = r"../../SRTM_v41_China_Tiles"

        if not os.path.exists(route_shp_path):
            print(f"Warning: route file for {city_name} does not exist: {route_shp_path}")
            return None

        result = gradient.process_srtm_data_with_route(route_shp_path, srtm_index_path, srtm_tiles_dir)

        if result[0] is not None:
            mosaic, out_meta, route_gdf = result

            energy_results = []

            for idx, route_row in route_gdf.iterrows():
                route_geometry = route_row['geometry']
                route_id = route_row.get('route_id', f'Unknown_{idx}')

                distances, elevations = gradient.get_elevation_profile(route_geometry, mosaic, out_meta)

                smoothed_distances, smoothed_elevations = gradient.smooth_elevation_by_slope(distances, elevations)

                energy_df = gradient.calculate_slope_energy_consumption(smoothed_distances, smoothed_elevations, route_id)
                energy_results.append(energy_df)

            if energy_results:
                final_energy_df = pd.concat(energy_results, ignore_index=True)
                return final_energy_df
        else:
            print(f"SRTM data processing failed for {city_name}")
            return None
    except Exception as e:
        print(f"Error processing {city_name}: {str(e)}")
        return None


def calculate_route_stop_counts(province_name, city_name):
    """
    Calculate the number of stops for each route_id.

    Parameters:
    province_name (str): province name
    city_name (str): city name

    Returns:
    pandas.DataFrame: containing route_name and stop_num
    """
    try:
        stop_shp_path = rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp'

        if not os.path.exists(stop_shp_path):
            print(f"Warning: stop file for {city_name} does not exist: {stop_shp_path}")
            return None

        stop_gdf = gpd.read_file(stop_shp_path)

        stop_counts = stop_gdf.groupby('route_name').size().reset_index(name='stop_num')

        return stop_counts
    except Exception as e:
        print(f"Error calculating stop counts for {city_name}: {str(e)}")
        return None


def check_negative_base_energy(df, city_name):
    """
    Check if there are routes with base_energy less than 0.

    Parameters:
    df (pandas.DataFrame): DataFrame containing base_energy field
    city_name (str): city name
    """
    try:
        if 'base_energy' not in df.columns:
            print(f"Warning: base_energy column not found in {city_name}")
            return

        negative_energy_rows = df[df['base_energy'] < 0]

        if not negative_energy_rows.empty:
            print(f"\nRoutes with negative base_energy in {city_name}:")
            negative_energy_unique = negative_energy_rows.drop_duplicates(subset=['route_id'])

            for idx, row in negative_energy_unique.iterrows():
                route_name = row.get('route_name', 'N/A')
                base_energy = row.get('base_energy', 'N/A')
                route_id = row.get('route_id', 'N/A')
                print(f"  - City: {city_name}, Route: {route_name}, route_id: {route_id}, base_energy: {base_energy:.2f}")
            print(f"  Total: {len(negative_energy_unique)} unique routes")
        else:
            print(f"No routes with negative base_energy in {city_name}")
    except Exception as e:
        print(f"Error checking negative energy for {city_name}: {str(e)}")


for i in range(len(city_list)):
    province_name, city_name = province_list[i], city_list[i]
    print(f"Processing city: {city_name} ({i+1}/{len(city_list)})")

    try:
        slope_energy_df = process_city_slope_energy(province_name, city_name)
        stop_count_df = calculate_route_stop_counts(province_name, city_name)

        if (slope_energy_df is not None and not slope_energy_df.empty) or \
           (stop_count_df is not None and not stop_count_df.empty):
            vs_parking_path = rf'../data/input/vs_parking_nodeid/{city_name}.csv'
            if os.path.exists(vs_parking_path):
                vs_parking_df = pd.read_csv(vs_parking_path)

                if slope_energy_df is not None and not slope_energy_df.empty:
                    vs_parking_with_slope = vs_parking_df.merge(
                        slope_energy_df,
                        on='route_id',
                        how='left'
                    )

                    if 'slope_energy_change_kwh' in vs_parking_with_slope.columns:
                        vs_parking_with_slope.rename(
                            columns={'slope_energy_change_kwh': 'grade_energy'},
                            inplace=True
                        )
                else:
                    vs_parking_with_slope = vs_parking_df.copy()
                    vs_parking_with_slope['grade_energy'] = np.nan

                if stop_count_df is not None and not stop_count_df.empty:
                    vs_parking_with_slope = vs_parking_with_slope.merge(
                        stop_count_df,
                        on='route_name',
                        how='left'
                    )
                else:
                    vs_parking_with_slope['stop_num'] = np.nan

                vs_parking_with_slope['base_energy'] = (
                    vs_parking_with_slope['grade_energy'] +
                    vs_parking_with_slope['distance'] / 1000 * (
                        -0.885 + 0.260 * 2 + 0.036 * 20 + 0.005 * 10 + 0.065 * 2 +
                        0.128 * vs_parking_with_slope['stop_num'] / vs_parking_with_slope['distance'] * 1000 +
                        0.007 * vs_parking_with_slope['avg_velocity'] + 0.173 * 0.6
                    )
                )

                output_base_dir = "../data/input/vs_parking_nodeid_new"
                if not os.path.exists(output_base_dir):
                    os.makedirs(output_base_dir)

                output_path = rf'{output_base_dir}/{city_name}.csv'
                vs_parking_with_slope.to_csv(output_path, index=False)
                print(f"  Data saved for {city_name}: {output_path}")

                check_negative_base_energy(vs_parking_with_slope, city_name)
            else:
                print(f"  Warning: vs_parking_nodeid file not found for {city_name}: {vs_parking_path}")
        else:
            print(f"  No valid slope energy or stop count data generated for {city_name}")
    except Exception as e:
        print(f"  Error processing {city_name}: {str(e)}")

    print("-" * 50)

print("All cities processed!")