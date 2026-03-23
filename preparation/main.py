import pandas as pd
import numpy as np
from osmnx.projection import is_projected
from road_network import city_roadnet, nodes_and_edges
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


# Define the worker function for processing a single city
def process_city(city_info):
    i, (province_name, city_name) = city_info

    try:
        route_shp = gpd.read_file(
            rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_route5.shp',
            crs=CRS.from_epsg(4326)
        )
        city_roadnet(route_shp, save_path=rf'../data/temp/{city_name}_{province_name}')

        stop_shp = gpd.read_file(
            rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_stop5.shp',
            crs=CRS.from_epsg(4326)
        )
        stop_shp_uni = stop_shp.drop_duplicates(subset=['stop_id'])
        stop_shp_uni = stop_shp_uni.reset_index(drop=True)
        stop_shp_uni = stop_shp_uni.to_crs("EPSG:4547")
        stop_shp_uni.to_file(rf'../data/temp/{city_name}_{province_name}/uni_stops.shp', index=False)

        road_shp = gpd.read_file(rf'../data/temp/{city_name}_{province_name}/edges.shp', crs=CRS.from_epsg(4326))
        road_shp = road_shp.to_crs("EPSG:4547")
        road_shp.to_file(rf'../data/temp/{city_name}_{province_name}/edges.shp', index=False)

        print(f'{city_name} in s1.1 is done!')
        return f"{city_name} processed successfully"
    except Exception as e:
        return f"Error processing {city_name}: {str(e)}"


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


def remove_duplicate_cs_nodes(city_name):
    """
    Handle duplicated node_id issues in charging stations for a single city,
    keeping the most central node.

    Parameters:
    city_name (str): City name
    """
    try:
        cs_gdf_path = rf'../data/input/cs_gdf/{city_name}.shp'

        # Check whether the file exists
        if not os.path.exists(cs_gdf_path):
            print(f"Warning: charging station file for city {city_name} does not exist")
            return False

        # Read charging station data
        cs_gdf = gpd.read_file(cs_gdf_path)
        original_count = len(cs_gdf)

        # Find duplicated node_id values
        duplicated_mask = cs_gdf.duplicated(subset=['node_id'], keep=False)
        duplicates = cs_gdf[duplicated_mask]

        if duplicates.empty:
            print(f"City {city_name} has no duplicated node_id")
            return True

        print(f"City {city_name} has {len(duplicates)} duplicated records")

        # Get all duplicated node_id values
        duplicate_node_ids = duplicates['node_id'].unique()

        # For each duplicated node_id, keep the most central record
        indices_to_remove = []
        for node_id in duplicate_node_ids:
            # Get all records with the same node_id
            same_id_records = cs_gdf[cs_gdf['node_id'] == node_id]

            if len(same_id_records) > 1:
                # Compute the geometric centroid
                centroid_x = same_id_records.geometry.x.mean()
                centroid_y = same_id_records.geometry.y.mean()

                # Compute the distance from each point to the centroid
                distances = same_id_records.geometry.apply(
                    lambda geom: ((geom.x - centroid_x) ** 2 + (geom.y - centroid_y) ** 2) ** 0.5
                )

                # Find the record closest to the centroid (keep it)
                closest_idx = distances.idxmin()

                # Mark the other records for deletion
                for idx in same_id_records.index:
                    if idx != closest_idx:
                        indices_to_remove.append(idx)

        # Remove duplicated records
        cs_gdf_cleaned = cs_gdf.drop(indices_to_remove)
        final_count = len(cs_gdf_cleaned)

        # Back up the original file
        backup_path = rf'../data/input/cs_gdf/{city_name}_backup.shp'
        if not os.path.exists(backup_path):
            cs_gdf.to_file(backup_path)
            print(f"Original file has been backed up to: {backup_path}")

        # Save the cleaned file
        cs_gdf_cleaned.to_file(cs_gdf_path)
        print(f"Duplicated node_id in city {city_name} has been processed: {original_count} -> {final_count}")

        return True

    except Exception as e:
        print(f"Error processing city {city_name}: {str(e)}")
        return False


"""0. Initialize"""
cities = pd.read_csv(rf'../data/224cities.csv')
province_list = cities['province'].to_list()
city_list = cities['city'].to_list()
city_info_list = list(enumerate(zip(province_list, city_list)))
date = '2025-1-20'


"""1. Road processing"""
'''1.1 get the osm road network and filter the stops'''
if __name__ == '__main__':
    num_processes = min(cpu_count(), len(city_info_list))

    # Create a process pool
    with Pool(processes=num_processes) as pool:
        # Process all cities
        results = pool.map(process_city, city_info_list)

    # Print the results
    for result in results:
        print(result)

'''1.2 split the road by the stops using arcpy (***please switch to the arcgispro-py3 environment***)'''
import arcpy
proxy_thold = 50  # Stops within 50 m will be considered proxy stops
for i in range(len(city_list)):
    province_name, city_name = province_list[i], city_list[i]
    point_path = rf'..\data\temp\{city_name}_{province_name}/uni_stops.shp'
    line_path = rf'..\data\temp\{city_name}_{province_name}\edges.shp'
    export_path = rf'..\data\temp\{city_name}_{province_name}\split_roads.shp'
    arcpy.management.SplitLineAtPoint(line_path, point_path, export_path, f"{proxy_thold} Meters")
    print(f'{city_name} in s1.2 is done!')

"""1.3 (***switch back to the original environment***)"""
def process_road_data(city_info):
    i, (province_name, city_name) = city_info
    try:
        output_dir = rf'../data/road/{city_name}_{province_name}'
        nodes_file = os.path.join(output_dir, 'nodes_sim.shp')
        edges_file = os.path.join(output_dir, 'edges_sim.shp')

        # Check whether the files exist and both are larger than 1 KB
        if (
            os.path.exists(nodes_file) and os.path.exists(edges_file) and
            os.path.getsize(nodes_file) > 1024 and os.path.getsize(edges_file) > 1024
        ):
            print(f'{city_name} files already exist and are larger than 1KB, skipping...')
            return f"{city_name} skipped (files exist)"

        road_split_shp = gpd.read_file(rf'../data/temp/{city_name}_{province_name}/split_roads.shp', crs=CRS.from_epsg(4326))
        road_split_shp.drop(columns=['u', 'v'], inplace=True)
        nodes_sim, edges_sim = nodes_and_edges(road_split_shp, crs='4547', meter_threshold=5)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nodes_sim.to_file(nodes_file, index=False)
        edges_sim.to_file(edges_file, index=False)
        print(f'{city_name} in s1.3 is done!')
        return f"{city_name} processed successfully"
    except Exception as e:
        return f"Error processing {city_name}: {str(e)}"


if __name__ == '__main__':
    num_processes = min(cpu_count(), len(city_info_list))

    # Create a process pool
    with Pool(processes=num_processes) as pool:
        # Process all cities
        results = pool.map(process_road_data, city_info_list)

    # Print the results
    for result in results:
        print(result)


"""2. Timetable"""
all_stat = []
for i in range(len(city_list)):
    province_name, city_name = province_list[i], city_list[i]
    logging.info(f"Processing {city_name}'s timetable ({i}/{len(city_list)})")

    """2.1 standardize service hour"""
    route_gdf = gpd.read_file(
        rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_route5.shp',
        crs=CRS.from_epsg(4326)
    )
    stat = [city_name, len(route_gdf)]
    clean_service_hour(route_gdf)

    """2.2 match chelaile data"""
    chelaile_path = rf'../data/chelaile/{city_name}_{province_name}result.csv'
    chelaile_data = read_chelaile(chelaile_path)

    route_dnmc = match_chelaile(route_gdf, chelaile_data)

    """2.3 get the raw data to match the completed timetable info"""
    raw_path = f'../data/cnbusdata/{province_name}/{city_name}/{city_name}_route.csv'
    raw_data = pd.read_csv(raw_path, encoding='gbk')
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

    # Only count routes that have both peak and off-peak interval data
    valid_peak_data = route_reserved['peak_interval'].dropna()
    valid_offpeak_data = route_reserved['offpeak_interval'].dropna()

    # Routes with both peak and off-peak data
    valid_both = route_reserved.dropna(subset=['peak_interval', 'offpeak_interval'])

    if len(valid_both) > 0:
        peak_avg = valid_both['peak_interval'].mean()
        offpeak_avg = valid_both['offpeak_interval'].mean()
    else:
        # If there are no routes with both peak and off-peak data, calculate them separately
        peak_avg = valid_peak_data.mean() if len(valid_peak_data) > 0 else None
        offpeak_avg = valid_offpeak_data.mean() if len(valid_offpeak_data) > 0 else None

    peak_avg = round(peak_avg) if not pd.isna(peak_avg) else None
    offpeak_avg = round(offpeak_avg) if not pd.isna(offpeak_avg) else None

    print(f'{city_name} peak-{peak_avg} mins; offpeak-{offpeak_avg} mins')
    assume_timetable(route_reserved, off_peak_interval=offpeak_avg, peak_interval=peak_avg)
    route_reserved.drop(columns=['peak_interval', 'offpeak_interval'], inplace=True)
    stat.extend([len(route_reserved), peak_avg, offpeak_avg])

    route_reserved.to_csv(rf'../data/timetable/{city_name}_{province_name}.csv', index=False)
    all_stat.append(stat)

all_stat = pd.DataFrame(
    all_stat,
    columns=[
        'city', 'total_routes', 'with_dynamic_timetable', 'with_all_timetable',
        'with_adjusted_timetable', 'reserved_routes', 'peak_avg_interval',
        'offpeak_avg_interval'
    ]
)


"""3. Duration: data crawling from amap api"""
key = 'XXX'

for i in range(len(city_list)):
    # for i in range(90,100):
    if i != 133:
        continue
    province_name, city_name = province_list[i], city_list[i]

    # Check whether the file already exists; if yes, skip it
    duration_csv_path = rf'../data/duration/{city_name}_{province_name}.csv'
    duration_json_path = rf'../data/duration/{city_name}_{province_name}.json'

    if os.path.exists(duration_csv_path):
        print(f"File already exists, skipping city: {city_name}")
        continue

    stop_gdf = gpd.read_file(
        rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_stop5.shp',
        crs=CRS.from_epsg(4326)
    )
    timetables = pd.read_csv(rf'../data/timetable/{city_name}_{province_name}.csv')

    timetables, pt_info_all = get_all_pt_duration(timetables, stop_gdf, city_name, date, key)

    timetables.to_csv(duration_csv_path, index=False)
    with open(duration_json_path, 'w') as json_file:
        json.dump(pt_info_all, json_file)

"""4. Final processing"""
for i in range(len(city_list)):
    if i != 160:
        continue
    province_name, city_name = province_list[i], city_list[i]

    '''4.1 process duration and generate vs_parking_df'''
    stop_gdf = gpd.read_file(
        rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_stop5.shp',
        crs=CRS.from_epsg(4326)
    )
    route_info = pd.read_csv(rf'../data/duration/{city_name}_{province_name}.csv')
    route_info = update_durations(route_info)
    route_info = route_info[route_info['timetables'] != '[]'].reset_index(drop=True)

    vs_parking_df = trajectory_generation(route_info, stop_gdf, date)

    '''4.2 generate vs_vs_parking_nodeid'''
    vs_parking_df = vehicle_scheduling(
        vs_parking_df, minInterval=5, speed=40, line_name='name2crawl',
        s_coords='orientation_coords', e_coords='destination_coords',
        s_time='s_time', e_time='e_time'
    )

    nodes_sim = gpd.read_file(rf'../data/road/{city_name}_{province_name}/nodes_sim.shp', crs=CRS.from_epsg(4547))
    nodes_sim = nodes_sim[['node_id', 'geometry']].copy()

    if isinstance(vs_parking_df['destination_coords'][0], str):
        vs_parking_df['destination_coords'] = vs_parking_df.apply(
            lambda x: ast.literal_eval(x['destination_coords']),
            axis=1
        )

    all_des = vs_parking_df[['destination_coords', 'distance']].copy()
    all_des = all_des.drop_duplicates(subset='destination_coords').reset_index(drop=True)
    all_des['geometry'] = all_des.apply(lambda x: Point(x['destination_coords']), axis=1)
    all_des = all_des.drop(columns=['distance'])
    all_des = gpd.GeoDataFrame(all_des, geometry='geometry')
    all_des.set_crs(epsg=4326, inplace=True)
    all_des.to_crs(epsg=4547, inplace=True)

    all_des = tbd.ckdnearest_point(all_des, nodes_sim)
    all_des_formatch = all_des[['node_id', 'destination_coords']].copy()
    vs_parking_df = pd.merge(vs_parking_df, all_des_formatch, on='destination_coords')
    vs_parking_df.rename(
        columns={'vehicle_no': 'v_name', 'trip_no': 'trip', 'node_id': 'destination'},
        inplace=True
    )

    vs_parking_df.to_csv(rf'../data/input/vs_parking_nodeid/{city_name}_{province_name}.csv', index=False)

    print(f"{city_name} {len(all_des)} vs {vs_parking_df['destination'].nunique()}")

    '''4.3 generate charging station'''
    all_des.rename(columns={'geometry_x': 'geometry'}, inplace=True)
    all_des = all_des[['node_id', 'destination_coords', 'geometry']].copy()
    all_des['lon'] = all_des.apply(lambda x: x['destination_coords'][0], axis=1)
    all_des['lat'] = all_des.apply(lambda x: x['destination_coords'][1], axis=1)
    all_des = all_des.drop(columns=['destination_coords'])

    all_des['coords'] = all_des.apply(lambda row: (row.geometry.x, row.geometry.y), axis=1)
    raw_cs = thin_points(all_des, min_distance=500)
    raw_cs = raw_cs.drop(columns=['coords'])
    raw_cs.to_file(f'../data/input/cs_gdf/{city_name}_{province_name}.shp', driver='ESRI Shapefile', encoding='utf-8')


"""5. Supplement the vehicle (route) type"""
for i in range(len(city_list)):
    # if i != 15:
    #     continue
    province_name, city_name = province_list[i], city_list[i]

    route_gdf = gpd.read_file(
        rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_route5.shp',
        crs=CRS.from_epsg(4326)
    )
    vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}_{province_name}.csv')
    route_gdf = route_type(route_gdf)

    # Match vehicle_type from route_gdf to vs_parking_df by route_id
    vs_parking_df = vs_parking_df.merge(
        route_gdf[['route_id', 'vehicle_type']].drop_duplicates(),
        on='route_id',
        how='left'
    )

    # Fill missing values (default: medium)
    vs_parking_df['vehicle_type'] = vs_parking_df['vehicle_type'].fillna('medium')
    vs_parking_df.to_csv(rf'../data/input/vs_parking_nodeid/{city_name}_{province_name}.csv', index=False)


"""6. Handle duplicated node_id issues in charging stations"""
print("Start processing duplicated node_id issues in charging stations...")
for i in range(len(city_list)):
    # if i != 15:  # Uncomment this if you only want to process a specific city
    #     continue
    province_name, city_name = province_list[i], city_list[i]
    remove_duplicate_cs_nodes(city_name)

"""7. Add slope information and stop counts to vs_parking_nodeid based on route5.shp"""
import importlib.util
import sys
import os
import gradient


def process_city_slope_energy(province_name, city_name):
    """
    Process slope-related energy consumption for all routes in a single city.

    Parameters:
    province_name (str): Province name
    city_name (str): City name

    Returns:
    pandas.DataFrame: A DataFrame containing route_id and slope_energy_change_kwh
    """
    try:
        # Configure paths
        route_shp_path = rf'../data/cnbusdata{province_name}/{city_name}/{city_name}_route5.shp'
        srtm_index_path = r"../../SRTM_v41_China_Tiles/index.shp"
        srtm_tiles_dir = r"../../SRTM_v41_China_Tiles"

        # Check whether the route file exists
        if not os.path.exists(route_shp_path):
            print(f"Warning: route file for city {city_name} does not exist: {route_shp_path}")
            return None

        # Process the data
        result = gradient.process_srtm_data_with_route(route_shp_path, srtm_index_path, srtm_tiles_dir)

        if result[0] is not None:
            mosaic, out_meta, route_gdf = result

            # Create a list to store energy results for all routes
            energy_results = []

            # Iterate over all routes
            for idx, route_row in route_gdf.iterrows():
                route_geometry = route_row['geometry']
                route_id = route_row.get('route_id', f'Unknown_{idx}')

                # Compute the elevation profile
                distances, elevations = gradient.get_elevation_profile(route_geometry, mosaic, out_meta)

                # Apply slope smoothing
                smoothed_distances, smoothed_elevations = gradient.smooth_elevation_by_slope(distances, elevations)

                # Compute slope-related energy consumption
                energy_df = gradient.calculate_slope_energy_consumption(
                    smoothed_distances, smoothed_elevations, route_id
                )
                energy_results.append(energy_df)

            # Merge energy results of all routes
            if energy_results:
                final_energy_df = pd.concat(energy_results, ignore_index=True)
                return final_energy_df
        else:
            print(f"Failed to process SRTM data for city {city_name}")
            return None
    except Exception as e:
        print(f"Error processing city {city_name}: {str(e)}")
        return None


def calculate_route_stop_counts(province_name, city_name):
    """
    Calculate the number of stops for each route_id.

    Parameters:
    province_name (str): Province name
    city_name (str): City name

    Returns:
    pandas.DataFrame: A DataFrame containing route_name and stop_num
    """
    try:
        # Read the stop5.shp file
        stop_shp_path = rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_stop5.shp'

        if not os.path.exists(stop_shp_path):
            print(f"Warning: stop file for city {city_name} does not exist: {stop_shp_path}")
            return None

        stop_gdf = gpd.read_file(stop_shp_path)

        # Count the number of stops for each route_name
        stop_counts = stop_gdf.groupby('route_name').size().reset_index(name='stop_num')

        return stop_counts
    except Exception as e:
        print(f"Error calculating stop counts for city {city_name}: {str(e)}")
        return None


def check_negative_base_energy(df, city_name):
    """
    Check whether there are routes with base_energy < 0 in the DataFrame
    and print related information.

    Parameters:
    df (pandas.DataFrame): DataFrame containing fields such as base_energy
    city_name (str): City name
    """
    try:
        # Check whether the base_energy column exists
        if 'base_energy' not in df.columns:
            print(f"Warning: there is no base_energy column in the data for city {city_name}")
            return

        # Filter rows where base_energy is less than 0
        negative_energy_rows = df[df['base_energy'] < 0]

        if not negative_energy_rows.empty:
            print(f"\nRoutes with base_energy < 0 found in city {city_name}:")
            # Deduplicate by route_id and keep the first record
            negative_energy_unique = negative_energy_rows.drop_duplicates(subset=['route_id'])

            for idx, row in negative_energy_unique.iterrows():
                route_name = row.get('route_name', 'N/A')
                base_energy = row.get('base_energy', 'N/A')
                route_id = row.get('route_id', 'N/A')
                print(
                    f"  - City: {city_name}, Route name: {route_name}, "
                    f"route_id: {route_id}, base_energy: {base_energy:.2f}"
                )
            print(f"  Total: {len(negative_energy_unique)} unique routes")
        else:
            print(f"No routes with base_energy < 0 found in city {city_name}")
    except Exception as e:
        print(f"Error checking negative-energy routes for city {city_name}: {str(e)}")


# Main processing loop
for i in range(len(city_list)):
    province_name, city_name = province_list[i], city_list[i]
    print(f"Processing city: {city_name} ({i + 1}/{len(city_list)})")

    try:
        # Compute slope-related energy consumption for this city
        slope_energy_df = process_city_slope_energy(province_name, city_name)

        # Compute the stop counts for this city
        stop_count_df = calculate_route_stop_counts(province_name, city_name)

        if (
            (slope_energy_df is not None and not slope_energy_df.empty) or
            (stop_count_df is not None and not stop_count_df.empty)
        ):
            # Read the original vs_parking_nodeid file
            vs_parking_path = rf'../data/input/vs_parking_nodeid/{city_name}_{province_name}.csv'
            if os.path.exists(vs_parking_path):
                vs_parking_df = pd.read_csv(vs_parking_path)

                # Merge slope energy information
                if slope_energy_df is not None and not slope_energy_df.empty:
                    vs_parking_with_slope = vs_parking_df.merge(
                        slope_energy_df,
                        on='route_id',
                        how='left'
                    )

                    # Rename the field to grade_energy
                    if 'slope_energy_change_kwh' in vs_parking_with_slope.columns:
                        vs_parking_with_slope.rename(
                            columns={'slope_energy_change_kwh': 'grade_energy'},
                            inplace=True
                        )
                else:
                    vs_parking_with_slope = vs_parking_df.copy()
                    vs_parking_with_slope['grade_energy'] = np.nan

                # Merge stop count information
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

                # Create the output directory
                output_base_dir = "../data/input/vs_parking_nodeid_new"
                if not os.path.exists(output_base_dir):
                    os.makedirs(output_base_dir)

                # Save the new vs_parking_nodeid file
                output_path = rf'{output_base_dir}/{city_name}_{province_name}.csv'
                vs_parking_with_slope.to_csv(output_path, index=False)
                print(f"  Slope energy and stop count data for city {city_name} have been saved to: {output_path}")

                # Check whether there are routes with base_energy < 0
                check_negative_base_energy(vs_parking_with_slope, city_name)
            else:
                print(f"  Warning: vs_parking_nodeid file for city {city_name} does not exist: {vs_parking_path}")
        else:
            print(f"  No valid slope energy or stop count data were generated for city {city_name}")
    except Exception as e:
        print(f"  Error processing city {city_name}: {str(e)}")

    print("-" * 50)

print("Slope energy calculation, stop count calculation, and data merging for all cities are completed!")