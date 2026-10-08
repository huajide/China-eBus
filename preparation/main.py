import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

import gradient
from road_network import city_roadnet

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
RAW_ROOT = DATA_ROOT / "raw_data"
OUTPUT_ROOT = RAW_ROOT / "generated" / "vs_parking_nodeid"
SRTM_INDEX = RAW_ROOT / "srtm" / "index.shp"
SRTM_TILES = RAW_ROOT / "srtm"

def process_city(city_info):
    i, (province_name, city_name) = city_info
    try:
        route_shp = gpd.read_file(f'{RAW_ROOT}/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp', crs=CRS.from_epsg(4326))
        city_roadnet(route_shp, save_path=f'{RAW_ROOT}/generated/roads/{city_name}')
        stop_shp = gpd.read_file(f'{RAW_ROOT}/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp', crs=CRS.from_epsg(4326))
        stop_shp_uni = stop_shp.drop_duplicates(subset=['stop_id'])
        stop_shp_uni = stop_shp_uni.reset_index(drop=True)
        stop_shp_uni = stop_shp_uni.to_crs('EPSG:4547')
        stop_shp_uni.to_file(f'{RAW_ROOT}/generated/roads/{city_name}/uni_stops.shp', index=False)
        road_shp = gpd.read_file(f'{RAW_ROOT}/generated/roads/{city_name}/edges.shp', crs=CRS.from_epsg(4326))
        road_shp = road_shp.to_crs('EPSG:4547')
        road_shp.to_file(f'{RAW_ROOT}/generated/roads/{city_name}/edges.shp', index=False)
        print(f'{city_name} in s1.1 is done!')
        return f'{city_name} processed successfully'
    except Exception as e:
        return f'Error processing {city_name}: {str(e)}'

def process_city_slope_energy(province_name, city_name):
    try:
        route_shp_path = f'{RAW_ROOT}/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp'
        srtm_index_path = SRTM_INDEX
        srtm_tiles_dir = SRTM_TILES
        if not os.path.exists(route_shp_path):
            print(f'Warning: route file not found for {city_name}: {route_shp_path}')
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
            print(f'SRTM processing failed for {city_name}')
            return None
    except Exception as e:
        print(f'Error processing {city_name}: {str(e)}')
        return None

def calculate_route_stop_counts(province_name, city_name):
    try:
        stop_shp_path = f'{RAW_ROOT}/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_stop5.shp'
        if not os.path.exists(stop_shp_path):
            print(f'Warning: stop file not found for {city_name}: {stop_shp_path}')
            return None
        stop_gdf = gpd.read_file(stop_shp_path)
        stop_counts = stop_gdf.groupby('route_name').size().reset_index(name='stop_num')
        return stop_counts
    except Exception as e:
        print(f'Error calculating stop counts for {city_name}: {str(e)}')
        return None

def check_negative_base_energy(df, city_name):
    try:
        if 'base_energy' not in df.columns:
            print(f'Warning: base_energy column not found for {city_name}')
            return
        negative_energy_rows = df[df['base_energy'] < 0]
        if not negative_energy_rows.empty:
            print(f'\nRoutes with negative base_energy in {city_name}:')
            negative_energy_unique = negative_energy_rows.drop_duplicates(subset=['route_id'])
            for idx, row in negative_energy_unique.iterrows():
                route_name = row.get('route_name', 'N/A')
                base_energy = row.get('base_energy', 'N/A')
                route_id = row.get('route_id', 'N/A')
                print(f'  - City: {city_name}, route: {route_name}, route_id: {route_id}, base_energy: {base_energy:.2f}')
            print(f'  Total: {len(negative_energy_unique)} unique routes')
        else:
            print(f'No routes with negative base_energy found in {city_name}')
    except Exception as e:
        print(f'Error checking negative route energy for {city_name}: {str(e)}')

def prepare_city_energy(province_name, city_name):
    try:
        slope_energy_df = process_city_slope_energy(province_name, city_name)
        stop_count_df = calculate_route_stop_counts(province_name, city_name)
        if slope_energy_df is not None and (not slope_energy_df.empty) or (stop_count_df is not None and (not stop_count_df.empty)):
            vs_parking_path = f'{RAW_ROOT}/input/vs_parking_nodeid/{city_name}.csv'
            if os.path.exists(vs_parking_path):
                vs_parking_df = pd.read_csv(vs_parking_path).drop(
                    columns=['grade_energy', 'stop_num', 'base_energy'], errors='ignore'
                )
                if slope_energy_df is not None and (not slope_energy_df.empty):
                    vs_parking_with_slope = vs_parking_df.merge(slope_energy_df, on='route_id', how='left')
                    if 'slope_energy_change_kwh' in vs_parking_with_slope.columns:
                        vs_parking_with_slope.rename(columns={'slope_energy_change_kwh': 'grade_energy'}, inplace=True)
                else:
                    vs_parking_with_slope = vs_parking_df.copy()
                    vs_parking_with_slope['grade_energy'] = np.nan
                if stop_count_df is not None and (not stop_count_df.empty):
                    vs_parking_with_slope = vs_parking_with_slope.merge(stop_count_df, on='route_name', how='left')
                else:
                    vs_parking_with_slope['stop_num'] = np.nan
                vs_parking_with_slope['base_energy'] = vs_parking_with_slope['grade_energy'] + vs_parking_with_slope['distance'] / 1000 * (-0.885 + 0.26 * 2 + 0.036 * 20 + 0.005 * 10 + 0.065 * 2 + 0.128 * vs_parking_with_slope['stop_num'] / vs_parking_with_slope['distance'] * 1000 + 0.007 * vs_parking_with_slope['avg_velocity'] + 0.173 * 0.6)
                output_base_dir = OUTPUT_ROOT
                if not os.path.exists(output_base_dir):
                    os.makedirs(output_base_dir)
                output_path = f'{output_base_dir}/{city_name}.csv'
                vs_parking_with_slope.to_csv(output_path, index=False)
                print(f'  Gradient energy and stop counts for {city_name} saved to: {output_path}')
                check_negative_base_energy(vs_parking_with_slope, city_name)
            else:
                raise FileNotFoundError(vs_parking_path)
        else:
            raise ValueError(f'{city_name}: no slope energy or stop counts were generated')
    except Exception as e:
        raise RuntimeError(f'{city_name}: energy input preparation failed') from e
    print('-' * 50)


def main():
    global RAW_ROOT, OUTPUT_ROOT, SRTM_INDEX, SRTM_TILES
    parser = argparse.ArgumentParser(description="Prepare route slope energy and stop counts")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--srtm-index", type=Path, default=SRTM_INDEX)
    parser.add_argument("--srtm-tiles", type=Path, default=SRTM_TILES)
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--stage", choices=("energy", "roads"), default="energy")
    args = parser.parse_args()
    RAW_ROOT = args.raw_root.resolve()
    OUTPUT_ROOT = args.output_root or RAW_ROOT / "generated" / "vs_parking_nodeid"
    SRTM_INDEX, SRTM_TILES = args.srtm_index, args.srtm_tiles
    records = json.loads((RAW_ROOT / "city_registry.json").read_text(encoding="utf-8-sig"))
    requested = set(args.cities or [])
    selected = [r for r in records if not requested or r["release_city"] in requested or r["city"] in requested]
    known = {r[k] for r in selected for k in ("city", "release_city")}
    if requested - known:
        parser.error("Unknown cities: " + ", ".join(sorted(requested - known)))
    if args.stage == "energy" and (not SRTM_INDEX.is_file() or not SRTM_TILES.is_dir()):
        parser.error("Supply existing --srtm-index and --srtm-tiles paths")
    for i, record in enumerate(selected):
        if args.stage == "roads":
            process_city((i, (record["province"], record["city"])))
        else:
            prepare_city_energy(record["province"], record["city"])


if __name__ == "__main__":
    main()


