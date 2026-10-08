import geopandas as gpd
import rasterio
from rasterio.merge import merge
import matplotlib.pyplot as plt
from shapely.geometry import box
import os
import numpy as np
from pyproj import Transformer


def calculate_slope_energy_consumption(distances, elevations, route_id):
    import pandas as pd

    distances = np.array(distances)
    elevations = np.array(elevations)

    total_energy_change = 0.0

    
    for i in range(len(distances) - 1):
        delta_distance = distances[i + 1] - distances[i]  
        delta_elevation = elevations[i + 1] - elevations[i]  

        if delta_distance > 0:
            
            slope_percent = (delta_elevation / delta_distance) * 100

            
            energy_change_per_km = slope_percent * 0.38

            
            distance_km = delta_distance / 1000  
            segment_energy_change = energy_change_per_km * distance_km

            total_energy_change += segment_energy_change

    
    result_df = pd.DataFrame({
        'route_id': [route_id],
        'slope_energy_change_kwh': [total_energy_change]
    })

    return result_df


def process_srtm_data_with_route(route_shp_path, srtm_index_path, srtm_tiles_dir):

    
    route_gdf = gpd.read_file(route_shp_path)

    
    if route_gdf.crs != "EPSG:4326":
        route_gdf = route_gdf.to_crs("EPSG:4326")

    
    bounds = route_gdf.total_bounds  
    route_bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])

    
    srtm_index = gpd.read_file(srtm_index_path)

    
    if srtm_index.crs != "EPSG:4326":
        srtm_index = srtm_index.to_crs("EPSG:4326")

    
    intersecting_tiles = srtm_index[srtm_index.intersects(route_bbox)]

    if intersecting_tiles.empty:
        print("No SRTM tiles intersect the route")
        return None, None

    print(f"Found {len(intersecting_tiles)} intersecting SRTM tiles:")
    tif_paths = []

    for idx, tile in intersecting_tiles.iterrows():
        location = tile['location']
        tif_path = os.path.join(srtm_tiles_dir, location)
        if os.path.exists(tif_path):
            tif_paths.append(tif_path)
            print(f"  - {location}")
        else:
            print(f"  - {location} (file not found)")

    if not tif_paths:
        print("No valid SRTM tile files found")
        return None, None

    
    src_files_to_merge = []
    for fp in tif_paths:
        src = rasterio.open(fp)
        src_files_to_merge.append(src)

    
    mosaic, out_trans = merge(src_files_to_merge)

    
    out_meta = src_files_to_merge[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans
    })

    
    for src in src_files_to_merge:
        src.close()

    return mosaic, out_meta, route_gdf


def visualize_srtm_with_route(mosaic, out_meta, route_gdf):
    
    fig, ax = plt.subplots(figsize=(12, 8))

    
    im = ax.imshow(mosaic[0], cmap='terrain',
                   extent=[out_meta['transform'][2],
                           out_meta['transform'][2] + out_meta['transform'][0] * out_meta['width'],
                           out_meta['transform'][5] + out_meta['transform'][4] * out_meta['height'],
                           out_meta['transform'][5]],
                   origin='upper')

    
    route_gdf.plot(ax=ax, color='red', linewidth=1, alpha=0.7)

    
    ax.set_title('SRTM Elevation Data with Bus Route Overlay')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Elevation (m)')

    plt.tight_layout()
    plt.show(block=True)


def get_elevation_profile(route_line, mosaic, out_meta, step_length=90):
    
    route_line_4547 = gpd.GeoSeries([route_line], crs="EPSG:4326").to_crs("EPSG:4547").iloc[0]

    
    total_length = route_line_4547.length

    
    sample_distances = np.arange(0, total_length, step_length)
    if sample_distances[-1] != total_length:
        sample_distances = np.append(sample_distances, total_length)

    
    points = [route_line_4547.interpolate(distance) for distance in sample_distances]

    
    transformer = Transformer.from_crs("EPSG:4547", "EPSG:4326", always_xy=True)
    points_wgs84 = [transformer.transform(point.x, point.y) for point in points]

    
    elevations = []
    for lon, lat in points_wgs84:
        
        px = int((lon - out_meta['transform'][2]) / out_meta['transform'][0])
        py = int((lat - out_meta['transform'][5]) / out_meta['transform'][4])

        
        if 0 <= px < out_meta['width'] and 0 <= py < out_meta['height']:
            elevation = mosaic[0, py, px]
            
            if elevation > 0:
                elevations.append(elevation)
            else:
                elevations.append(np.nan)
        else:
            elevations.append(np.nan)

    
    elevations = np.array(elevations)
    distances = np.array(sample_distances)

    
    valid_indices = ~np.isnan(elevations)
    if np.any(valid_indices):  
        
        elevations = np.interp(distances,
                               distances[valid_indices],
                               elevations[valid_indices])

    
    
    
    
    
    

    return distances, elevations


def plot_elevation_profile(distances, elevations, route_name=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    
    ax.plot(distances, elevations, 'b-', linewidth=2, marker='o', markersize=3)

    
    ax.set_xlabel('Distance along route (kilometers)')
    ax.set_ylabel('Elevation (meters)')
    ax.set_title(f'Elevation Profile Along Bus Route{" - " + route_name if route_name else ""}')
    ax.grid(True, alpha=0.3)

    
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f'{int(x/1000)}' for x in ax.get_xticks()])

    plt.tight_layout()
    plt.show(block=True)


def smooth_elevation_by_slope(distances, elevations, max_distance=5000):
    distances = np.array(distances)
    elevations = np.array(elevations)

    smoothed_distances = [distances[0]]
    smoothed_elevations = [elevations[0]]

    n = 0
    while n < len(distances) - 1:
        
        slopes = []
        n1 = n+1
        while n1 <= len(distances):
            dist = distances[n1] - distances[n]
            slope = (elevations[n1] - elevations[n]) / dist
            n1 += 1
            slopes.append(slope)
            if n1+1>=len(distances):
                break
            if elevations[n1+1] < elevations[n1]:
                break
        slope = max(slopes)


        
        if slope > 0.08:
            
            found = False

            
            for threshold in [0.005,0.01,0.02,0.03,0.04,0.05,0.1]:
                if threshold <= 0.02:
                    N = find_suitable_point(n, distances, elevations, dist, 3000, threshold)
                else:
                    N = find_suitable_point(n, distances, elevations, dist, max_distance, threshold)
                if N is not None:
                    
                    interpolate_points(n, N, distances, elevations,
                                       smoothed_distances, smoothed_elevations)
                    n = N  
                    found = True
                    break

            if not found:
                
                smoothed_distances.append(distances[n + 1])
                smoothed_elevations.append(elevations[n + 1])
                n += 1
        else:
            
            smoothed_distances.append(distances[n + 1])
            smoothed_elevations.append(elevations[n + 1])
            n += 1

    return np.array(smoothed_distances), np.array(smoothed_elevations)


def find_suitable_point(start_idx, distances, elevations, min_distance, max_distance, slope_threshold):
    start_distance = distances[start_idx]

    
    for i in range(start_idx + 2, len(distances)):
        dist = distances[i] - start_distance
        if dist > max_distance:
            break
        elif dist <= min_distance:
            continue

        
        delta_elev = elevations[i] - elevations[start_idx]
        delta_dist = distances[i] - distances[start_idx]

        if delta_dist > 0:
            slope = abs(delta_elev / delta_dist)
            if slope <= slope_threshold:
                return i

    return None


def interpolate_points(start_idx, end_idx, distances, elevations,
                       smoothed_distances, smoothed_elevations):
    start_distance = distances[start_idx]
    end_distance = distances[end_idx]
    start_elevation = elevations[start_idx]
    end_elevation = elevations[end_idx]

    
    smoothed_distances.append(end_distance)
    smoothed_elevations.append(end_elevation)

