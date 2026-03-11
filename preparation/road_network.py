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
    km_per_degree = 111
    delta_lat = 10 / km_per_degree
    delta_lon = 10 / (km_per_degree * np.cos(np.radians(lat_center)))

    expanded_bounds = (
        minx - delta_lon,
        miny - delta_lat,
        maxx + delta_lon,
        maxy + delta_lat
    )

    G = ox.graph_from_bbox(expanded_bounds[3], expanded_bounds[1], expanded_bounds[2], expanded_bounds[0],
                           network_type='drive')
    if show:
        G_projected = ox.project_graph(G)
        ox.plot_graph(G_projected)
    if len(save_path):
        ox.save_graph_shapefile(G, save_path)

# Example usage
if __name__ == '__main__':
    point_path = r"stops_test.shp"
    line_path = r'edges.shp'
    export_path = r'split_arc_test.shp'
    arcpy.management.SplitLineAtPoint(line_path, point_path, export_path, "50 Meters")
