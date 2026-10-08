import numpy as np

import os

import osmnx as ox

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

    
    
    G = ox.graph_from_bbox(expanded_bounds,
                           network_type='drive')

    if show:
        G_projected = ox.project_graph(G)
        ox.plot_graph(G_projected)
    if len(save_path):
        
        os.makedirs(save_path, exist_ok=True)
        
        nodes, edges = ox.graph_to_gdfs(G)
        
        nodes.to_file(os.path.join(save_path, 'nodes.shp'), encoding='utf-8')
        edges.to_file(os.path.join(save_path, 'edges.shp'), encoding='utf-8')







