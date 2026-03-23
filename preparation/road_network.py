import numpy as np
import osmnx as ox
import matplotlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import time
import math
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


def nodes_and_edges(gdf_roads, crs: str, meter_threshold=0.1):
    """
    Generate nodes and edges based on roads. Nodes are created along the roads
    and assigned unique identifiers.

    :param gdf_roads: A GeoDataFrame containing roads and their geometry.
    :param crs: The CRS must be a projected coordinate system.
    :param meter_threshold: The threshold for deciding whether two points are the same.
    :return: Two GeoDataFrames containing nodes and edges.
    """

    start_time = time.time()
    # Step 1: Generate nodes and assign IDs
    # nodes = gpd.GeoDataFrame(columns=['node_id', 'geometry'])
    nodes_id, nodes_site = [], []
    node_id = 0

    # Iterate through each line and extract the start and end points
    se_point_idx = [0, -1]
    for idx, row in gdf_roads.iterrows():
        for point_idx in se_point_idx:
            point = row['geometry'].coords[point_idx]
            # Check whether this point already exists in the node list to avoid duplication
            if point in nodes_site:
                continue
            nodes_id.append(node_id)
            nodes_site.append(point)
            node_id += 1

    nodes = pd.DataFrame({'node_id': nodes_id, 'site': nodes_site})

    end_time = time.time()
    print(f'Nodes generation time: {end_time - start_time} seconds')
    start_time = time.time()

    # Step 2: Assign temporary start-node and end-node IDs to each line
    gdf_roads['site'] = gdf_roads.apply(lambda x: x['geometry'].coords[0], axis=1)
    gdf_roads = pd.merge(gdf_roads, nodes, on='site')
    gdf_roads.rename(columns={'node_id': 'start_node'}, inplace=True)

    gdf_roads['site'] = gdf_roads.apply(lambda x: x['geometry'].coords[-1], axis=1)
    gdf_roads = pd.merge(gdf_roads, nodes, on='site')
    gdf_roads.rename(columns={'node_id': 'end_node'}, inplace=True)
    gdf_roads = gdf_roads.drop(columns=['site'])

    end_time = time.time()
    print(f'Node connection time: {end_time - start_time} seconds')
    start_time = time.time()

    # Step 3: Points with distance smaller than meter_threshold (default: 0.1 m)
    # are treated as the same point. To ensure road connectivity, they are unified.
    nodes['geometry'] = nodes.apply(lambda x: Point(x['site']), axis=1)
    nodes = gpd.GeoDataFrame(nodes, geometry='geometry')
    nodes.set_crs(epsg=4326, inplace=True)
    nodes = nodes.drop(columns=['site'])

    new_node_id = nodes['node_id'].to_list()
    new_geometry = nodes['geometry'].to_list()

    # Convert to a projected CRS for distance calculation to improve efficiency
    nodes_projective = nodes.to_crs(epsg=crs)
    nodes_projective_x, nodes_projective_y = [], []

    for i in range(len(nodes_projective)):
        x, y = nodes_projective.iloc[i].geometry.x, nodes_projective.iloc[i].geometry.y
        nodes_projective_x.append(x)
        nodes_projective_y.append(y)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            delta_x = abs(nodes_projective_x[j] - nodes_projective_x[i])
            delta_y = abs(nodes_projective_y[j] - nodes_projective_y[i])

            if delta_x > meter_threshold or delta_y > meter_threshold:
                continue

            dis = math.sqrt(delta_x ** 2 + delta_y ** 2)
            if dis < meter_threshold:
                new_node_id[j] = new_node_id[i]
                new_geometry[j] = new_geometry[i]
                print(i, j)

    nodes['new_node_id'] = new_node_id
    nodes['new_geometry'] = new_geometry
    nodes['new_node_id'] = nodes['new_node_id'].factorize()[0]

    gdf_roads['node_id'] = gdf_roads['start_node']
    gdf_roads = pd.merge(gdf_roads, nodes[['node_id', 'new_geometry', 'new_node_id']], on='node_id')
    gdf_roads.rename(columns={'new_geometry': 'new_start_geometry', 'new_node_id': 'new_start_id'}, inplace=True)

    gdf_roads['node_id'] = gdf_roads['end_node']
    gdf_roads = pd.merge(gdf_roads, nodes[['node_id', 'new_geometry', 'new_node_id']], on='node_id')
    gdf_roads.rename(columns={'new_geometry': 'new_end_geometry', 'new_node_id': 'new_end_id'}, inplace=True)

    gdf_roads = gdf_roads.drop(columns=['start_node', 'end_node', 'node_id'])
    gdf_roads.rename(columns={'new_start_id': 'start_node', 'new_end_id': 'end_node'}, inplace=True)

    end_time = time.time()
    print(f'Node and edge connection time: {end_time - start_time} seconds')
    start_time = time.time()

    # Step 4: Iterate through the GeoDataFrame and replace the start and end
    # points of each LineString
    node_id_list = nodes['new_node_id'].tolist()
    changed_id = set([x for x in node_id_list if node_id_list.count(x) > 1])

    for idx, row in gdf_roads.iterrows():
        if row['start_node'] in changed_id or row['end_node'] in changed_id:
            # Get the original LineString and the updated start/end points
            line = row['geometry']

            # The correct way is to directly build a new LineString:
            # start from the new start point, then keep the middle part
            # of the original LineString, and end at the new end point
            gdf_roads.loc[idx, 'geometry'] = LineString(
                [row['new_start_geometry'].coords[0]] +
                list(line.coords)[1:-1] +
                [row['new_end_geometry'].coords[0]]
            )

    gdf_roads = gdf_roads.drop(columns=['new_start_geometry', 'new_end_geometry'])

    gdf_roads_projective = gdf_roads['geometry']
    gdf_roads_projective = gdf_roads_projective.to_crs(epsg=crs)
    gdf_roads['length'] = gdf_roads_projective.geometry.length

    gdf_roads.rename(columns={'start_node': 'u', 'end_node': 'v'}, inplace=True)
    gdf_roads['unique_line_id'] = gdf_roads.index

    edges_return = gdf_roads[['unique_line_id', 'u', 'v', 'length', 'geometry']].copy()
    edges_return.rename(columns={'unique_line_id': 'edge_id'}, inplace=True)

    nodes.rename(columns={'new_node_id': 'unique_node_id'}, inplace=True)
    nodes['x'] = nodes.geometry.x
    nodes['y'] = nodes.geometry.y

    nodes_return = nodes[['unique_node_id', 'x', 'y', 'geometry']].copy()
    nodes_return = nodes_return.drop_duplicates(subset='unique_node_id', keep='first').reset_index(drop=True)
    nodes_return.rename(columns={'unique_node_id': 'node_id'}, inplace=True)

    end_time = time.time()
    print(f'Update time: {end_time - start_time} seconds')

    return nodes_return, edges_return


# Example usage
if __name__ == '__main__':
    point_path = r"XXX"
