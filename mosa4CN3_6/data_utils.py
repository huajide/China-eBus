"""
@Project ：DissertationPY
@File    ：data_utils.py
@IDE     ：PyCharm
@Author  ：Xander PENG
@Date    ：11/8/2022 00:59
"""
import logging

import networkx as nx
import numpy as np
import pandas as pd
import transbigdata as tbd
from haversine import haversine
import geopandas as gpd
from dateutil import parser
from ast import literal_eval
from pyproj import Transformer
import torch
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
from collections import defaultdict
import time


def trip_seg(raw_data: pd.DataFrame, trip_gap: int, speed: int) -> pd.DataFrame:
    """
    For processing the raw traj data, identify vehicles' trips;

    :param raw_data: raw trajectory data DataFrame
    :param trip_gap: the time gap between 2 trips, unit: second
    :param speed: the speed limit of traj data
    :return: A data frame with trips column
    """

    # Identify each vehicle's trip
    trip_traj = raw_data.groupby('traj_id').apply(lambda x: tbd.clean_traj(data=x,
                                                                           col=['traj_id', 'DateTime', 'lon', 'lat'],
                                                                           tripgap=trip_gap,
                                                                           disgap=50000,
                                                                           speedlimit=speed))

    # Delete useless columns
    trip_traj.drop(columns=['traj_id'], inplace=True)
    trip_traj.reset_index(inplace=True)
    del trip_traj['level_1']
    # Calculate accumulative distances and replace the original column
    cumsum_d = trip_traj.groupby('traj_id').apply(lambda x: x['Distance'].cumsum())
    cumsum_d = cumsum_d.to_list()
    trip_traj['Distance'] = cumsum_d  # Change the Distance list

    return trip_traj


def derive_parking_info(df: pd.DataFrame) -> dict:
    df['tripid'] = df['tripid'] + 1

    df['trip_id_down'] = df['tripid'].shift(1)
    df.fillna(0, inplace=True)
    df['trip_id_down'] = df['trip_id_down'].map(lambda x: int(x))

    df['trip_id_up'] = df['tripid'].shift(-1)
    df.fillna(0, inplace=True)
    df['trip_id_up'] = df['trip_id_up'].map(lambda x: int(x))

    v_name = df.iloc[0].traj_id
    trip_o = []
    trip_d = []
    for idx, line in df.iterrows():
        if (line.tripid - line.trip_id_down == 1) & (line.tripid == 1):

            trip_o.append(line[1:8])

        elif line.tripid == line.trip_id_down != line.trip_id_up:
            trip_d.append(line[1:8])

        elif line.tripid == line.trip_id_up != line.trip_id_down:
            trip_o.append(line[1:8])

    trips_od = list(zip(trip_o, trip_d))
    trips_info = dict(zip(list(i + 1 for i in range(len(trips_od))), trips_od))
    v_trips_info = {v_name: trips_info}

    return v_trips_info


def get_vs_parking_info(vs_trip_traj: pd.DataFrame) -> pd.DataFrame:
    """
    Get a df contains all vs and their accordance parking info that stored in a dict

    :param vs_trip_traj: A data frame with identified trips
    :return: A 2-col data frame contains v-v's parking info
    """

    vs_parking_info_df = pd.DataFrame(vs_trip_traj.
                                      groupby('traj_id').
                                      apply(lambda x: derive_parking_info(x)), columns=['trips_info'])
    vs_parking_info_df.reset_index(inplace=True)

    return vs_parking_info_df


def get_loc_nearest_node(edges_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame,
                         m_edge: int, lat: float, lon: float) -> int:
    """
    For a certain gps traj point with its matched edge id, find whether u or v is much closer

    :param edges_gdf: network edges gdf
    :param nodes_gdf: network nodes gdf
    :param m_edge: matched edge id for this certain point
    :param lat: latitude of this point
    :param lon: longitude of this point
    :return: the id of u or v
    """

    # Get u and v nodes for certain m_edge
    u = edges_gdf.loc[m_edge, 'u']
    v = edges_gdf.loc[m_edge, 'v']
    # Get v and v loc info
    u_loc = tuple(nodes_gdf.query("node_id == @u").iloc[0, 1:3])  # (lat, lon)
    v_loc = tuple(nodes_gdf.query("node_id == @v").iloc[0, 1:3])
    # Cal which node is near
    u_distance = haversine((lat, lon), u_loc)
    v_distance = haversine((lat, lon), v_loc)
    if u_distance >= v_distance:
        return u
    else:
        return v


def get_od_nodes_time(row: pd.Series, edges_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For processing vs_parking_info df, analyze certain row's parking info and return a df containing
    v_name, trip(start from 1), time, destination(parking node id) and trip_distance(meter)

    :param row: certain row in vs_parking_info df
    :param edges_gdf: network edge gdf
    :param nodes_gdf: network nodes gdf
    :return: An extended df of vs_parking_info
    """

    v_name: int = row.traj_id
    v_trips_info: dict = row.trips_info.get(v_name)
    trips_amount = len(v_trips_info.keys())  # Get how many trips this v has
    current_v_info_dict = {}
    for i in range(trips_amount):
        current_td_info = v_trips_info.get(i + 1)[1]  # Get destination list
        current_ts_info = v_trips_info.get(i + 1)[0]  # Get origin list
        # Use dict to store the info
        current_t_dict = {'v_name': v_name,
                          'trip': i + 1,
                          's_time': current_ts_info[0],
                          'e_time': current_td_info[0],
                          'destination': get_loc_nearest_node(edges_gdf, nodes_gdf,
                                                              current_td_info[1],
                                                              current_td_info[3],
                                                              current_td_info[2]),
                          'distance': current_td_info[4] - current_ts_info[4]}

        current_v_info_dict.update({i + 1: current_t_dict})
    return pd.DataFrame.from_dict(current_v_info_dict, orient='index')


def get_vs_locations_df(vs_trips_info: pd.DataFrame,
                        edges_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Get vs parking locations df with time and trip_idx(start from 1);
    And sort the df based on the parking event time

    :param vs_trips_info: A df containing vs_parking info
    :param edges_gdf: edges gdf
    :param nodes_gdf: nodes gdf
    :return: vs parking locations df
    """

    vs_locations_df = pd.concat(list(vs_trips_info.apply(lambda x:
                                                         get_od_nodes_time(x, edges_gdf, nodes_gdf),
                                                         axis=1)))
    vs_locations_df['avg_velocity'] = vs_locations_df.apply(
        lambda x: (x.distance / 1000) / ((x.e_time - x.s_time).seconds / 3600), axis=1)  # unit: km/h

    # Sort the df based on parking time
    vs_locations_df = vs_locations_df.sort_values('e_time')

    return vs_locations_df


def derive_simV_info(df: pd.DataFrame):
    # Sort
    df.sort_values('e_time', inplace=True)
    max_trip = df['trip'].max()
    trips = [i + 1 for i in range(max_trip)]
    s_time_list = df['s_time'].to_list()
    e_time_list = df['e_time'].to_list()
    destinations = df['destination'].to_list()
    distances = df['distance'].to_list()
    velocity = df['avg_velocity'].to_list()

    vehicle_types = df['vehicle_type'].unique().tolist()
    priority_order = {'large': 2, 'medium': 1, 'small': 0}
    most_important_type = max(vehicle_types, key=lambda x: priority_order[x])

    return {'trip': trips,
            's_time': s_time_list,
            'e_time': e_time_list,
            'destination': destinations,
            'distance': distances,
            'avg_velocity': velocity,
            'vehicle_type': most_important_type
            }


def set_var_num(cs_dict: dict):
    """
    Get how many vars should be set; Note: This function will manipulate the "sim_v_info"

    :param sim_n:
    :param sim_v_info:
    :param cs_dict:
    :return: Number of variables
    """

    # The num of charging stations
    x_num = len(cs_dict)

    vh_num = 3  # vehicle types
    return x_num + vh_num


def find_near_cs(location: int, n: int, cs_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame, is_projected=False) \
        -> pd.DataFrame:
    """
    Find n near charging stations for current v's parking location based on Euclidean distance;

    :param is_projected:
    :param location: the location id
    :param n: the amount of cs to be selected
    :param cs_gdf: candidate charging stations gdf
    :param nodes_gdf: nodes gdf
    :return: A df containing nearest n osmid of candidate charging stations and corresponding d2s
    """
    # Get parking destination's lon and lat
    destination_node = location
    destination_lat, destination_lon = nodes_gdf.query("node_id == @destination_node").iloc[0][['y', 'x']]
    destination_loc = (destination_lat, destination_lon)

    # Cal the distance between destination and every candidate charging station (unit: km)
    if is_projected:
        d2s_df = pd.DataFrame(
            cs_gdf.apply(
                lambda x: np.sqrt((x.lon - destination_lon) ** 2 + (x.lat - destination_lat) ** 2),
                axis=1
            ),
            columns=['d2s']
        )
    else:
        d2s_df = pd.DataFrame(cs_gdf.apply(lambda x: haversine((x.lat, x.lon), destination_loc), axis=1),
                              columns=['d2s'])

    # Filter n nearest charging stations(Euclidean distance)
    near_d2s_df: pd.DataFrame = d2s_df.nsmallest(n, 'd2s')  # Get n smallest d2s distance df
    near_cs_idx: list = near_d2s_df.index.to_list()  # Get accordance idx
    near_cs: list = cs_gdf.loc[near_cs_idx, 'node_id'].to_list()  # Get near cs osmid
    near_d2s_df['node_id'] = near_cs

    return near_d2s_df


def find_sim_cs(location: int, near_cs: pd.DataFrame, network: nx.Graph, nodes_gdf: gpd.GeoDataFrame,
                cs_gdf: gpd.GeoDataFrame, distance_limit: float, n: int, projected=False) -> pd.DataFrame:
    """
    Find n closest charging stations between vehicle destination and candidate charging stations;
    Here is a @near_cs_list containing some candidate cs id, which is derived from @find_near_cs;

    :param projected:
    :param location: the location id
    :param near_cs: A df containing charging stations osmid and d2s
    :param network: networkx Graph
    :param nodes_gdf: nodes gdf
    :param cs_gdf: charging stations gdf
    :param distance_limit: the limit between destination location and candidate station, unit:m
    :param n: the amount of sim_stations to be selected
    :return: A 2-col data frame [station_id, d2s_path_length] unit: m
    """

    destination_node = location

    d2s_dict = {}
    for cs in near_cs['node_id']:
        try:  # Find the shortest path length in the network graph
            d2s_path_length = nx.shortest_path_length(network, destination_node, cs, weight='length')
        except nx.NetworkXException:  # It is possible that there is no path from d to s
            d2s_path_length = 9999  # Appoint a large number in order to filter it later

        if d2s_path_length <= distance_limit:
            d2s_dict.update({cs: d2s_path_length})

    # Once there are not enough cs, bridge the gap (For Test)
    if len(d2s_dict) < n:
        selected_cs = list(d2s_dict.keys())
        gap = n - len(d2s_dict)
        gap_cs_df = near_cs.query("node_id not in @selected_cs").nsmallest(gap, 'd2s')
        for idx, row in gap_cs_df.iterrows():
            d2s_dict.update({int(row.node_id): row.d2s * 1000})

    # Convert the dict into a DataFrame
    d2s_df = pd.DataFrame.from_dict(d2s_dict, orient='index', columns=['distance'])

    # Since there is a distance_limit constraint, the d2s_dict could be three situations:

    # 1.the number of cs in d2s_dict >= n
    if len(d2s_df) >= n:
        sim_cs = d2s_df.nsmallest(n, 'distance')  # Filter n cs

    # # the number of cs between (0, n) -> V1
    # elif (len(d2s_df) != 0) & (len(d2s_df) < n):
    #     sim_cs = d2s_df

    # the number of cs between (0,n) -> V2: Enlarge the candidate cs set and find target cs until meet 'n'
    elif (len(d2s_df) != 0) & (len(d2s_df) < n):
        # Enlarge near cs search filed and distance_limit until find valid station
        near_cs_list = find_near_cs(location, len(near_cs) * 2, cs_gdf, nodes_gdf, is_projected=projected)
        sim_cs = find_sim_cs(location, near_cs_list, network, nodes_gdf, cs_gdf, distance_limit * 2, n, projected=projected)

    else:  # the number of cs == 0
        # print(self.v_name + 'has no valid candidate sim charging station!')
        logging.log(logging.DEBUG, ' has no valid candidate sim charging station!' +
                    ' current near_cs_list length is: ' + str(len(near_cs)))

        # Enlarge near cs search filed and distance_limit until find valid station
        near_cs_list = find_near_cs(location, len(near_cs) * 2, cs_gdf, nodes_gdf, is_projected=projected)
        sim_cs = find_sim_cs(location, near_cs_list, network, nodes_gdf, cs_gdf, distance_limit * 2, n, projected=projected)

    return sim_cs


def get_d2s_dict(parking_df: pd.DataFrame, cs_gdf: gpd.GeoDataFrame, nodes_sim: gpd.GeoDataFrame, network,
                 near_n=20, sim_n=3, distance_limit=100000.0, is_projected=False):
    """
    Find all destinations and corresponding sim_cs in advance
    :param is_projected:
    :param distance_limit:
    :param near_n:
    :param sim_n:
    :param parking_df: vs_parking_trip_df
    :param cs_gdf:
    :param nodes_sim:
    :param network:
    :return: A MultiIndex df like [destination, station, d2s_length]
    """
    destinations = parking_df['destination'].unique().tolist()
    cs_list = cs_gdf['node_id'].to_list()

    sim_cs_dict = {}
    '''Use @find_near_cs and @find_sim_cs to find all d2s'''
    for d in destinations:
        near_cs = find_near_cs(d, near_n, cs_gdf, nodes_sim, is_projected=is_projected)
        sim_cs = find_sim_cs(d, near_cs, network, nodes_sim, cs_gdf, distance_limit, sim_n, projected=is_projected)
        # cs_dict = dict(map(lambda x: [x[0], list(x[1].values())[0]], sim_cs.T.to_dict().items()))
        # cs_dict = {}
        for s, row in sim_cs.iterrows():
            sim_cs_dict.update({(d, s): row.distance})

    # sim_cs_df = pd.Series(sim_cs_dict).to_frame()
    # sim_cs_df.columns = ['distance']
    sim_cs_df = pd.DataFrame(
        [(d, s, distance) for (d, s), distance in sim_cs_dict.items()],
        columns=['destination', 'station', 'distance'])
    return sim_cs_df


def load_d2s(path):
    all_d2s = pd.read_csv(path)
    all_d2s.columns = ['destination', 'station', 'distance']
    # all_d2s.set_index('destination', inplace=True)

    return all_d2s


def derive_vs_cs(sim_v_info: pd.DataFrame, all_d2s: pd.DataFrame, cs_gdf: gpd.GeoDataFrame,
                 sim_n: int):
    """
    Get a vs_ds DataFrame for the sake of setting CV3

    :param sim_n:
    :param sim_v_info:
    :param all_d2s:
    :param cs_gdf:
    :return:
    """
    cs_dict = dict(zip(cs_gdf['node_id'], cs_gdf.index))

    destination_list,station_list = all_d2s['destination'].tolist(), all_d2s['station'].tolist()
    d2s_dict = {key: [] for key in destination_list}  # 初始化字典
    for key, value in zip(destination_list, station_list):
        d2s_dict[key].append(cs_dict[value])

    destination_info = sim_v_info['destination'].tolist()
    station_info = []
    for dl in destination_info:
        station_info.append([d2s_dict[key] for key in dl if key in d2s_dict])
    vs_cs = pd.DataFrame([str(x) for x in station_info], columns=['destination'])
    vs_cs['destination'] = vs_cs['destination'].apply(literal_eval)

    '''simple but slower running code replacing the code above'''
    # vs_cs = pd.DataFrame(sim_v_info['destination'].map(lambda x: [all_d2s.query("destination == @d")
    #                                                               ['station'].map(lambda y: cs_gdf.query("node_id == @y").
    #                                                                               index[0]).tolist()
    #                                                               for d in x]))

    vs_cs['var_loc'] = vs_cs['destination'].map(lambda x: len(x) * sim_n).cumsum().shift(1)
    vs_cs.fillna({'var_loc': 0}, inplace=True)
    vs_cs['var_loc'] = vs_cs['var_loc'].astype('int')

    vs_cs['trip_count'] = vs_cs['destination'].map(lambda x: len(x))
    return vs_cs


def read_vs_cs(path):
    vs_cs = pd.read_csv(path, index_col=0)
    vs_cs['destination'] = vs_cs['destination'].map(lambda x: literal_eval(x))
    return vs_cs


def derive_z2cs(all_d2s: pd.DataFrame, vs_parking_df: pd.DataFrame, cs_num: int, cs_gdf: gpd.GeoDataFrame,
                sim_n: int):
    cs_dict = dict(zip(cs_gdf['node_id'], cs_gdf.index))
    destination_list,station_list = all_d2s['destination'].tolist(), all_d2s['station'].tolist()
    d2s_dict = {key: [] for key in destination_list}  # 初始化字典
    for key, value in zip(destination_list, station_list):
        d2s_dict[key].append(value)

    z2ds_dict = {}
    for idx, row in vs_parking_df.iterrows():
        d = row.destination
        stations = d2s_dict[d]
        for idx_s, s in enumerate(stations):
            z2ds_dict.update({idx * sim_n + idx_s + cs_num * 3: s})
    z2s_df = pd.DataFrame.from_dict(z2ds_dict, orient='index', columns=['station'])
    z2s_df['loc'] = z2s_df['station'].apply(lambda x: cs_dict[x])
    return z2s_df


def load_vs_models(path):
    vs_models = pd.read_csv(path, index_col=0)
    vs_models.set_index('v_name', inplace=True)

    return vs_models


def get_d2s_realdict(parking_df: pd.DataFrame, cs_gdf: gpd.GeoDataFrame, nodes_sim: gpd.GeoDataFrame, network,
                 near_n=20, sim_n=3, distance_limit=100000.0, is_projected=False):
    """
    Find all destinations and corresponding sim_cs in advance
    :param distance_limit:
    :param near_n:
    :param sim_n:
    :param parking_df: vs_parking_trip_df
    :param cs_gdf:
    :param nodes_sim:
    :param network:
    :return: A MultiIndex df like [destination, station, d2s_length]
    """
    destinations = parking_df['destination'].unique().tolist()
    cs_list = cs_gdf['node_id'].to_list()

    sim_cs_dict = {}
    '''Use @find_near_cs and @find_sim_cs to find all d2s'''
    for d in destinations:
        near_cs = find_near_cs(d, near_n, cs_gdf, nodes_sim, is_projected=is_projected)
        sim_cs = find_sim_cs(d, near_cs, network, nodes_sim, cs_gdf, distance_limit, sim_n, projected=is_projected)
        # cs_dict = dict(map(lambda x: [x[0], list(x[1].values())[0]], sim_cs.T.to_dict().items()))
        # cs_dict = {}
        for s, row in sim_cs.iterrows():
            if d in sim_cs_dict:
                sim_cs_dict[d].append({'station': s, 'distance': row.distance})
            else:
                # 如果 des_id 不存在，可以选择创建一个新的条目
                sim_cs_dict[d] = [{'station': s, 'distance': row.distance}]

    return sim_cs_dict


def simplify_vs_df(vs_parking: pd.DataFrame,bus_range=50):
    vs_df = vs_parking.copy(deep=True)
    # vs_df = vs_df.sort_values(by=['v_name', 'trip'])
    vs_df['index']=vs_df.index
    real_range = bus_range * 1000

    result_df = pd.DataFrame(columns=vs_df.columns)

    # 遍历每个 v_name 的组
    for v_name, group in vs_df.groupby('v_name'):
        group = group.reset_index(drop=True)
        # 初始化变量
        accumulated_distance = 0
        accumulated_duration = 0
        merged = 0
        i=0
        while i < len(group):
        # for i in range(len(group)):
            accumulated_distance += group.iloc[i]['distance']
            accumulated_duration += group.iloc[i]['duration']
            if merged or len(group)==1:
                merged_row = group.iloc[i].copy()
                result_df = result_df._append(merged_row, ignore_index=True)
            elif i != len(group) - 1:
                if (group['s_time'][i+1]-group['e_time'][i]).total_seconds()/60 >= 60:
                    accumulated_distance -= group.iloc[i]['distance']
                    accumulated_duration -= group.iloc[i]['duration']
                    i -=1
                    end_index = i
                    if accumulated_distance:
                        merged_row = group.iloc[end_index].copy()
                        merged_row['distance'] = accumulated_distance
                        merged_row['duration'] = accumulated_duration
                        merged_row['avg_velocity'] = accumulated_distance / 1000 / accumulated_duration * 60
                        merged_row['s_time'] = group.iloc[0]['s_time']
                        merged_row['orientation_coords'] = group.iloc[0]['orientation_coords']
                        result_df = result_df._append(merged_row, ignore_index=True)
                    merged = 1
                elif accumulated_distance > real_range:
                    accumulated_distance -= (group.iloc[i]['distance']+group.iloc[i-1]['distance'])
                    accumulated_duration -= (group.iloc[i]['duration']+group.iloc[i-1]['duration'])
                    i -= 2
                    end_index = i
                    if accumulated_distance:
                        merged_row = group.iloc[end_index].copy()
                        merged_row['distance'] = accumulated_distance
                        merged_row['duration'] = accumulated_duration
                        merged_row['avg_velocity'] = accumulated_distance/1000/accumulated_duration*60
                        merged_row['s_time'] = group.iloc[0]['s_time']
                        merged_row['orientation_coords'] = group.iloc[0]['orientation_coords']
                        result_df = result_df._append(merged_row, ignore_index=True)
                    merged = 1
            else:
                if accumulated_distance > real_range:
                    accumulated_distance -= (group.iloc[i]['distance']+group.iloc[i-1]['distance'])
                    accumulated_duration -= (group.iloc[i]['duration']+group.iloc[i-1]['duration'])
                    i -= 2
                    end_index = i
                    if accumulated_distance:
                        merged_row = group.iloc[end_index].copy()
                        merged_row['distance'] = accumulated_distance
                        merged_row['duration'] = accumulated_duration
                        merged_row['avg_velocity'] = accumulated_distance/1000/accumulated_duration*60
                        merged_row['s_time'] = group.iloc[0]['s_time']
                        merged_row['orientation_coords'] = group.iloc[0]['orientation_coords']
                        result_df = result_df._append(merged_row, ignore_index=True)
                    merged = 1
                else:
                    accumulated_distance -= group.iloc[i]['distance']
                    accumulated_duration -= group.iloc[i]['duration']
                    i -= 1
                    end_index = i
                    if accumulated_distance:
                        merged_row = group.iloc[end_index].copy()
                        merged_row['distance'] = accumulated_distance
                        merged_row['duration'] = accumulated_duration
                        merged_row['avg_velocity'] = accumulated_distance / 1000 / accumulated_duration * 60
                        merged_row['s_time'] = group.iloc[0]['s_time']
                        merged_row['orientation_coords'] = group.iloc[0]['orientation_coords']
                        result_df = result_df._append(merged_row, ignore_index=True)
                    merged = 1

            i+=1
    result_df = result_df.sort_values(by=['index'], ascending=[True])
    result_df = result_df.drop('index', axis=1)

    # result_df = result_df.sort_values(by=['v_name', 'trip'], ascending=[True, True])
    result_df['trip'] = result_df.groupby('v_name').cumcount() + 1
    # result_df = result_df.sort_values(by=['e_time', 's_time'], ascending=[True, True])
    result_df = result_df.reset_index(drop=True)
    return result_df


def str_to_tuple(coord_str):
    return tuple(map(float, coord_str.strip('()').split(',')))


transformer = Transformer.from_crs("epsg:4326", "epsg:4547", always_xy=True)
def transform_coord(xy):
    x_lon, y_lat = xy
    x_newcrs, y_newcrs = transformer.transform(x_lon, y_lat)
    return (x_newcrs, y_newcrs)

def preprocess_vs_df(vs_parking: pd.DataFrame):
    vs_parking['s_time'] = vs_parking['s_time'].apply(parser.parse)
    vs_parking['e_time'] = vs_parking['e_time'].apply(parser.parse)
    vs_parking.sort_values(['e_time'], inplace=True, ignore_index=True)

    vs_parking['orientation_coords'] = vs_parking['orientation_coords'].apply(str_to_tuple)
    vs_parking['destination_coords'] = vs_parking['destination_coords'].apply(str_to_tuple)
    vs_parking['orientation_coords'] = vs_parking['orientation_coords'].apply(transform_coord)
    vs_parking['destination_coords'] = vs_parking['destination_coords'].apply(transform_coord)

    return vs_parking


def get_edge_attr_dist(coords, edge_index):
    # 计算节点之间的欧氏距离矩阵 → [N, N]
    distance_matrix = cdist(coords, coords)

    # 将距离矩阵转换为边权重
    attr_dist = torch.tensor(distance_matrix[edge_index[0], edge_index[1]], dtype=torch.float32).view(-1, 1)
    return attr_dist


def parse_coord(s):
    return tuple(map(float, s.strip('() ').split(',')))


# 查找每个 origin/dest 的最近站点索引
def find_nearest_station(point, tree):
    dist, idx = tree.query([point], k=1)
    return idx[0][0]


def get_edge_attr_route(vs_parking0: pd.DataFrame, coords, num_nodes, edge_index):
    vs_parking = vs_parking0.copy(deep=False)
    # 将字符串解析为浮点元组
    # vs_parking['origin_point'] = vs_parking['orientation_coords'].apply(parse_coord)
    # vs_parking['dest_point'] = vs_parking['destination_coords'].apply(parse_coord)

    # 构建 BallTree 加速最近邻查询
    tree = BallTree(coords, metric='euclidean')

    vs_parking['origin_node'] = vs_parking['orientation_coords'].apply(lambda p: find_nearest_station(p, tree))
    vs_parking['dest_node'] = vs_parking['destination_coords'].apply(lambda p: find_nearest_station(p, tree))

    # 初始化 OD 边频次字典
    od_count = defaultdict(int)

    for _, row in vs_parking.iterrows():
        o = row['origin_node']
        d = row['dest_node']
        if o != d:
            od_count[(o, d)] += 1

    # 构建稀疏 OD 边权重矩阵
    od_weights = np.zeros((num_nodes, num_nodes))

    for (o, d), count in od_count.items():
        od_weights[o, d] = count

    # 转换为 PyTorch Tensor 并映射到边上
    od_edge_weights = torch.tensor(od_weights[edge_index[0], edge_index[1]], dtype=torch.float32).view(-1, 1)

    return od_edge_weights


def get_edge_attrs(vs_parking0: pd.DataFrame, station_gdf: gpd.GeoDataFrame):
    """
    station_gdf: epsg4507
    """
    coords = np.array([(point.x, point.y) for point in station_gdf['geometry']])
    num_nodes = coords.shape[0]
    edge_index = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes)).T  # [2, N*N]

    edge_attr_dist = get_edge_attr_dist(coords, edge_index)
    edge_attr_route = get_edge_attr_route(vs_parking0, coords, num_nodes, edge_index)

    edge_attr = torch.cat([edge_attr_dist, edge_attr_route], dim=1)  # shape: [N*N, 2]

    return edge_attr


if __name__ == '__main__':
    vs_parking_df = pd.read_csv(r'../../hkbus/data/HKsimpreparation/HK_all_vs_parking_nodeid.csv')  # 车辆行程
    vs_parking_df.reset_index(drop=True, inplace=True)
    vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
    vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
    vs_parking_df = vs_parking_df.drop(columns=['distance'])
    vs_parking_df = vs_parking_df.rename(columns={'evrange': 'distance'})

    vs_parking_df_new = simplify_vs_df(vs_parking_df,260,191)
    print(vs_parking_df['distance'].sum(), vs_parking_df_new['distance'].sum())

