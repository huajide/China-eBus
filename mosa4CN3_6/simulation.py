"""
@Project ：DissertationPY 
@File    ：simulation.py
@IDE     ：PyCharm 
@Author  ：Xander PENG
@Date    ：11/8/2022 02:02 
"""

from datetime import datetime, timedelta
import geopandas as gpd
import networkx as nx
import pandas as pd
from haversine import haversine
import logging
from treelib import Tree, Node
from typing import List, Dict
from vehicle_type import VehicleTypes
# import random
# import station
# from charger import Charger, SlowCharger, FastCharger
import sim_utils


class SimVehicle:

    def __init__(self,
                 name,
                 trips: List[int],
                 s_times: List[datetime or str],
                 e_times: List[datetime or str],
                 parking_nodes: List[int],
                 distances: List[float],
                 speeds: List[float] = None,
                 vehicle_type: VehicleTypes = VehicleTypes(0, v_type='large')
                 ):

        # Static Attributes
        self.v_name = name
        self.trips = trips

        # Convert 'time' into Datetime instance
        if isinstance(s_times[0], str):  # If it is string, convert it into Datetime instance
            self.trip_s_time = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in s_times]
        else:
            self.trip_s_time = s_times

        if isinstance(e_times[0], str):  # If it is string, convert it into Datetime instance
            self.trip_e_time = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in e_times]
        else:
            self.trip_e_time = e_times

        self.destinations = parking_nodes
        self.distances = distances

        # Vehicle type attr
        self.model = vehicle_type
        self.driving_range = self.model.driving_range
        self.battery = self.model.battery
        self.e2d_rate = self.driving_range / self.battery  # n km per kwh

        self.velocity_avg = speeds  # km/h

        self.data_tree = Tree()  # Build a data tree once instantiation
        self.data_tree.create_node(tag=self.v_name, identifier=self.v_name, data={})  # Set root

        # # Dynamic Attributes
        # self.simulated_trip = []  # Record which trip is being simulated
        # self.visited_stations = {}  # Record {trip: visited stations}
        # self.arr_station_time = {}  # Record {trip1: {s1: arr_time1, s2: arr_time2, ...}, trip2: ...}
        # self.charging_time = {}  # Record {trip1: {s1: ch_time1, s2: ch_time2, ...}, trip2: ...}
        # self.back_time = {}  # Record {trip1: {s1: back_time1, s2: back_time2, ...}, trip2: ...}


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model
        self.driving_range = new_model.driving_range
        self.battery = new_model.battery
        self.e2d_rate = self.driving_range / self.battery  # n km per kwh


    def get_trip_info(self, trip_idx: int, sim_instance: bool = False):

        """
        Get certain trip's info of this simVehicle instance; Return a dict(Default) or Simulation instance)

        :param trip_idx: The index of trip(start from 1)
        :param sim_instance: True: return Simulation instance; False: return dict
        :return: Certain trip's info
        """

        # Check the legibility of input trip_idx
        if trip_idx <= len(self.trips):
            idx = trip_idx - 1
        else:
            raise IndexError('The input trip is larger than the max trip number!')

        # Return a dict or Simulation instance based on @sim_instance value
        if not sim_instance:  # Return a dict containing this trip info
            trip_info_dict = {'v_name': self.v_name,
                              'trip': self.trips[idx],
                              's_time': self.trip_s_time[idx],
                              'e_time': self.trip_e_time[idx],
                              'destination': self.destinations[idx],
                              'distance': self.distances[idx],

                              }

            if self.velocity_avg is not None:
                trip_info_dict.update({'velocity_avg': self.velocity_avg[idx]})

            return trip_info_dict

        elif sim_instance:  # Return a Simulation instance
            if self.velocity_avg is not None:
                sim = SimVehicleTrip(self.v_name, trip_idx, self.trip_s_time[idx], self.trip_e_time[idx],
                                     self.destinations[idx], self.distances[idx], self.driving_range, self.battery,
                                     self.velocity_avg[idx])
            else:
                sim = SimVehicleTrip(self.v_name, trip_idx, self.trip_s_time[idx], self.trip_e_time[idx],
                                     self.destinations[idx], self.distances[idx], self.driving_range, self.battery)

            return sim

        else:  # If the @sim_instance appointed a wrong value. raise error
            raise ValueError("The param 'sim_instance' should be 'True' or 'False'!")


class SimVehicleTrip:

    def __init__(self,
                 v_name,
                 trip_idx: int,  # the index of vehicle trip, start from 1
                 trip_s_time: datetime or str,
                 trip_e_time: datetime or str,
                 parking_node: int,
                 trip_distance: float,
                 driving_range: float,
                 battery: float,
                 velocity_avg=None,
                 free_speed=30):

        # Attributes
        self.v_name = v_name
        self.trip_idx = int(trip_idx)
        self.free_speed = free_speed  # km/h, speed of no stopping

        # Check and try to convert trip_time format
        if isinstance(trip_s_time, str):
            st = datetime.strptime(str(trip_s_time), "%Y-%m-%d %H:%M:%S")  # Covert time str into datetime object
            self.s_time = st
        elif isinstance(trip_s_time, datetime):
            self.s_time = trip_s_time
        else:
            raise ValueError('trip_time is neither str or datetime')
        # Process trip end time
        if isinstance(trip_e_time, str):
            et = datetime.strptime(str(trip_e_time), "%Y-%m-%d %H:%M:%S")  # Covert time str into datetime object
            self.e_time = et
        elif isinstance(trip_e_time, datetime):
            self.e_time = trip_e_time
        else:
            raise ValueError('trip_time is neither str or datetime')

        self.destination = int(float(parking_node))
        self.distance = float(trip_distance)
        self.driving_range = driving_range
        self.battery = battery
        self.velocity_avg = velocity_avg

    def find_near_cs(self, n: int, cs_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Find n near charging stations for current v's parking location based on Euclidean distance;

        :param n: the amount of cs to be selected
        :param cs_gdf: candidate charging stations gdf
        :param nodes_gdf: nodes gdf
        :return: A df containing nearest n osmid of candidate charging stations and corresponding d2s
        """
        # Get parking destination's lon and lat
        destination_node = self.destination
        destination_lat, destination_lon = nodes_gdf.query("node_id == @destination_node").iloc[0][['y', 'x']]
        destination_loc = (destination_lat, destination_lon)

        # Cal the distance between destination and every candidate charging station
        d2s_df = pd.DataFrame(cs_gdf.apply(lambda x: haversine((x.lat, x.lon), destination_loc), axis=1),
                              columns=['d2s'])

        # Filter n nearest charging stations(Euclidean distance)
        near_d2s_df: pd.DataFrame = d2s_df.nsmallest(n, 'd2s')  # Get n smallest d2s distance df
        near_cs_idx: list = near_d2s_df.index.to_list()  # Get accordance idx
        near_cs: list = cs_gdf.loc[near_cs_idx, 'node_id'].to_list()  # Get near cs osmid
        near_d2s_df['node_id'] = near_cs

        return near_d2s_df

    def find_sim_cs(self, near_cs: pd.DataFrame, network: nx.Graph, nodes_gdf: gpd.GeoDataFrame,
                    cs_gdf: gpd.GeoDataFrame, distance_limit: float, n: int) -> pd.DataFrame:
        """
        Find n closest charging stations between vehicle destination and candidate charging stations;
        Here is a @near_cs_list containing some candidate cs id, which is derived from @find_near_cs;

        :param near_cs: A df containing charging stations osmid and d2s
        :param network: networkx Graph
        :param nodes_gdf: nodes gdf
        :param cs_gdf: charging stations gdf
        :param distance_limit: the limit between destination location and candidate station
        :param n: the amount of sim_stations to be selected
        :return: A 2-col data frame [station_id, d2s_path_length]
        """

        destination_node = self.destination

        d2s_dict = {}
        for cs in near_cs['node_id']:
            try:  # Find the shortest path length in the network graph
                d2s_path_length = nx.shortest_path_length(network, destination_node, cs, weight='length')
            except nx.NetworkXException:  # It is possible that there is no path from d to s
                d2s_path_length = 999999  # Appoint a large number in order to filter it later

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
            near_cs_list = self.find_near_cs(len(near_cs) * 2, cs_gdf, nodes_gdf)
            sim_cs = self.find_sim_cs(near_cs_list, network, nodes_gdf, cs_gdf, distance_limit * 2, n)

        else:  # the number of cs == 0
            # print(self.v_name + 'has no valid candidate sim charging station!')
            logging.log(logging.DEBUG, self.v_name + ' has no valid candidate sim charging station!' +
                        ' current near_cs_list length is: ' + str(len(near_cs)))

            # Enlarge near cs search filed and distance_limit until find valid station
            near_cs_list = self.find_near_cs(len(near_cs) * 2, cs_gdf, nodes_gdf)
            sim_cs = self.find_sim_cs(near_cs_list, network, nodes_gdf, cs_gdf, distance_limit * 2, n)

        return sim_cs

    def get_sim_cs(self, all_d2s_dict):
        """
        Get current destination's corresponding sim_cs via @all_d2s_dict

        :param all_d2s_dict: A dict containing all destination' stations
        :return: A df of this destination's stations
        """
        d = self.destination
        current_sim_cs = all_d2s_dict.get(d, [])
        return current_sim_cs

    def simulation(self, sim_v: SimVehicle,
                   station_dict,  #: Dict[int, station.Station]
                   solutions,  # Vars: An individual of the population
                   sim_cs_method='find',
                   all_d2s_dict=None,
                   cs_dict: dict=None
                   ):
        """
        The main entrance for simulating the behavior of operating and finding charging facility

        :param network: The road network
        :param nodes_gdf: A GeoDataFrame of network nodes
        :param edges_gdf: A GeoDataFrame of network edges
        :param cs_gdf: A GeoDataFrame of candidate charging stations
        :param sim_v: SimVehicle instance
        :param station_dict: A charging station dict containing all Stations instance
        :param sim_v_info: A DataFrame of all Vehicles' info
        :param solutions: Vars
        :param sim_cs_method: 'find' indicates use @find_near/sim_cs; 'get' means read sim_cs directly from @all_d2s_dict
        :param all_d2s_dict: A dict containing all destinations' corresponding stations
        :return:  [e_demand, d2s_all, timeout_all, wait_time_all]
        """

        if self.trip_idx == 1:  # For the first trip of every sim_v
            # initialize the data tree
            sim_v.data_tree = Tree()  # Build a data tree once instantiation
            sim_v.data_tree.create_node(tag=sim_v.v_name, identifier=sim_v.v_name, data={})  # Set root

            # the soc at the end of trip; within [0, 1]
            soc_trip: float = (self.driving_range - (self.distance / 1000)) / self.driving_range
            # the energy consumption of this trip
            e_trip = (1 - soc_trip) * sim_v.battery

            # Cal charging demand (if remaining driving range can fulfill the next trip -> 0; else 1; or interval > 40)
            try:
                next_trip_distance: float = sim_v.distances[self.trip_idx]  # Get next trip's distance; unit: m
            except IndexError:  # Only 1 trip for this simV
                next_trip_distance = 59999.9

            # Get the time interval between the end of this trip and the start of next trip; unit: minutes
            try:
                interval = (sim_v.trip_s_time[self.trip_idx] - sim_v.trip_e_time[self.trip_idx - 1]).seconds / 60
            except IndexError:  # Only 1 trip for this simV
                interval = 9999.9
            # cd充电需求
            cd: int or List[int] = (lambda x, y: 1 if (x <= y) | (interval >= 60) else 0)(soc_trip * self.driving_range,
                                                                                          next_trip_distance / 1000)
            # There will be two situations: cd == 1 or 0;
            if cd == 0:  # No charging demand generated
                # charging_distance = 0
                # e_demand = 0  # electricity demand
                soc_back = soc_trip
                node_data = {'cd': cd,
                             'e_trip': e_trip,
                             'e_distance': 0,
                             'soc_back': soc_back,
                             'ch_timeout': 0,
                             'wait_time': 0
                             }

                sim_v.data_tree.create_node(tag='t1s0',
                                            identifier=(0,),
                                            parent=self.v_name,
                                            data=node_data)

                return [e_trip, 0, 0, 0, self.distance/1000]

            else:  # cd == 1, there is charging demand
                e_demand = (1 - soc_trip) * sim_v.battery  # The charging demand volume; unit: Kw/h, within [0, battery]

                sim_cs_nodeid, sim_cs_dist = self.choose_station(solutions, all_d2s_dict,cs_dict)

                # Get the soc when sim_v arrive at target station. dict(s_osmid: soc_arr)
                soc_arr_s = {sim_cs_nodeid: soc_trip - (sim_cs_dist / 1000 / self.driving_range)}  # Dict[int, float]
                arr_time_s = {sim_cs_nodeid:
                                  self.e_time + timedelta
                                  (hours=sim_cs_dist / 1000 / self.free_speed)}  # Dict[int, datetime]

                # Simulate charging behavior in each station
                target_station = station_dict.get(sim_cs_nodeid)  # Get sim stations instance
                # Store the sum of d2s, timeout and wait_time
                d2s_all = 0
                timeout_all = 0
                wait_time_all = 0
                # The distance between destination and target station; unit: meter
                current_cs_name = target_station.s_name
                d2s = sim_cs_dist
                e_trip_time = d2s/1000/self.free_speed  # hour

                # Get charging info; Format: [the soc after charging, charging_timeout, waiting time]
                soc_ch, ch_timeout, wait_time = target_station.charging_request(sim_v, self,
                                                                    arr_time_s.get(current_cs_name),
                                                                    soc_arr_s.get(current_cs_name), e_trip_time)
                # Sum the 3 ObjV in this scenario
                d2s_all += d2s*2
                timeout_all += ch_timeout
                wait_time_all += wait_time

                soc_back = soc_ch - (d2s / 1000 / self.driving_range)
                node_data = {'cd': cd,
                             'e_trip': e_trip,
                             'e_distance': d2s / 1000 / sim_v.driving_range * sim_v.battery,
                             'soc_back': soc_back,
                             'ch_timeout': ch_timeout,
                             'wait_time': wait_time}

                # Add data into the tree root
                sim_v.data_tree.create_node(tag='t1s' + str(current_cs_name),
                                            identifier=tuple([current_cs_name]),
                                            parent=self.v_name,
                                            data=node_data
                                            )
                # Add visit record in the root
                sim_v.data_tree.get_node(sim_v.data_tree.root).data.update({tuple([0]): [sim_cs_nodeid]})
                '''
                When there is cd in t1, t1 will be recorded in the data_tree as {(0,): [target_cs]};
                However, if there is cd in t2, the data_tree will record {(0,) : [target_cs]} too,
                which leads to confusing.
                Even though t1 will be represented as {(0,): [cs]} and t2 will be described as {(cs_id,): [cs]}
                when there are cd in both t1 and t2.
                '''
                return [e_trip, d2s_all / 1000 / sim_v.driving_range * sim_v.battery, timeout_all, wait_time_all,
                        (self.distance+d2s_all)/1000]

        else:  # For non_first trip
            # energy consumption of this trip
            e_trip = (self.distance / 1000 / self.driving_range) * sim_v.battery

            # Get previous trip's soc_back
            previous_nodes = sim_v.data_tree.leaves()  # Get last trip's all leaves nodes

            # try:
            previous_soc = [node.data.get('soc_back') for node in previous_nodes]  # Get the soc_back of last trip
            # except:
            #     print(sim_v.v_name)
            #     print(self.trip_idx)
            #     print(sim_v.data_tree.all_nodes())

            # Cal the soc after this trip
            soc_trips = [ps - (self.distance / 1000 / self.driving_range) for ps in previous_soc]
            # Cal the charging demand as well as their volume
            try:
                next_trip_distance = sim_v.distances[self.trip_idx]
            except IndexError:  # The last trip's index will out of range
                next_trip_distance = 59999.9

            try:
                interval = (sim_v.trip_s_time[self.trip_idx] - sim_v.trip_e_time[self.trip_idx - 1]).seconds / 60
            except IndexError:  # The last trip's index will out of range
                interval = 9999

            cds = [(lambda x, y: 1 if (x <= y) | (interval >= 60) else 0)(soc_trip * self.driving_range,
                                                                          next_trip_distance / 1000)
                   for soc_trip in soc_trips]

            e_demand_list = [(1 - soc_trip) * sim_v.battery for soc_trip in soc_trips]  # Electricity demand

            # if (self.v_name == 'B67762') & (self.trip_idx == 2):
            #     print(previous_nodes)
            #     print(soc_trips)
            #     print(cds)
            if any(map(lambda x: x == 1, cds)):
                # Find or get valid candidate charging stations
                sim_cs_nodeid, sim_cs_dist = self.choose_station(solutions, all_d2s_dict, cs_dict)

            # Create 3 variables to store the final results of all scenarios
            e_trip_all = 0
            d2s_all = 0
            timeout_all = 0
            wait_time_all = 0

            for cd_idx, cd in enumerate(cds):
                last_node: Node = previous_nodes[cd_idx]  # The accordance last trip's scenario
                if cd == 0:  # No charging demand
                    e_trip_all += e_trip
                    # soc_back = previous_soc[cd_idx]
                    soc_back = soc_trips[cd_idx]  # Zili

                    node_data = {'cd': cd,
                                 'e_trip': e_trip,
                                 'e_distance': 0,
                                 'soc_back': soc_back,
                                 'ch_timeout': 0,
                                 'wait_time': 0
                                 }
                    # Add this scenario's info into the data tree
                    sim_v.data_tree.create_node(tag='t' + str(self.trip_idx) + 's0',
                                                identifier=sim_utils.set_node(last_node, self.trip_idx, para='id'),
                                                parent=last_node.identifier,
                                                data=node_data)

                else:  # cd = 1
                    e_demand = e_demand_list[cd_idx]
                    e_trip_all += e_trip
                    # Get the soc when sim_v arrive at target station. dict(s_osmid: soc_arr)
                    soc_arr_s = {
                        sim_cs_nodeid: soc_trips[cd_idx] - (sim_cs_dist / 1000 / self.driving_range)}  # Dict[int, float]
                    arr_time_s = {sim_cs_nodeid:
                                      self.e_time + timedelta
                                      (hours=sim_cs_dist / 1000 / self.free_speed)}  # Dict[int, datetime]

                    # Simulate charging behavior in each station
                    target_station = station_dict.get(sim_cs_nodeid)  # Get sim stations instance

                    # Create 3 variables to store accordance value in a singe 'cd' scenario
                    d2s_sum = 0
                    timeout_sum = 0
                    wait_time_sum = 0
                    current_cs_name = target_station.s_name
                    d2s = sim_cs_dist
                    e_trip_time = d2s / 1000 / self.free_speed  # hour

                    # Get charging info; Format: [the soc after charging, charging_timeout, waiting time]
                    soc_ch, ch_timeout, wait_time = target_station.charging_request(sim_v, self,
                                                                        arr_time_s.get(current_cs_name),
                                                                        soc_arr_s.get(current_cs_name),e_trip_time)
                    # Add the value of this station scenario into the 'sum_variable'
                    d2s_sum += d2s*2
                    timeout_sum += ch_timeout
                    wait_time_sum += wait_time

                    soc_back = soc_ch - (d2s / 1000 / self.driving_range)
                    node_data = {'cd': cd,
                                 'e_trip': e_trip,
                                 'e_distance': d2s / 1000 / self.driving_range * sim_v.battery,
                                 'soc_back': soc_back,
                                 'ch_timeout': ch_timeout,
                                 'wait_time': wait_time}

                    # Add data into the data tree
                    sim_v.data_tree.create_node(tag='t' + str(self.trip_idx) + 's' + str(current_cs_name),
                                                identifier=sim_utils.set_node(last_node, self.trip_idx,
                                                                              cs_id=current_cs_name, cd=1, para='id'
                                                                              ),
                                                parent=last_node.identifier,
                                                data=node_data)
                    # Add the 'sum_value' into the all ObjV.
                    d2s_all += d2s_sum
                    timeout_all += timeout_sum
                    wait_time_all += wait_time_sum

                    # Add visit records
                    sim_v.data_tree.get_node(sim_v.data_tree.root).data.update({last_node.identifier: [sim_cs_nodeid]})

            return [e_trip, d2s_all / 1000 / self.driving_range * sim_v.battery, timeout_all, wait_time_all,
                    (self.distance+d2s_all)/1000]


    def choose_station(self, solutions, all_d2s_dict, cs_dict):
        d = self.destination
        chosen_cs_nodeid = -1
        chosen_cs_distance = 99000  # Init with max distance
        sim_cs_list = all_d2s_dict.get(d, [])
        for one_d2s in sim_cs_list:
            if solutions[cs_dict[one_d2s['station']]] == 1:
                chosen_cs_nodeid = one_d2s['station']
                chosen_cs_distance = one_d2s['distance']
                break
        if chosen_cs_nodeid < 0:
            print('no candidate station is built')
        return chosen_cs_nodeid, chosen_cs_distance  # node_id of selected charging station