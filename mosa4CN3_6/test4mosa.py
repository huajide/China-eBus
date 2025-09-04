# @Time : 2025-06-18 16:39
# @Author : Xander PENG
# @Revised : Zili Tian
# @File : test4mosa.py
# @Software: PyCharm
# @Description: v3.6 keeps the single objective - cost only.
# Correspondingly, the numbers of vehicles and chargers are directly decided by the simulation process.

import time
import numpy as np
import pandas as pd
import geopandas as gpd
import math

from simulation import SimVehicle, SimVehicleTrip
from station import Station
import networkx as nx
import data_utils
import sim_utils
from vehicle_type import VehicleTypes,driving_range
from mosa import MOSA
from mosa import Problem
import pickle
from pyproj import CRS
from vehicle_scheduling import extra_timesave
from dateutil import parser
import os


class Location(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self,
                         kwargs.get('num_vars'),  # var_num
                         [0] * kwargs.get('num_vars'),  # Integer or Real
                         [0] * len(kwargs.get('cs_gdf')) + [0,0,0],  # lb
                         [1] * len(kwargs.get('cs_gdf')) + [9999,9999,9999],  # ub
                         2,  # num of f
                         1,  # num of cv
                         **kwargs
                         )
        self.num_vars = kwargs.get('num_vars')
        self.sim_v_info = kwargs.get('sim_v_info')
        self.sim_v_dict = kwargs.get('sim_v_dict')
        self.cs_gdf = kwargs.get('cs_gdf')
        self.cs_num = len(self.cs_gdf)
        self.vs_parking_df = kwargs.get('vs_parking_df')
        self.all_d2s_dict = kwargs.get('all_d2s_dict')
        self.cs_dict = kwargs.get('cs_dict')

        self.v_name = self.vs_parking_df['v_name'].to_list()
        self.trip = self.vs_parking_df['trip'].to_list()
        self.s_time = self.vs_parking_df['s_time'].to_list()
        self.e_time = self.vs_parking_df['e_time'].to_list()
        self.destination = self.vs_parking_df['destination'].to_list()
        self.distance = self.vs_parking_df['distance'].to_list()
        self.avg_velocity = self.vs_parking_df['avg_velocity'].to_list()
        self.v_type = self.vs_parking_df['vehicle_type'].to_list()

        self.simv_v_name = self.sim_v_info.index.to_list()
        self.simv_trip = self.sim_v_info['trip'].to_list()
        self.simv_s_time = self.sim_v_info['s_time'].to_list()
        self.simv_e_time = self.sim_v_info['e_time'].to_list()
        self.simv_destination = self.sim_v_info['destination'].to_list()
        self.simv_distance = self.sim_v_info['distance'].to_list()
        self.simv_avg_velocity = self.sim_v_info['avg_velocity'].to_list()
        self.simv_v_type = self.sim_v_info['vehicle_type'].to_list()

        self.large_num = self.simv_v_type.count('large')
        self.medium_num = self.simv_v_type.count('medium')
        self.small_num = self.simv_v_type.count('small')

        self.e_price = kwargs.get('e_price')


    def eval_vars(self, vars_, is_test=False, *args):

        if "CV" in args:
            cv1 = sim_utils.set_cv1(self.cs_num, vars_)
            return np.hstack([cv1])

        cal_s_time = time.perf_counter()
        # Instantiate all sim_v and stations as well as simVTrip
        for i in range(len(self.simv_v_name)):
            if self.simv_v_type[i] == 'large':
                vehi_type = vars_[self.cs_num]
            elif self.simv_v_type[i] == 'medium':
                vehi_type = vars_[self.cs_num+1]
            else:
                vehi_type = vars_[self.cs_num+2]
            vehi = VehicleTypes(vehi_type, v_type=self.simv_v_type[i])
            self.sim_v_dict[self.simv_v_name[i]].model = vehi

        station_dict = {row.node_id: Station(row.node_id, idx, fast_charger=20,
                                        slow_charger=20) for idx, row in self.cs_gdf.iterrows()}

        timeout_list = []
        # Start simulation
        e_trip_sum, e_d2s_sum, wait_time_sum, emission_sum = 0, 0, 0, 0 # Initialize 4 variables for storing simulation values
        for i in range(len(self.v_name)):
            sim_v_trip = SimVehicleTrip(self.v_name[i], self.trip[i], self.s_time[i], self.e_time[i], self.destination[i],
                                        self.distance[i],self.sim_v_dict.get(self.v_name[i]).driving_range,
                                        self.sim_v_dict.get(self.v_name[i]).battery, self.avg_velocity[i])
            e_trip, e_d2s, timeout, wait_time, trip_dist = sim_v_trip.simulation(
                                                                      self.sim_v_dict.get(self.v_name[i]),
                                                                      station_dict, vars_,
                                                                      sim_cs_method='get',
                                                                      all_d2s_dict=self.all_d2s_dict, cs_dict=self.cs_dict)
            e_trip_sum += e_trip
            e_d2s_sum += e_d2s
            wait_time_sum += wait_time
            timeout_list.append(timeout)

            emission_sum += trip_dist * self.sim_v_dict.get(self.v_name[i]).model.per_emission

        fast_charger_counts = [station.max_used_fast_charger for station in station_dict.values()]
        slow_charger_counts = [station.max_used_slow_charger for station in station_dict.values()]

        # add extra vehicle:
        saved_time, subs_idx, extra_large, extra_medium, extra_small = extra_timesave(self.v_name,self.distance,timeout_list,self.v_type,
                                    VehicleTypes(vars_[self.cs_num], v_type='large').driving_range,
                                    VehicleTypes(vars_[self.cs_num+1], v_type='medium').driving_range,
                                    VehicleTypes(vars_[self.cs_num+2], v_type='small').driving_range)
        if is_test:
            return timeout_list, subs_idx

        # System costs
        # Vehicle cost
        vehicle_cost = ((self.large_num+extra_large)*VehicleTypes(vars_[self.cs_num],v_type='large').fix_cost +
                        (self.medium_num + extra_medium) * VehicleTypes(vars_[self.cs_num+1], v_type='medium').fix_cost +
                        (self.small_num + extra_small) * VehicleTypes(vars_[self.cs_num+2], v_type='small').fix_cost)

        # Station construction and maintenance costs per year
        station_cost = sum(vars_[0: self.cs_num]) * 600000 * 1
        station_emission = sum(vars_[0: self.cs_num]) * 80
        # 9w and 3w for fast and slow charger, respectively
        # Get chargers' num of selected stations and do calculation
        # sel_x = np.where(vars_[:self.cs_num] == 1)[0]  # Indexes of selected stations
        charger_cost = sum(fast_charger_counts) * 4000 + sum(slow_charger_counts) * 2000

        # 1.2 yuan/kwh
        # The life cycle cost of energy consumption in this trip， including both operation and go-charging distances
        trip_cost = (e_trip_sum+e_d2s_sum) * 365 * self.e_price * 1

        # print(f'Station Count: {sum(vars_[:self.cs_num])}; Extra Vehicles: {extra_large} {extra_medium} {extra_small}')

        '''one-objective'''
        # emission_cost = (emission_sum*365/1000+station_emission)*1.05  # kg * 1.05 yuan/kg social cost of emission
        # f0 = -(vehicle_cost + station_cost + charger_cost + trip_cost + emission_cost) / 1000000  # 1M yuan/year
        '''multi-objective'''
        f1 = -(vehicle_cost + station_cost + charger_cost + trip_cost)/1000000  # 1M yuan/year
        f2 = -(emission_sum*365/1000+station_emission)/1000  # T/year

        cv1 = sim_utils.set_cv1(self.cs_num, vars_)  # cv是约束,eq22(ub,lb是eq25-26)
        cv_and_params = ([cv1]+[-x for x in fast_charger_counts]+[-x for x in slow_charger_counts]+
                         [-extra_large, -extra_medium, -extra_small])
        cal_e_time = time.perf_counter()
        # print(f'Single calculation time: {(cal_e_time - cal_s_time):.2f}s')

        # print(f"f1: {-f1:.1f} f2: {-f2:.1f}")

        return np.array([f1,f2]), np.hstack(cv_and_params)


if __name__ == '__main__':
    is_simplified = False
    to_simplify = True
    is_referred = False
    is_dict_loaded = True
    is_tested = True
    SIM_N = 50
    cities = pd.read_csv(rf'../data/18cities.csv')
    city_i = 8
    city_name = cities['city'][city_i]

    if is_simplified:
        vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid_simplified/{city_name}.csv')  # 车辆行程
    else:
        vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')  # 车辆行程

    vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
    vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
    vs_parking_df.sort_values(['e_time'], inplace=True, ignore_index=True)

    cs_gdf = gpd.read_file(rf'../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4507))  # 充电站
    cs_gdf['lon'] = cs_gdf['geometry'].x
    cs_gdf['lat'] = cs_gdf['geometry'].y

    if not is_simplified and to_simplify:
        vs_parking_df0 = vs_parking_df.copy(deep=False)
        vs_parking_df = data_utils.simplify_vs_df(vs_parking_df, min(min(driving_range)))
        vs_parking_df.to_csv(rf'../data/input/vs_parking_nodeid_simplified/{city_name}.csv')

    sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').  # 将车辆行程按车辆集计
                                        apply(lambda x: data_utils.derive_simV_info(x)).to_dict(), orient='index')
    sim_v_dict = {}
    for idx, row in sim_v_info.iterrows():
        sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance, row.avg_velocity)

    if is_dict_loaded:
        with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'rb') as f:
            all_d2s_dict = pickle.load(f)
    else:
        nodes_sim = gpd.read_file(
            rf'../data/road/{city_name}/nodes_sim.shp', crs=CRS.from_epsg(4507))  # 道路节点
        edges_sim = gpd.read_file(
            rf'../data/road/{city_name}/edges_sim.shp', crs=CRS.from_epsg(4507))  # 道路路段
        G = nx.from_pandas_edgelist(df=edges_sim, source='u', target='v', edge_attr=['edge_id', 'length'],
                                    create_using=nx.Graph())
        all_d2s_dict = data_utils.get_d2s_realdict(vs_parking_df, cs_gdf, nodes_sim, G,
                                          near_n=100, sim_n=SIM_N, distance_limit=10000.0, is_projected=True)
        with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'wb') as f:
            pickle.dump(all_d2s_dict, f)

    num_vars = data_utils.set_var_num(cs_gdf)  # sim_n是后加的，选最近3个station为备选
    print("num_vars: ", num_vars)

    cs_dict = {v: k for k, v in zip(cs_gdf.index, cs_gdf['node_id'])}
    print('read files successfully!')

    problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, sim_v_dict=sim_v_dict, cs_gdf=cs_gdf,
                       vs_parking_df=vs_parking_df,
                       all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, e_price=cities['eprice'][city_i])

    annealing_iters = 100  # 100
    algorithm = MOSA(problem, annealing_iters=annealing_iters)
    # algorithm.end_temperature = 1  # 900
    algorithm.annealing_strength = 0.6
    algorithm.cooling_alpha = 0.9953
    algorithm.min_delta_hv = 0.01
    algorithm.multiprocess = False
    algorithm.is_tested = is_tested

    # # estimate how long would be processed
    print(f"Estimated: {math.log(algorithm.end_temperature/algorithm.initial_temperature)/
                        math.log(algorithm.cooling_alpha)*annealing_iters*2/3600:.1f} h")  # 10s is estimated circle time

    algorithm.early_termination = {'max_iters': 500000, 'max_duration': 240,
                                   'max_no_eliminated': 20000}  # max_duration:运行时长，按小时计
    if is_referred:
        refer_vars = pd.read_csv(
            rf"../data/output/250630/{city_name}_250630/archive_vars.csv").values[:, 1:]
    else:
        refer_vars = sim_utils.set_refer_vars(len(cs_gdf), refer_num=10)  # 生成初始解

    algorithm.load_refer_solutions(refer_vars)
    save_path = rf"../data/output/{city_name}_250728"
    if not os.path.exists(save_path):
        os.makedirs(save_path) # 如果不存在，则创建文件夹
    algorithm.run(inf='infeasible', is_cv=True, path=save_path, store='store',multi_tasks=4)  # ,multi_tasks=4

    if is_tested:
        # convergence = algorithm.output_fitness()

        A = algorithm.archive_history
        # 将所有目标函数值合并成一个数组
        all_objs = np.vstack([objs for objs in A if objs.size != 0])
        # 找出每个目标函数的最大值和最小值
        min_vals = np.min(all_objs, axis=0)
        max_vals = np.max(all_objs, axis=0)
        print("每个目标函数的最小值:", min_vals)
        print("每个目标函数的最大值:", max_vals)
        standard_convergence = algorithm.output_fitness(min_vals, max_vals)

