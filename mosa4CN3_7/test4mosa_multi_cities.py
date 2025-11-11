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
import os
import multiprocessing as mp
from functools import partial

from simulation import SimVehicle, SimVehicleTrip
from station import Station
import networkx as nx
import data_utils
import sim_utils
from vehicle_type import VehicleTypes
from mosa import MOSA
from mosa import Problem
import pickle
from pyproj import CRS
from vehicle_scheduling import extra_timesave
from dateutil import parser


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
        self.base_energy = self.vs_parking_df['base_energy'].to_list()

        self.simv_v_name = self.sim_v_info.index.to_list()
        self.simv_v_type = self.sim_v_info['vehicle_type'].to_list()

        self.large_num = self.simv_v_type.count('large')
        self.medium_num = self.simv_v_type.count('medium')
        self.small_num = self.simv_v_type.count('small')

        self.e_price = kwargs.get('e_price')
        self.default_chargers = 20

    def eval_vars(self, vars_, is_test=False, **kwargs):
        # 获取传入的充电桩数量参数，如果没有则使用默认值
        fast_chargers = kwargs.get('fast_chargers', [self.default_chargers] * self.cs_num)
        slow_chargers = kwargs.get('slow_chargers', [self.default_chargers] * self.cs_num)

        # 检查是否传入了额外车辆参数
        use_custom_extra = ('extra_large' in kwargs and
                            'extra_medium' in kwargs and
                            'extra_small' in kwargs)

        if use_custom_extra:
            extra_large = kwargs.get('extra_large', 0)
            extra_medium = kwargs.get('extra_medium', 0)
            extra_small = kwargs.get('extra_small', 0)

        power_factor = kwargs.get('power_factor', 1)
        e_price_factor = kwargs.get('e_price_factor', 1)
        degradation_factor = kwargs.get('degradation_factor', 1)  # not larger than 1

        cal_s_time = time.perf_counter()
        # Instantiate all sim_v and stations as well as simVTrip
        for i in range(len(self.simv_v_name)):
            if self.simv_v_type[i] == 'large':
                vehi_type = vars_[self.cs_num]
            elif self.simv_v_type[i] == 'medium':
                vehi_type = vars_[self.cs_num + 1]
            else:
                vehi_type = vars_[self.cs_num + 2]

            # 创建一个带有衰减因子的新VehicleTypes对象
            degraded_vehi = VehicleTypes(vehi_type, v_type=self.simv_v_type[i])
            degraded_vehi.battery = degraded_vehi.battery * degradation_factor

            self.sim_v_dict[self.simv_v_name[i]].model = degraded_vehi
            # self.sim_v_dict[self.simv_v_name[i]].driving_range = degraded_vehi.driving_range
            self.sim_v_dict[self.simv_v_name[i]].battery = degraded_vehi.battery
            self.sim_v_dict[self.simv_v_name[i]].mass = degraded_vehi.mass

        # 使用传入的充电桩数量创建Station对象
        station_dict = {row.node_id: Station(row.node_id, idx,
                                             fast_charger=fast_chargers[idx] if idx < len(fast_chargers) else self.default_chargers,
                                             slow_charger=slow_chargers[idx] if idx < len(slow_chargers) else self.default_chargers,
                                             power_factor=power_factor)
                        for idx, row in self.cs_gdf.iterrows()}

        timeout_list = []
        # Start simulation
        e_trip_sum, e_d2s_sum, wait_time_sum, emission_sum = 0, 0, 0, 0  # Initialize 4 variables for storing simulation values
        for i in range(len(self.v_name)):
            sim_v_trip = SimVehicleTrip(self.v_name[i], self.trip[i], self.s_time[i], self.e_time[i],
                                        self.destination[i],
                                        self.distance[i], self.base_energy[i], self.avg_velocity[i])
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

        # 通过tree统计每个站点实际使用的快充和慢充桩数量（更准确的方法）
        charger_usage = {}  # {station_id: {'fast': set(), 'slow': set()}} 用于统计每个站点使用的充电桩

        # 遍历所有车辆，统计充电站使用情况
        for sim_v in self.sim_v_dict.values():
            # 遍历车辆的data_tree，统计充电信息
            for node in sim_v.data_tree.all_nodes():
                if node.data and 'charger_used' in node.data:
                    charger_info = node.data['charger_used']
                    # 从tag中提取站点ID，格式如"t1s12345"
                    if 's' in node.tag and node.tag != sim_v.v_name:  # 排除根节点
                        parts = node.tag.split('s')
                        if len(parts) >= 2:
                            try:
                                station_id = int(parts[-1])  # 获取站点ID

                                # 初始化站点统计信息
                                if station_id not in charger_usage:
                                    charger_usage[station_id] = {'fast': set(), 'slow': set()}

                                # 根据充电器名称统计快充或慢充
                                if charger_info.startswith('f'):
                                    charger_usage[station_id]['fast'].add(charger_info)
                                elif charger_info.startswith('s'):
                                    charger_usage[station_id]['slow'].add(charger_info)
                            except ValueError:
                                continue

        # 按照station_dict的顺序统计实际使用的快充和慢充桩数量
        actual_fast_charger_counts = []
        actual_slow_charger_counts = []

        for node_id, station in station_dict.items():
            fast_count = len(charger_usage.get(node_id, {}).get('fast', set()))
            slow_count = len(charger_usage.get(node_id, {}).get('slow', set()))
            actual_fast_charger_counts.append(fast_count)
            actual_slow_charger_counts.append(slow_count)


        if use_custom_extra:
            # 使用传入的额外车辆参数，跳过extra_timesave计算
            # 后面在计算vehicle_cost时直接使用这些值
            saved_time, subs_idx, extra_large, extra_medium, extra_small = extra_timesave(
                self.v_name, self.distance, self.base_energy, timeout_list, self.v_type,
                VehicleTypes(vars_[self.cs_num], v_type='large'),
                VehicleTypes(vars_[self.cs_num + 1], v_type='medium'),
                VehicleTypes(vars_[self.cs_num + 2], v_type='small'),
                extra_large, extra_medium, extra_small, degradation=degradation_factor
                )
        else:
            # 通过extra_timesave计算额外车辆数量
            saved_time, subs_idx, extra_large, extra_medium, extra_small = extra_timesave(
                self.v_name, self.distance, self.base_energy, timeout_list, self.v_type,
                VehicleTypes(vars_[self.cs_num], v_type='large'),
                VehicleTypes(vars_[self.cs_num + 1], v_type='medium'),
                VehicleTypes(vars_[self.cs_num + 2], v_type='small'),
                degradation=degradation_factor
                )
        timeout_sum = (sum(timeout_list) - saved_time)
        if is_test:
            return timeout_list, subs_idx

        # System costs
        # Vehicle cost
        vehicle_cost = ((self.large_num + extra_large) * VehicleTypes(vars_[self.cs_num], v_type='large').fix_cost +
                        (self.medium_num + extra_medium) * VehicleTypes(vars_[self.cs_num + 1],
                                                                        v_type='medium').fix_cost +
                        (self.small_num + extra_small) * VehicleTypes(vars_[self.cs_num + 2], v_type='small').fix_cost)

        # Station construction and maintenance costs per year
        station_cost = sum(vars_[0: self.cs_num]) * 600000 * 1
        station_emission = sum(vars_[0: self.cs_num]) * 80
        # 9w and 3w for fast and slow charger, respectively
        # Get chargers' num of selected stations and do calculation
        charger_cost = sum(actual_fast_charger_counts) * 4000 + sum(actual_slow_charger_counts) * 2000

        # 1.2 yuan/kwh
        # The life cycle cost of energy consumption in this trip， including both operation and go-charging distances
        trip_cost = (e_trip_sum + e_d2s_sum) * 365 * self.e_price * e_price_factor

        # print(f'Station Count: {sum(vars_[:self.cs_num])}; Extra Vehicles: {extra_large} {extra_medium} {extra_small}')

        '''one-objective'''
        # emission_cost = (emission_sum*365/1000+station_emission)*1.05  # kg * 1.05 yuan/kg social cost of emission
        # f0 = -(vehicle_cost + station_cost + charger_cost + trip_cost + emission_cost) / 1000000  # 1M yuan/year
        '''multi-objective'''
        f1 = -(vehicle_cost + station_cost + charger_cost + trip_cost) / 1000000  # 1M yuan/year
        f2 = -(emission_sum * 365 / 1000 + station_emission) / 1000  # T/year

        cv1 = sim_utils.set_cv1(self.cs_num, vars_)  # cv是约束,eq22(ub,lb是eq25-26)

        cv_and_params = ([cv1] + [-x for x in actual_fast_charger_counts] + [-x for x in actual_slow_charger_counts] +
                         [trip_cost/1000000, timeout_sum] + [-extra_large, -extra_medium, -extra_small])
        cal_e_time = time.perf_counter()
        # print(f'Single calculation time: {(cal_e_time - cal_s_time):.2f}s')

        # print(f"f1: {-f1:.1f} f2: {-f2:.1f}")

        return np.array([f1, f2]), np.hstack(cv_and_params)


def process_single_city(city_i, cities_df, is_simplified=False, is_referred=False,
                        is_dict_loaded=True, is_tested=True, SIM_N=50, what_if=False, min_delta_hv=0.01):
    """
    处理单个城市的函数，保持与原main函数相同的逻辑
    """
    city_name = cities_df['city'][city_i]
    e_price = cities_df['eprice'][city_i]

    try:
        print(f"开始处理城市: {city_name}")

        if is_simplified:
            vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid_simplified/{city_name}.csv')  # 车辆行程
        else:
            vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')  # 车辆行程

        vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
        vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
        vs_parking_df.sort_values(['e_time'], inplace=True, ignore_index=True)

        cs_gdf = gpd.read_file(rf'../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4547))  # 充电站
        cs_gdf['lon'] = cs_gdf['geometry'].x
        cs_gdf['lat'] = cs_gdf['geometry'].y

        sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').  # 将车辆行程按车辆集计
                                            apply(lambda x: data_utils.derive_simV_info(x)).to_dict(), orient='index')
        sim_v_dict = {}
        for idx, row in sim_v_info.iterrows():
            sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance,
                                         row.base_energy, row.avg_velocity)

        if is_dict_loaded:
            with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'rb') as f:
                all_d2s_dict = pickle.load(f)
        else:
            nodes_sim = gpd.read_file(
                rf'../data/road/{city_name}/nodes_sim.shp', crs=CRS.from_epsg(4547))  # 道路节点
            edges_sim = gpd.read_file(
                rf'../data/road/{city_name}/edges_sim.shp', crs=CRS.from_epsg(4547))  # 道路路段
            G = nx.from_pandas_edgelist(df=edges_sim, source='u', target='v', edge_attr=['edge_id', 'length'],
                                        create_using=nx.Graph())
            all_d2s_dict = data_utils.get_d2s_realdict(vs_parking_df, cs_gdf, nodes_sim, G,
                                                       near_n=100, sim_n=SIM_N, distance_limit=100000,
                                                       is_projected=True)
            with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'wb') as f:
                pickle.dump(all_d2s_dict, f)

        num_vars = data_utils.set_var_num(cs_gdf)  # sim_n是后加的，选最近3个station为备选
        print("num_vars: ", num_vars)

        cs_dict = {v: k for k, v in zip(cs_gdf.index, cs_gdf['node_id'])}
        print('read files successfully!')

        problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, sim_v_dict=sim_v_dict, cs_gdf=cs_gdf,
                           vs_parking_df=vs_parking_df,
                           all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, e_price=e_price)

        annealing_iters = 100  # 100
        algorithm = MOSA(problem, annealing_iters=annealing_iters)
        # algorithm.end_temperature = 1  # 900
        algorithm.annealing_strength = 0.6
        algorithm.cooling_alpha = 0.9953
        algorithm.min_delta_hv = min_delta_hv
        algorithm.multiprocess = False
        algorithm.what_if = what_if
        algorithm.is_tested = is_tested

        # # estimate how long would be processed
        print(f"Estimated: {math.log(algorithm.end_temperature / algorithm.initial_temperature) /
                            math.log(algorithm.cooling_alpha) * annealing_iters * 2 / 3600:.1f} h")  # 10s is estimated circle time

        algorithm.early_termination = {'max_iters': 500000, 'max_duration': 240,
                                       'max_no_eliminated': 20000}  # max_duration:运行时长，按小时计
        if is_referred:
            refer_vars = pd.read_csv(
                rf"../data/output/mosa/250630/{city_name}_250630/archive_vars.csv").values[:, 1:]
        else:
            refer_vars = sim_utils.set_refer_vars(len(cs_gdf), refer_num=20)  # 生成初始解

        algorithm.load_refer_solutions(refer_vars)
        save_path = rf"../data/output/mosa/251010/{city_name}_251010"
        # 检查save_path是否已存在且不为空，如果满足条件则跳过该城市
        if os.path.exists(save_path) and os.listdir(save_path):
            print(f"城市 {city_name} 的结果已存在且不为空，跳过处理")
            return f"城市 {city_name} 已存在结果，跳过处理"

        # 原有的创建目录逻辑
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # 如果不存在，则创建文件夹
        algorithm.run(inf='infeasible', is_cv=True, path=save_path, store='store')  # ,multi_tasks=4

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

        print(f"城市 {city_name} 处理完成")
        return f"城市 {city_name} 处理成功"

    except Exception as e:
        print(f"处理城市 {city_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"城市 {city_name} 处理失败: {str(e)}"


def main(cities_file='../data/18cities.csv', processes=1, city_indices=None,
         is_simplified=False, is_referred=False, is_dict_loaded=True,
         is_tested=True, SIM_N=50, what_if=False, min_delta_hv=0.01):
    """
    主函数，支持多城市并行处理和参数调整

    参数:
    cities_file: 城市列表文件路径
    processes: 并行进程数，1表示单进程，>1表示多进程
    city_indices: 要处理的城市索引列表，None表示处理所有城市
    is_simplified: 是否使用简化数据
    is_referred: 是否使用参考解
    is_dict_loaded: 是否加载预处理字典
    is_tested: 是否进行测试
    SIM_N: SIM_N参数
    what_if: 是否启用what_if模式
    min_delta_hv: min_delta_hv参数
    """

    # 读取城市列表
    cities = pd.read_csv(cities_file)

    # 确定要处理的城市索引
    if city_indices is None:
        city_indices = list(range(len(cities)))

    print(f"准备处理 {len(city_indices)} 个城市")
    print(f"使用进程数: {processes}")
    print(f"参数设置: simplified={is_simplified}, referred={is_referred}, dict_loaded={is_dict_loaded}")
    print(f"          tested={is_tested}, SIM_N={SIM_N}, what_if={what_if}, min_delta_hv={min_delta_hv}")

    if processes == 1:
        # 单进程处理
        results = []
        for city_i in city_indices:
            result = process_single_city(
                city_i, cities,
                is_simplified=is_simplified,
                is_referred=is_referred,
                is_dict_loaded=is_dict_loaded,
                is_tested=is_tested,
                SIM_N=SIM_N,
                what_if=what_if,
                min_delta_hv=min_delta_hv
            )
            results.append(result)
    else:
        # 多进程处理
        # 创建部分函数，固定一些参数
        process_func = partial(
            process_single_city,
            cities_df=cities,
            is_simplified=is_simplified,
            is_referred=is_referred,
            is_dict_loaded=is_dict_loaded,
            is_tested=is_tested,
            SIM_N=SIM_N,
            what_if=what_if,
            min_delta_hv=min_delta_hv
        )

        # 使用多进程池并行处理
        num_processes = min(processes, len(city_indices))
        print(f"使用 {num_processes} 个进程处理 {len(city_indices)} 个城市")

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_func, city_indices)

    # 输出结果统计
    success_count = sum(1 for result in results if "处理成功" in result)
    fail_count = len(results) - success_count

    print(f"\n处理完成:")
    print(f"成功: {success_count} 个城市")
    print(f"失败: {fail_count} 个城市")

    # 打印失败的城市
    if fail_count > 0:
        print("失败的城市:")
        for result in results:
            if "处理失败" in result:
                print(result)

    return results


# 原有的单城市处理逻辑保持不变
if __name__ == '__main__':
    main(
        cities_file='../data/224cities_test.csv',  # '../data/18cities.csv' '../data/224cities.csv'
        processes=10,
        is_simplified=True,
        is_referred=False,
        is_tested=False
    )
