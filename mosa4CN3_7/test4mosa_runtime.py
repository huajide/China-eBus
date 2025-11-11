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


def balance_cities_into_groups(all_cities_df, n_groups):
    """
    将城市按照平均运行时间分成N组，使得每组的总时间尽可能接近

    Parameters:
    all_cities_df: 包含城市和avg_eval_time的DataFrame
    n_groups: 分组数量

    Returns:
    DataFrame: 添加了group列的DataFrame
    """
    # 复制数据以避免修改原始数据
    df = all_cities_df.copy()

    # 移除avg_eval_time为空的城市
    df = df.dropna(subset=['avg_eval_time'])

    # 按照avg_eval_time降序排列
    df = df.sort_values('avg_eval_time', ascending=False).reset_index(drop=True)

    # 初始化组和组的总时间
    groups = [[] for _ in range(n_groups)]
    group_times = [0 for _ in range(n_groups)]

    # 使用贪心算法分配城市到各组
    for idx, row in df.iterrows():
        # 找到当前总时间最小的组
        min_group_idx = group_times.index(min(group_times))

        # 将城市分配给该组
        groups[min_group_idx].append(row['city'])
        group_times[min_group_idx] += row['avg_eval_time']

    # 创建城市到组的映射
    city_to_group = {}
    for group_idx, cities in enumerate(groups):
        for city in cities:
            city_to_group[city] = group_idx

    # 添加分组列
    df['group'] = df['city'].map(city_to_group)

    # 显示各组的总时间
    print("各组总运行时间:")
    for i in range(n_groups):
        group_total = df[df['group'] == i]['avg_eval_time'].sum()
        print(f"  组 {i}: {group_total:.2f} 秒")

    return df


def test_city_performance(city_name, cities_df, city_index):
    """
    测试单个城市运行一次eval_vars的性能
    """
    try:
        # 获取电价
        e_price = cities_df['eprice'][city_index]

        # 加载vs_parking_df数据
        vs_parking_df_path = rf'../data/input/vs_parking_nodeid/{city_name}.csv'
        if not os.path.exists(vs_parking_df_path):
            print(f"文件不存在: {vs_parking_df_path}")
            return None

        vs_parking_df = pd.read_csv(vs_parking_df_path)
        vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
        vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
        vs_parking_df.sort_values(['e_time'], inplace=True, ignore_index=True)

        # 加载充电站数据
        cs_gdf_path = rf'../data/input/cs_gdf/{city_name}.shp'
        if not os.path.exists(cs_gdf_path):
            print(f"文件不存在: {cs_gdf_path}")
            return None

        cs_gdf = gpd.read_file(cs_gdf_path, crs=CRS.from_epsg(4507))
        cs_gdf['lon'] = cs_gdf['geometry'].x
        cs_gdf['lat'] = cs_gdf['geometry'].y

        # 简化数据（如果需要）
        vs_parking_df = data_utils.simplify_vs_df(vs_parking_df, min(min(driving_range)))

        # 创建sim_v_info和sim_v_dict
        sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').
                                            apply(lambda x: data_utils.derive_simV_info(x)).to_dict(), orient='index')
        sim_v_dict = {}
        for idx, row in sim_v_info.iterrows():
            sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination,
                                         row.distance, row.avg_velocity)

        # 加载all_d2s_dict
        all_d2s_dict_path = rf"../data/input/all_d2s_dict/{city_name}.pkl"
        if not os.path.exists(all_d2s_dict_path):
            print(f"文件不存在: {all_d2s_dict_path}")
            return None

        with open(all_d2s_dict_path, 'rb') as f:
            all_d2s_dict = pickle.load(f)

        # 设置变量数量
        num_vars = data_utils.set_var_num(cs_gdf)

        # 创建cs_dict
        cs_dict = {v: k for k, v in zip(cs_gdf.index, cs_gdf['node_id'])}

        # 创建Location实例
        problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, sim_v_dict=sim_v_dict,
                           cs_gdf=cs_gdf, vs_parking_df=vs_parking_df,
                           all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, e_price=e_price)

        # 创建测试变量（这里使用简单的参考变量）
        refer_vars = sim_utils.set_refer_vars(len(cs_gdf), refer_num=1)[0]  # 取第一个参考解

        # 运行三次取平均
        times = []
        for i in range(3):
            start_time = time.perf_counter()
            try:
                problem.eval_vars(refer_vars)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"运行第{i + 1}次时出错: {e}")
                continue

        if times:
            avg_time = sum(times) / len(times)
            print(f"{city_name} 平均运行时间: {avg_time:.2f}秒")
            return avg_time
        else:
            print(f"{city_name} 无法完成任何一次测试")
            return None

    except Exception as e:
        print(f"测试城市 {city_name} 时出错: {e}")
        return None


# 注意：Location类需要从test4mosa.py复制过来，这里为了完整性重新定义
class Location(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self,
                         kwargs.get('num_vars'),  # var_num
                         [0] * kwargs.get('num_vars'),  # Integer or Real
                         [0] * len(kwargs.get('cs_gdf')) + [0, 0, 0],  # lb
                         [1] * len(kwargs.get('cs_gdf')) + [9999, 9999, 9999],  # ub
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
        # 简化版本的eval_vars，只关注执行时间
        cal_s_time = time.perf_counter()

        # 实例化所有sim_v和stations以及simVTrip
        for i in range(len(self.simv_v_name)):
            if self.simv_v_type[i] == 'large':
                vehi_type = vars_[self.cs_num]
            elif self.simv_v_type[i] == 'medium':
                vehi_type = vars_[self.cs_num + 1]
            else:
                vehi_type = vars_[self.cs_num + 2]
            vehi = VehicleTypes(vehi_type, v_type=self.simv_v_type[i])
            self.sim_v_dict[self.simv_v_name[i]].model = vehi

        station_dict = {row.node_id: Station(row.node_id, idx, fast_charger=20,
                                             slow_charger=20) for idx, row in self.cs_gdf.iterrows()}

        timeout_list = []
        # 开始模拟
        e_trip_sum, e_d2s_sum, wait_time_sum, emission_sum = 0, 0, 0, 0
        for i in range(len(self.v_name)):
            sim_v_trip = SimVehicleTrip(self.v_name[i], self.trip[i], self.s_time[i], self.e_time[i],
                                        self.destination[i],
                                        self.distance[i], self.sim_v_dict.get(self.v_name[i]).driving_range,
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

        # 添加额外车辆
        saved_time, subs_idx, extra_large, extra_medium, extra_small = extra_timesave(self.v_name, self.distance,
                                                                                      timeout_list, self.v_type,
                                                                                      VehicleTypes(vars_[self.cs_num],
                                                                                                   v_type='large').driving_range,
                                                                                      VehicleTypes(
                                                                                          vars_[self.cs_num + 1],
                                                                                          v_type='medium').driving_range,
                                                                                      VehicleTypes(
                                                                                          vars_[self.cs_num + 2],
                                                                                          v_type='small').driving_range)

        cal_e_time = time.perf_counter()

        if "CV" in args:
            cv1 = sim_utils.set_cv1(self.cs_num, vars_)
            return np.hstack([cv1])

        return np.array([0, 0]), np.hstack([0])  # 简化返回值



# ... 原有代码保持不变 ...

if __name__ == '__main__':
    # 原有主程序代码保持不变，添加以下性能测试代码

    # 性能测试：测试18个城市的eval_vars平均运行时间
    print("\n开始性能测试...")

    # 读取所有城市数据
    path_name = rf'../data/18cities.csv'  # or:rf'../data/18cities.csv'
    all_cities = pd.read_csv(path_name)

    # 为每个城市添加平均运行时间列（如果不存在）
    if 'avg_eval_time' not in all_cities.columns:
        all_cities['avg_eval_time'] = np.nan

    # 测试所有城市的运行时间
    for city_idx in range(len(all_cities)):
        city_name = all_cities['city'][city_idx]
        print(f"测试城市 {city_name} ({city_idx + 1}/{len(all_cities)})")

        try:
            # 加载该城市的数据
            city_vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')
            city_vs_parking_df['s_time'] = city_vs_parking_df['s_time'].apply(parser.parse)
            city_vs_parking_df['e_time'] = city_vs_parking_df['e_time'].apply(parser.parse)
            city_vs_parking_df.sort_values(['e_time'], inplace=True, ignore_index=True)

            city_cs_gdf = gpd.read_file(rf'../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4507))

            # 简化数据
            # city_vs_parking_df = data_utils.simplify_vs_df(city_vs_parking_df, min(min(driving_range)))

            # 创建车辆信息
            city_sim_v_info = pd.DataFrame.from_dict(city_vs_parking_df.groupby('v_name').
                                                     apply(lambda x: data_utils.derive_simV_info(x)).to_dict(),
                                                     orient='index')
            city_sim_v_dict = {}
            for idx, row in city_sim_v_info.iterrows():
                city_sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination,
                                                  row.distance, row.avg_velocity)

            try:
                # 加载距离字典
                with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'rb') as f:
                    city_all_d2s_dict = pickle.load(f)
            except:
                nodes_sim = gpd.read_file(
                    rf'../data/road/{city_name}/nodes_sim.shp', crs=CRS.from_epsg(4507))  # 道路节点
                edges_sim = gpd.read_file(
                    rf'../data/road/{city_name}/edges_sim.shp', crs=CRS.from_epsg(4507))  # 道路路段
                G = nx.from_pandas_edgelist(df=edges_sim, source='u', target='v', edge_attr=['edge_id', 'length'],
                                            create_using=nx.Graph())
                city_all_d2s_dict = data_utils.get_d2s_realdict(city_vs_parking_df, city_cs_gdf, nodes_sim, G,
                                                           near_n=100, sim_n=50, distance_limit=10000.0,
                                                           is_projected=True)
                with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'wb') as f:
                    pickle.dump(city_all_d2s_dict, f)

            # 设置变量数量
            city_num_vars = data_utils.set_var_num(city_cs_gdf)
            city_cs_dict = {v: k for k, v in zip(city_cs_gdf.index, city_cs_gdf['node_id'])}

            # 创建问题实例
            city_problem = Location(num_vars=city_num_vars, sim_v_info=city_sim_v_info,
                                    sim_v_dict=city_sim_v_dict, cs_gdf=city_cs_gdf,
                                    vs_parking_df=city_vs_parking_df,
                                    all_d2s_dict=city_all_d2s_dict, cs_dict=city_cs_dict,
                                    e_price=all_cities['eprice'][city_idx])

            # 生成三个参考变量
            city_refer_vars = sim_utils.set_refer_vars(len(city_cs_gdf), refer_num=3)

            # 对每个参考变量都运行三次取平均，然后计算所有结果的平均值
            times = []
            for ref_var in city_refer_vars:  # 对每个参考变量
                for i in range(3):  # 每个参考变量运行3次
                    start_time = time.perf_counter()
                    try:
                        city_problem.eval_vars(ref_var)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"  运行参考变量时出错: {e}")
                        continue

            # 计算所有测试的平均时间
            if times:
                avg_time = sum(times) / len(times)
                all_cities.loc[city_idx, 'avg_eval_time'] = avg_time
                print(f"  {city_name} 平均运行时间: {avg_time:.2f} 秒")
            else:
                print(f"  {city_name} 无法完成测试")

        except Exception as e:
            print(f"  测试城市 {city_name} 时出错: {e}")
            continue

    # 显示结果
    print("\n所有城市性能测试结果:")
    print(all_cities[['city', 'avg_eval_time']])

    # 将城市分为3组
    grouped_cities = balance_cities_into_groups(all_cities, 3)

    # 保存更新后的数据到原文件
    # all_cities.to_csv(path_name, index=False)
