import os
import sys
import multiprocessing as mp
from functools import partial
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import pickle
from dateutil import parser
from test4plot2 import clearing
from analysis import find_knee

# 添加上级目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from test4mosa import Location
from simulation import SimVehicle
import data_utils


os.environ['GDAL_DATA'] = r'/usr/share/gdal'

root_name = '250630'
root_dir = rf"../../data/output/mosa/{root_name}"
cities = pd.read_csv(rf'../../data/18cities.csv')

# 定义测试参数组合
test_params = {
    'power_factor': [0.6, 0.7, 0.8, 0.9, 1.0],
    'e_price_factor': [0.5, 0.75, 1.0, 1.25, 1.5],
    'degradation_factor': [0.6, 0.7, 0.8, 0.9, 1.0]
}


def process_city(city_name):
    """处理单个城市的数据"""
    try:
        print(f"Processing city: {city_name}")

        # 构建路径
        city_dir = os.path.join(root_dir, f"{city_name}_{root_name}")

        # 读取数据
        cs_gdf = gpd.read_file(rf'../../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4507))  # 充电站

        # 去重后的数据
        cs_gdf_unique = cs_gdf.drop_duplicates(subset=['node_id'], keep='first')
        cs_num_original = len(cs_gdf)  # 原始站点数
        cs_num_unique = len(cs_gdf_unique)  # 去重后站点数

        OBJ_G = pd.read_csv(rf"{city_dir}/archive_objs.csv")
        VAR_G = pd.read_csv(rf"{city_dir}/archive_vars.csv")
        CV_G = pd.read_csv(rf"{city_dir}/archive_cvs.csv")
        OBJ_G = clearing(OBJ_G)

        knee_no_G = find_knee(OBJ_G)

        # 找到knee point对应的vars并计算cost和排放
        knee_solution = VAR_G[VAR_G.iloc[:, 0] == knee_no_G]  # 第一列是solution_no
        if len(knee_solution) > 0:
            # 获取除第一列(solution_no)外的所有变量值
            knee_vars = knee_solution.iloc[0, 1:].values

            # 从CV_G中提取fast和slow充电桩数量（取负数为正）
            knee_cv = CV_G[CV_G.iloc[:, 0] == knee_no_G]

            # 处理重复node_id的情况
            if cs_num_original != cs_num_unique:
                print(f"发现重复node_id: 原始{cs_num_original}个站点，去重后{cs_num_unique}个站点")

                # 创建node_id到索引的映射，保留每个node_id第一次出现的索引
                node_id_to_index = {}
                for idx, row in cs_gdf.iterrows():
                    node_id = row['node_id']
                    if node_id not in node_id_to_index:
                        node_id_to_index[node_id] = idx

                # 创建完整的fast_chargers和slow_chargers列表，初始化为0
                fast_chargers = [0] * cs_num_original
                slow_chargers = [0] * cs_num_original

                # 从CV中提取实际值
                fast_values = [-x for x in knee_cv.iloc[0, 2:cs_num_unique + 2].tolist()]
                slow_values = [-x for x in knee_cv.iloc[0, cs_num_unique + 2:2 * cs_num_unique + 2].tolist()]

                # 为每个站点分配对应的充电桩数量
                unique_node_ids = cs_gdf_unique['node_id'].tolist()
                for idx, row in cs_gdf.iterrows():
                    node_id = row['node_id']
                    # 找到该node_id在去重后的数据中的索引
                    if node_id in unique_node_ids:
                        unique_index = unique_node_ids.index(node_id)
                        fast_chargers[idx] = fast_values[unique_index]
                        slow_chargers[idx] = slow_values[unique_index]

            else:
                # 没有重复node_id的情况
                fast_chargers = [-x for x in knee_cv.iloc[0, 2:cs_num_unique + 2].tolist()]
                slow_chargers = [-x for x in knee_cv.iloc[0, cs_num_unique + 2:2 * cs_num_unique + 2].tolist()]

            # 加载计算cost和排放所需的数据
            # 读取必要数据
            vs_parking_df = pd.read_csv(rf"../../data/input/vs_parking_nodeid//{city_name}.csv")
            vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
            vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
            vs_parking_df.sort_values(['e_time'], inplace=True, ignore_index=True)

            cs_gdf['lon'] = cs_gdf['geometry'].x
            cs_gdf['lat'] = cs_gdf['geometry'].y

            sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').
                                                apply(lambda x: data_utils.derive_simV_info(x)).to_dict(),
                                                orient='index')
            sim_v_dict = {}
            for idx, row in sim_v_info.iterrows():
                sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance,
                                             row.avg_velocity)

            with open(rf"../../data/input/all_d2s_dict/{city_name}.pkl", 'rb') as f:
                all_d2s_dict = pickle.load(f)

            cs_dict = {v: k for k, v in zip(cs_gdf.index, cs_gdf['node_id'])}

            # 创建问题实例
            city_row = cities[cities['city'] == city_name].iloc[0]

            problem = Location(num_vars=len(knee_vars), sim_v_info=sim_v_info, sim_v_dict=sim_v_dict, cs_gdf=cs_gdf,
                               vs_parking_df=vs_parking_df, all_d2s_dict=all_d2s_dict, cs_dict=cs_dict,
                               e_price=city_row['eprice'])

            # 计算cost和排放，传入充电桩数量参数
            extra_large = -int(knee_cv.iloc[0, -3])  # 原为负数，需要转为正数
            extra_medium = -int(knee_cv.iloc[0, -2])
            extra_small = -int(knee_cv.iloc[0, -1])

            city_stats = []

            # 基准测试（默认参数）
            f_values, _ = problem.eval_vars(knee_vars,
                                            fast_chargers=fast_chargers,
                                            slow_chargers=slow_chargers,
                                            extra_large=abs(extra_large),
                                            extra_medium=abs(extra_medium),
                                            extra_small=abs(extra_small))
            f1, f2 = f_values  # f1是cost，f2是emission
            tripcost = _[-5]
            timeout = _[-4]

            # 计算基准情况下的充电桩利用率
            baseline_utilization = calculate_charger_utilization(
                tripcost, city_row['eprice'], fast_chargers, slow_chargers, 1.0)

            # 存储基准结果
            city_stats.append({
                'city_name': city_name,
                'indicator': 'cost',
                'what_if': 'baseline',
                'value': -f1  # 转换回正值
            })
            city_stats.append({
                'city_name': city_name,
                'indicator': 'emission',
                'what_if': 'baseline',
                'value': -f2  # 转换回正值
            })
            city_stats.append({
                'city_name': city_name,
                'indicator': 'timedelay',
                'what_if': 'baseline',
                'value': timeout
            })
            city_stats.append({
                'city_name': city_name,
                'indicator': 'tripcost',
                'what_if': 'baseline',
                'value': tripcost
            })
            city_stats.append({
                'city_name': city_name,
                'indicator': 'utilization',
                'what_if': 'baseline',
                'value': baseline_utilization
            })

            print(
                f"{city_name} baseline: cost={-f1:.2f}, emission={-f2:.2f}, timedelay={timeout:.2f}, "
                f"tripcost={tripcost:.2f}, utilization={baseline_utilization:.4f}")

            # 测试不同的power_factor值
            for pf in test_params['power_factor']:
                f_values, _ = problem.eval_vars(knee_vars,
                                                fast_chargers=fast_chargers,
                                                slow_chargers=slow_chargers,
                                                extra_large=abs(extra_large),
                                                extra_medium=abs(extra_medium),
                                                extra_small=abs(extra_small),
                                                power_factor=pf)
                f1, f2 = f_values
                tripcost = _[-5]
                timeout = _[-4]

                # 计算基准情况下的充电桩利用率
                current_utilization = calculate_charger_utilization(
                    tripcost, city_row['eprice'], fast_chargers, slow_chargers, pf)

                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'cost',
                    'what_if': f'power_factor_{pf}',
                    'value': -f1
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'emission',
                    'what_if': f'power_factor_{pf}',
                    'value': -f2
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'timedelay',
                    'what_if': f'power_factor_{pf}',
                    'value': timeout
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'tripcost',
                    'what_if': f'power_factor_{pf}',
                    'value': tripcost
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'utilization',
                    'what_if': f'power_factor_{pf}',
                    'value': current_utilization
                })

                print(f"{city_name} power_factor={pf}: cost={-f1:.2f}, emission={-f2:.2f}, timedelay={timeout:.2f}, "
                      f"tripcost={tripcost:.2f}, utilization={current_utilization:.4f}")

            # 测试不同的e_price_factor值
            for epf in test_params['e_price_factor']:
                f_values, _ = problem.eval_vars(knee_vars,
                                                fast_chargers=fast_chargers,
                                                slow_chargers=slow_chargers,
                                                extra_large=abs(extra_large),
                                                extra_medium=abs(extra_medium),
                                                extra_small=abs(extra_small),
                                                e_price_factor=epf)
                f1, f2 = f_values
                tripcost = _[-5]
                timeout = _[-4]

                # 计算基准情况下的充电桩利用率
                current_utilization = calculate_charger_utilization(
                    tripcost, city_row['eprice']*epf, fast_chargers, slow_chargers, 1)

                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'cost',
                    'what_if': f'e_price_factor_{epf}',
                    'value': -f1
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'emission',
                    'what_if': f'e_price_factor_{epf}',
                    'value': -f2
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'timedelay',
                    'what_if': f'e_price_factor_{epf}',
                    'value': timeout
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'tripcost',
                    'what_if': f'e_price_factor_{epf}',
                    'value': tripcost
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'utilization',
                    'what_if': f'e_price_factor_{epf}',
                    'value': current_utilization
                })

                print(f"{city_name} e_price_factor={epf}: cost={-f1:.2f}, emission={-f2:.2f}, timedelay={timeout:.2f}, "
                      f"tripcost={tripcost:.2f}, utilization={current_utilization:.4f}")

            # 测试不同的degradation_factor值
            for df in test_params['degradation_factor']:
                f_values, _ = problem.eval_vars(knee_vars,
                                                fast_chargers=fast_chargers,
                                                slow_chargers=slow_chargers,
                                                extra_large=abs(extra_large),
                                                extra_medium=abs(extra_medium),
                                                extra_small=abs(extra_small),
                                                degradation_factor=df)
                f1, f2 = f_values
                tripcost = _[-5]
                timeout = _[-4]

                # 计算基准情况下的充电桩利用率
                current_utilization = calculate_charger_utilization(
                    tripcost, city_row['eprice'], fast_chargers, slow_chargers, 1)

                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'cost',
                    'what_if': f'degradation_factor_{df}',
                    'value': -f1
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'emission',
                    'what_if': f'degradation_factor_{df}',
                    'value': -f2
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'timedelay',
                    'what_if': f'degradation_factor_{df}',
                    'value': timeout
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'tripcost',
                    'what_if': f'degradation_factor_{df}',
                    'value': tripcost
                })
                city_stats.append({
                    'city_name': city_name,
                    'indicator': 'utilization',
                    'what_if': f'degradation_factor_{df}',
                    'value': current_utilization
                })

                print(
                    f"{city_name} degradation_factor={df}: cost={-f1:.2f}, emission={-f2:.2f}, timedelay={timeout:.2f}, "
                    f"tripcost={tripcost:.2f}, utilization={current_utilization:.4f}")

            return city_stats
        return []
    except Exception as e:
        print(f"Error processing city {city_name}: {e}")
        return []


def calculate_total_power(fast_chargers, slow_chargers, power_factor):
    """
    计算一天24小时内fast和slow充电桩的总kWh输出

    参数:
    fast_chargers: fast充电桩数量列表
    slow_chargers: slow充电桩数量列表
    power_factor: fast充电桩的功率因子

    返回:
    total_fast_kwh: fast充电桩总kWh
    total_slow_kwh: slow充电桩总kWh
    total_kwh: 总kWh
    """

    # 每个充电桩的功率(kW)
    fast_power_per_unit = 100  # kW
    slow_power_per_unit = 50  # kW

    # 计算fast充电桩总功率
    total_fast_chargers = sum(fast_chargers)
    total_fast_power = total_fast_chargers * fast_power_per_unit * power_factor

    # 计算slow充电桩总功率
    total_slow_chargers = sum(slow_chargers)
    total_slow_power = total_slow_chargers * slow_power_per_unit

    # 24小时总kWh
    total_fast_kwh = total_fast_power * 24
    total_slow_kwh = total_slow_power * 24
    total_kwh = total_fast_kwh + total_slow_kwh

    return total_fast_kwh, total_slow_kwh, total_kwh


def calculate_charger_utilization(tripcost, e_price, fast_chargers, slow_chargers, power_factor):
    """
    计算充电桩利用率

    参数:
    city_name: 城市名称
    tripcost: 出行成本
    e_price: 电价
    fast_chargers: fast充电桩数量列表
    slow_chargers: slow充电桩数量列表
    power_factor: fast充电桩的功率因子

    返回:
    utilization: 充电桩利用率
    """
    # 计算一天内耗能 (kWh)
    daily_energy_consumption = tripcost * 1000000 / 365 / e_price

    # 计算一天内charger最多提供的能量 (kWh)
    _, _, max_daily_energy = calculate_total_power(fast_chargers, slow_chargers, power_factor)

    # 计算利用率
    if max_daily_energy > 0:
        utilization = daily_energy_consumption / max_daily_energy
    else:
        utilization = 0

    return utilization


# 获取所有需要处理的城市
city_names = []
for name in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, name)) and name.endswith(f'_{root_name}'):
        city_name = name[:-len(f'_{root_name}')]
        city_names.append(city_name)

# 按照18cities.csv中的顺序排序
city_order = cities['city'].tolist()
city_names = [city for city in city_order if city in city_names]

print(f"Processing cities in order: {city_names}")

# 使用多进程处理
if __name__ == '__main__':
    with mp.Pool(processes=min(len(city_names), mp.cpu_count())) as pool:
        results = pool.map(process_city, city_names)

    # 合并所有结果
    Indicator_STATS = []
    for result in results:
        Indicator_STATS.extend(result)

    # 创建指标统计表格
    Indicator_DF = pd.DataFrame(Indicator_STATS)

    # 重新整理数据格式，行是城市，列是指标和what_if组合
    pivot_df = Indicator_DF.pivot_table(index='city_name', columns=['indicator', 'what_if'], values='value')

    # 按照18cities.csv中的城市顺序排序
    pivot_df = pivot_df.reindex(city_order)

    print("指标统计表格:")
    print(pivot_df)

    # 保存结果到CSV文件
    # pivot_df.to_csv("../../data/output/mosa/what_if_results_250630.csv", index=False)

    print("Results saved to CSV files")
