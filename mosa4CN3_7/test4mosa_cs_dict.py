import multiprocessing as mp
import pandas as pd
import geopandas as gpd
import pickle
import os
from pyproj import CRS
import networkx as nx
import data_utils


def generate_all_d2s_dict_for_city(city_name, sim_n=50):
    """
    为单个城市生成all_d2s_dict

    Parameters:
    city_name (str): 城市名称
    sim_n (int): 模拟次数
    """
    try:
        print(f"开始处理城市: {city_name}")

        # 读取车辆行程数据
        vs_parking_df = pd.read_csv(f'../data/input/vs_parking_nodeid/{city_name}.csv')

        # 读取充电站数据
        cs_gdf = gpd.read_file(f'../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4547))

        # 读取路网数据
        nodes_sim = gpd.read_file(f'../data/road/{city_name}/nodes_sim.shp', crs=CRS.from_epsg(4547))
        edges_sim = gpd.read_file(f'../data/road/{city_name}/edges_sim.shp', crs=CRS.from_epsg(4547))

        # 构建图
        G = nx.from_pandas_edgelist(df=edges_sim, source='u', target='v',
                                    edge_attr=['edge_id', 'length'], create_using=nx.Graph())

        # 生成all_d2s_dict
        all_d2s_dict = data_utils.get_d2s_realdict(
            vs_parking_df, cs_gdf, nodes_sim, G,
            near_n=150, sim_n=sim_n, distance_limit=100000.0, is_projected=True
        )

        # 保存结果
        output_dir = f"../data/input/all_d2s_dict"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"{output_dir}/{city_name}.pkl", 'wb') as f:
            pickle.dump(all_d2s_dict, f)

        print(f"城市 {city_name} 处理完成")
        return f"城市 {city_name} 处理成功"

    except Exception as e:
        print(f"处理城市 {city_name} 时出错: {str(e)}")
        return f"城市 {city_name} 处理失败: {str(e)}"


def batch_generate_all_d2s_dict(city_list, sim_n=50, num_processes=None):
    """
    批量多进程生成all_d2s_dict

    Parameters:
    city_list (list): 城市名称列表
    sim_n (int): 模拟次数
    num_processes (int): 进程数，默认为CPU核心数
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"使用 {num_processes} 个进程处理 {len(city_list)} 个城市")

    # 创建进程池
    with mp.Pool(processes=num_processes) as pool:
        # 使用进程池处理所有城市
        results = pool.starmap(generate_all_d2s_dict_for_city,
                               [(city, sim_n) for city in city_list])

    # 输出结果统计
    success_count = sum(1 for result in results if "处理成功" in result)
    fail_count = len(results) - success_count

    print(f"\n处理完成:")
    print(f"成功: {success_count} 个城市")
    print(f"失败: {fail_count} 个城市")

    return results


def get_cities_without_all_d2s_dict(all_cities_list):
    """
    获取还没有生成all_d2s_dict的城市列表

    Parameters:
    all_cities_list (list): 所有城市名称列表

    Returns:
    list: 尚未生成all_d2s_dict的城市列表
    """
    all_d2s_dir = "../data/input/all_d2s_dict"

    if not os.path.exists(all_d2s_dir):
        return all_cities_list

    existing_files = os.listdir(all_d2s_dir)
    existing_cities = [f.replace('.pkl', '') for f in existing_files if f.endswith('.pkl')]

    missing_cities = [city for city in all_cities_list if city not in existing_cities]
    return missing_cities


# 使用示例
if __name__ == '__main__':
    # 读取所有城市列表
    cities_df = pd.read_csv(rf'../data/224cities.csv')  # rf'../data/224cities.csv'
    all_cities = cities_df['city'].tolist()

    # 获取尚未生成all_d2s_dict的城市
    cities_to_process = get_cities_without_all_d2s_dict(all_cities)

    if cities_to_process:
        print(f"需要处理的城市: {cities_to_process}")
        # 批量生成
        results = batch_generate_all_d2s_dict(cities_to_process, sim_n=80, num_processes=16)
    else:
        print("所有城市的all_d2s_dict均已存在")
