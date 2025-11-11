import multiprocessing as mp
from functools import partial
import os
import pandas as pd
from vehicle_type import battery_capacity
import data_utils
from dateutil import parser


def process_single_city(city_name, cities_file='../data/18cities.csv',
                        input_dir='../data/input/vs_parking_nodeid',
                        output_dir='../data/input/vs_parking_nodeid_simplified'):
    """
    处理单个城市的数据简化

    Parameters:
    city_name: 城市名称
    cities_file: 城市列表文件路径
    input_dir: 原始停车数据目录
    output_dir: 简化版停车数据输出目录
    """
    try:
        print(f"开始处理城市: {city_name}")

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 检查是否已经存在简化版文件
        output_file = os.path.join(output_dir, f"{city_name}.csv")
        if os.path.exists(output_file):
            print(f"  简化版文件已存在，跳过: {city_name}")
            return True

        # 读取原始停车数据
        input_file = os.path.join(input_dir, f"{city_name}.csv")
        if not os.path.exists(input_file):
            print(f"  原始数据文件不存在，跳过: {input_file}")
            return False

        vs_parking_df = pd.read_csv(input_file)
        print(f"  读取到 {len(vs_parking_df)} 条停车记录")

        # 解析时间字段
        vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
        vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)

        # 获取最小续航里程作为简化参数
        min_range_large = min(battery_capacity[0])
        min_range_medium = min(battery_capacity[1])
        min_range_small = min(battery_capacity[2])

        print(f"  使用最小电池容量: {min_range_large}/{min_range_medium}/{min_range_small} kwh 作为简化参数")

        # 使用data_utils中的simplify_vs_df函数进行简化
        print(f"  开始简化处理...")
        vs_parking_df_simplified = data_utils.simplify_vs_df(vs_parking_df,
                                                             min_range_large, min_range_medium, min_range_small)
        print(f"  简化后剩余 {len(vs_parking_df_simplified)} 条记录")

        # 保存简化版数据
        vs_parking_df_simplified.to_csv(output_file, index=False)
        print(f"  简化版数据已保存: {output_file}")

        return True

    except Exception as e:
        print(f"  处理城市 {city_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_simplified_parking_data_multiprocess(cities_file='../data/224cities.csv',
                                                  input_dir='../data/input/vs_parking_nodeid',
                                                  output_dir='../data/input/vs_parking_nodeid_simplified',
                                                  num_processes=None):
    """
    使用多进程生成所有城市的简化版停车数据

    Parameters:
    cities_file: 城市列表文件路径
    input_dir: 原始停车数据目录
    output_dir: 简化版停车数据输出目录
    num_processes: 进程数，默认为CPU核心数
    """
    # 读取城市列表
    cities = pd.read_csv(cities_file)
    city_names = cities['city'].tolist()
    print(f"总共 {len(city_names)} 个城市需要处理")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用partial固定部分参数
    process_func = partial(process_single_city,
                           cities_file=cities_file,
                           input_dir=input_dir,
                           output_dir=output_dir)

    # 使用多进程处理
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"使用 {num_processes} 个进程进行处理")

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, city_names)

    # 统计处理结果
    success_count = sum(results)
    print(f"\n处理完成！成功处理 {success_count}/{len(city_names)} 个城市")
    print(f"简化版数据保存在: {output_dir}")


# 使用示例
if __name__ == '__main__':
    # 批量处理所有城市（多进程）
    print("=== 多进程批量生成所有城市的简化版停车数据 ===")
    generate_simplified_parking_data_multiprocess()
