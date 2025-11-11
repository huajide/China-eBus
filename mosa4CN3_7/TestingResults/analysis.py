import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test4mosa import Location
import data_utils
from simulation import SimVehicle
import pickle
from dateutil import parser
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from test4plot2 import clearing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


def basic_stat(obj_df,var_df,cv_df0, cs_num):
    """
    df: which processed after function 'clearing'
    """
    cv_df = cv_df0.copy()
    for col in range(1,cs_num+1):
        mask = var_df.iloc[:, col] == 0  # 找出第 col 列为 0 的行
        target1 = col+1 # 后推 cs_num 的列索引
        target2 = col + cs_num+1  # 后推 2*cs_num 的列索引
        cv_df.iloc[mask, target1] = np.nan
        cv_df.iloc[mask, target2] = np.nan

    stat_df = obj_df.copy()
    stat_df['bulit_num'] = var_df.iloc[:, 1:cs_num + 1].sum(axis=1)
    # 计算非零值的平均值
    stat_df['fast_num_avg'] = cv_df.iloc[:, 2:cs_num+2].mean(axis=1)
    stat_df['slow_num_avg'] = cv_df.iloc[:, 2+cs_num:2*cs_num+2].mean(axis=1)
    stat_df['fast_num_avg'] = stat_df['fast_num_avg'].apply(lambda x: -x if x < 0 else x)
    stat_df['slow_num_avg'] = stat_df['slow_num_avg'].apply(lambda x: -x if x < 0 else x)

    stat_df['large_num'] = cv_df.iloc[:, -3]
    stat_df['medium_num'] = cv_df.iloc[:, -2]
    stat_df['small_num'] = cv_df.iloc[:, -1]
    stat_df['large_num'] = stat_df['large_num'].apply(lambda x: -x if x < 0 else x)
    stat_df['medium_num'] = stat_df['medium_num'].apply(lambda x: -x if x < 0 else x)
    stat_df['small_num'] = stat_df['small_num'].apply(lambda x: -x if x < 0 else x)
    stat_df['large_model'] = var_df.iloc[:, -3]
    stat_df['medium_model'] = var_df.iloc[:, -2]
    stat_df['small_model'] = var_df.iloc[:, -1]

    return stat_df


# def find_knee(objs):
#     df = objs.copy()
#     scaler = MinMaxScaler()
#     df[['obj1', 'obj2', 'obj3']] = scaler.fit_transform(df[['obj1', 'obj2', 'obj3']])
#
#     min_values = df[['obj1', 'obj2', 'obj3']].min()
#
#     df['distance'] = ((df[['obj1', 'obj2', 'obj3']] - min_values) ** 2).sum(axis=1) ** 0.5
#     closest_row = df.loc[df['distance'].idxmin()]
#
#     # 返回该行的 'solution_no'
#     solution_no = closest_row['solution_no']
#     return solution_no


def find_knee(objs):
    """
    minmax=True: 当目标函数标准化需要基于情景计算时
    """
    df = objs.copy()
    scaler = MinMaxScaler()
    df[['obj1', 'obj2']] = scaler.fit_transform(df[['obj1', 'obj2']])
    min_values = df[['obj1', 'obj2']].min()

    df['distance'] = ((df[['obj1', 'obj2']] - min_values) ** 2).sum(axis=1) ** 0.5
    closest_row = df.loc[df['distance'].idxmin()]

    # 返回该行的 'solution_no'
    solution_no = closest_row['solution_no']
    return solution_no


def least_station(gdf, max_dis=5000, plot=False):
    if gdf.crs != 'EPSG:2326':
        gdf = gdf.to_crs('EPSG:2326')
    sindex = gdf.sindex

    # 4. 计算每个站点的邻居（距离小于等于5公里的站点）
    neighbors = []
    for i in range(len(gdf)):
        point = gdf.geometry.iloc[i]
        # 使用空间索引查询距离小于等于5000米（5公里）的站点
        buffer = point.buffer(max_dis)  # 创建 5 公里缓冲区
        nearby_indices = list(sindex.query(buffer, predicate='intersects'))
        neighbors.append(nearby_indices)

    # 5. 贪心算法选择最少的充电站
    uncovered = set(range(len(gdf)))  # 未覆盖的站点集合
    selected = []  # 选中的充电站索引

    while uncovered:
        max_cover = 0
        best_station = None
        # 遍历所有未选中的站点，找到覆盖最多未覆盖站点的候选站点
        for i in range(len(gdf)):
            if i not in selected:
                cover = set(neighbors[i]) & uncovered  # 该站点能覆盖的未覆盖站点
                if len(cover) > max_cover:
                    max_cover = len(cover)
                    best_station = i
        if best_station is not None:
            selected.append(best_station)
            uncovered -= set(neighbors[best_station])  # 移除已覆盖的站点
        else:
            break  # 如果没有站点可选，退出循环（理论上不会发生，因为每个站点至少覆盖自己）
    charging_list = [1 if i in selected else 0 for i in range(len(gdf))]
    if plot:
        # 6. 生成充电站分布图
        gdf['is_charging'] = 0  # 添加列，默认值为0
        for idx in selected:
            gdf.loc[idx, 'is_charging'] = 1  # 选中的站点标记为1

        # 绘制分布图
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column='is_charging', cmap='cool', legend=True, ax=ax)
        ax.set_title('Charging Stations Distribution in Hong Kong')
        plt.show(block=True)
    return charging_list

if __name__ == '__main__':
    result_path = '250630/莆田市_250630'
    solutions_folder = rf"../../data/output/{result_path}"
    cs_gdf = gpd.read_file(r'../../data/input/cs_gdf/莆田市.shp', crs=CRS.from_epsg(4326))  # 充电站
    cs_gdf = cs_gdf.drop_duplicates(subset=['node_id'], keep='first').reset_index(drop=True)
    CS_NUM = len(cs_gdf)

    """module 1: convert solutions to shp"""
    # # # input
    # solution_name = 'SA10552M2'
    #
    # solutions_data = pd.read_csv(solutions_folder+r'/archive_vars.csv')
    # first_column = solutions_data.iloc[:, 0]
    # row_indices = first_column[first_column == solution_name].index.tolist()[0]
    #
    # cs_gdf['if_built'] = solutions_data.iloc[row_indices, 1:cs_num+1].tolist()
    # cs_gdf['fast_num'] = solutions_data.iloc[row_indices, 1+cs_num:2*cs_num+1].tolist()
    # cs_gdf['slow_num'] = solutions_data.iloc[row_indices, 1+2*cs_num:3*cs_num+1].tolist()
    #
    # # # output
    # cs_gdf.to_file(f"{solutions_folder}/cs_{solution_name}.shp", driver='ESRI Shapefile')


    """module 2: summary report"""
    objs = pd.read_csv(solutions_folder+r'/archive_objs.csv')
    vars = pd.read_csv(solutions_folder+r'/archive_vars.csv')
    cvs = pd.read_csv(solutions_folder + r'/archive_cvs.csv')
    objs = clearing(objs)

    stat = basic_stat(objs, vars, cvs, CS_NUM)

    # knee_point = find_knee(objs, minmax=True, is_grade=True)



    # objs.to_csv(f"{solutions_folder}/summary_report.csv")



