import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from code4BEBAndHB.mosa4HK3.test4mosa import Location
# from code4BEBAndHB.mosa4HK3 import data_utils
from test4mosa import Location
import data_utils
from simulation import SimVehicle
from joypy import joyplot
from analysis import basic_stat, find_knee
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from test4plot2 import clearing
from dateutil import parser
import pickle
import matplotlib.pyplot as plt
import ast
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import folium
from folium.plugins import HeatMap
import numpy as np

ax = None

def plot_single_zone(gdf,charger_type,scenario,norm=None):
    global ax
    if norm is not None:
        gdf.plot(column=charger_type, ax=ax, legend=False, norm=norm, cmap='OrRd')  # 使用颜色映射
    else:
        gdf.plot(column=charger_type, ax=ax, legend=True, cmap='OrRd')
    ax.set_title(f'{scenario} {charger_type} chargers')
    ax.set_axis_off()


def plot_demand_heatmap(knee_b, knee_g, thereshold=99999):
    global ax
    cs_gdf = gpd.read_file(r'../../../hkbus/data/input/cs_gdf/cs_gdf.shp', crs=CRS.from_epsg(4326))  # 充电站
    vs_parking_df = pd.read_csv(r'../../../hkbus/data/HKsimpreparation/HK_all_vs_parking_nodeid.csv')  # 车辆行程
    vs_parking_df.reset_index(drop=True, inplace=True)
    vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
    vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
    cs_dict = {v: k for k, v in zip(cs_gdf.index, cs_gdf['node_id'])}
    num_vars = data_utils.set_var_num(cs_gdf)

    knee_list = [knee_b, knee_g]
    grade_list = [False, True]

    vs_delay = []
    for i in range(2):
        grade = grade_list[i]
        solu_vars = knee_list[i].iloc[:, 1:].to_numpy()

        if grade:
            vs_parking_df = vs_parking_df.drop(columns=['distance'])
            vs_parking_df = vs_parking_df.rename(columns={'evrange': 'distance'})
            model_name = 'Grade'
        else:
            model_name = 'Baseline'
        with open(rf"../../../hkbus/data/input/formosa/all_d2s_dict_{model_name}.pkl", 'rb') as f:
            all_d2s_dict = pickle.load(f)
        sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').  # 将车辆行程按车辆集计
                                            apply(lambda x: data_utils.derive_simV_info(x)).to_dict(), orient='index')
        sim_v_dict = {}
        for idx, row in sim_v_info.iterrows():
            sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance,
                                         row.avg_velocity)
        problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, cs_gdf=cs_gdf, vs_parking_df=vs_parking_df,
                           all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, is_grade=grade, sim_v_dict=sim_v_dict)

        solu_vars = solu_vars.flatten()
        timeout, subs_idx = problem.eval_vars(solu_vars, is_test=True)
        subs_idx = []  # if what to show the delay before adding new vehicles
        vs_parking_ana = vs_parking_df.copy(deep=False)
        vs_parking_ana['timeout'] = timeout
        vs_replaced = vs_parking_ana.iloc[subs_idx]
        vs_remaining = vs_parking_ana.drop(subs_idx)
        vs_remaining = vs_remaining[vs_remaining['timeout'] > 0]
        vs_remaining['model']=model_name

        if len(vs_delay):
            vs_delay = pd.concat([vs_delay, vs_remaining])
        else:
            vs_delay = vs_remaining.copy(deep=False)

        timeout_group = vs_remaining.groupby('destination_coords')['timeout'].sum().reset_index()
        timeout_group['destination_coords'] = timeout_group['destination_coords'].apply(ast.literal_eval)

        timeout_group['geometry'] = timeout_group['destination_coords'].apply(lambda coords: Point(coords))
        timeout_group = gpd.GeoDataFrame(timeout_group, geometry='geometry')
        timeout_group.set_crs(epsg=4326, inplace=True)
        # timeout_group = timeout_group.to_crs(epsg=2326)
        # timeout_group = timeout_group.drop(timeout_group.columns[0], axis=1)
        timeout_group.loc[timeout_group['timeout'] > thereshold, 'timeout'] = thereshold

        # timeout_group.to_file(rf'shp/delay_{model_name}.shp', driver='ESRI Shapefile')

        m = folium.Map(location=[22.37404, 114.11304],zoom_start=11)
        heat_data = [[coords[1], coords[0], weight] for coords, weight in
                     zip(timeout_group['destination_coords'], timeout_group['timeout'])]
        HeatMap(heat_data).add_to(m)
        m.save(f'{model_name}.html')
    return vs_delay


def calculate_hours(ts):
    # 提取小时和分钟
    hour = ts.hour
    minute = ts.minute
    # 计算小时数
    hours = hour + minute / 60.0
    # 如果日期是第二天，加上24小时
    if ts.day > 18:  # 假设基准日期是9月18日
        hours += 24
    return hours

# def plot_ridgeline(vs_delay):


if __name__ == '__main__':
    cs_gdf = gpd.read_file(r'../../../hkbus/data/input/cs_gdf/cs_gdf.shp', crs=CRS.from_epsg(4326))  # 充电站
    result_path = 'Results0218B1'
    cs_num = len(cs_gdf)
    OBJ_G = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}-Grade\archive_objs.csv")
    OBJ_B = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}-Baseline\archive_objs.csv")
    VAR_G = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}-Grade\archive_vars.csv")
    VAR_B = pd.read_csv(rf"../../../hkbus\data\output\MOSA\\{result_path}-Baseline\archive_vars.csv")
    OBJ_G = clearing(OBJ_G)
    OBJ_B = clearing(OBJ_B)
    STAT_G = basic_stat(OBJ_G, VAR_G, cs_num)
    STAT_B = basic_stat(OBJ_B, VAR_B, cs_num)

    solu_no = find_knee(STAT_G)
    # solu_no = 'SA1940M1'
    KNEE_G = VAR_G[VAR_G.iloc[:,0] == solu_no].reset_index(drop=True)
    solu_no = find_knee(STAT_B)
    # solu_no = 'SA2576M1'
    KNEE_B = VAR_B[VAR_B.iloc[:,0] == solu_no].reset_index(drop=True)

    zoneGDF = gpd.read_file(r'../../../../QGIS\HK\HKBoundary\SPU.shp', crs=CRS.from_epsg(2326))
    zoneGDF['zone_id'] = zoneGDF.index
    zoneGDF = zoneGDF.iloc[:, -2:]

    delay_df = plot_demand_heatmap(KNEE_B, KNEE_G,100)
    delay_df['destination_coords'] = delay_df['destination_coords'].apply(ast.literal_eval)
    delay_df['geometry'] = delay_df['destination_coords'].apply(lambda x: Point(x))
    delay_df = gpd.GeoDataFrame(delay_df, geometry='geometry')
    delay_df.set_crs(epsg=4326, inplace=True)
    delay_df = delay_df.to_crs(epsg=2326)
    delay_df = gpd.sjoin(delay_df, zoneGDF, op='within')
    delay_df['hours'] = delay_df['e_time'].apply(calculate_hours)

    b_group = delay_df[delay_df['model'] == 'Baseline']
    b_group = b_group.groupby('zone_id').agg({
        'timeout': 'sum',
        'lineDirection': lambda x: x.mode()[0]  # 取出现次数最多的值
    }).reset_index()
    b_group.sort_values(by='timeout', ascending=False,inplace=True)
    b_group = b_group.reset_index(drop=True)

    g_group = delay_df[delay_df['model'] == 'Grade']
    g_group = g_group.groupby('zone_id').agg({
        'timeout': 'sum',
        'lineDirection': lambda x: x.mode()[0]  # 取出现次数最多的值
    }).reset_index()
    g_group.sort_values(by='timeout', ascending=False,inplace=True)
    g_group = g_group.reset_index(drop=True)

    '''plot top 10 stations with biggest gap of delay btw baseline and gradient'''
    two_group = pd.merge(b_group,g_group,how='outer',on='zone_id')
    two_group = two_group.fillna(0)
    two_group['gap_delay'] = abs(two_group['timeout_x']-two_group['timeout_y'])
    two_group.sort_values(by='gap_delay', ascending=False, inplace=True)
    two_group = two_group.reset_index(drop=True)
    two_group = two_group.head(10)
    zone_list = two_group['zone_id'].to_list()

    two_group = two_group.iloc[:, [0,2]]
    two_group.rename(columns={'lineDirection_x':'region'},inplace=True)

    b_group2 = delay_df[delay_df['zone_id'].isin(zone_list)]
    b_group2['zone_id'] = pd.Categorical(b_group2['zone_id'], categories=zone_list, ordered=True)
    b_group2 = b_group2.sort_values('zone_id')
    b_group2 = b_group2[b_group2['model'] == 'Baseline']
    b_group2 = b_group2.loc[b_group2.index.repeat(b_group2['timeout']*10)]
    b_group2 = b_group2.iloc[:, [-2,-1]]
    b_group2 = pd.merge(b_group2, two_group, how='inner', on='zone_id')

    g_group2 = delay_df[delay_df['zone_id'].isin(zone_list)]
    g_group2['zone_id'] = pd.Categorical(g_group2['zone_id'], categories=zone_list, ordered=True)
    g_group2 = g_group2.sort_values('zone_id')
    g_group2 = g_group2[g_group2['model'] == 'Grade']
    g_group2 = g_group2.loc[g_group2.index.repeat(g_group2['timeout']*10)]
    g_group2 = g_group2.iloc[:, [-2,-1]]
    g_group2 = pd.merge(g_group2, two_group, how='inner', on='zone_id')

    '''plot top 5 stations with highest delay'''
    # b_group = b_group.head(5)
    # b_region = b_group['lineDirection'].to_list()
    # b_group = b_group.iloc[:, [0,-1]]
    # b_group.rename(columns={'lineDirection':'region'},inplace=True)
    # b_group = pd.merge(delay_df, b_group,on=['zone_id'],how='inner')
    # b_group = b_group[b_group['model'] == 'Baseline']
    # b_group = b_group.iloc[:, [-7,-2,-1]]
    # b_group = b_group.loc[b_group.index.repeat(b_group['timeout'])]
    #
    # g_group = g_group.head(5)
    # g_region = g_group['lineDirection'].to_list()
    # g_group = g_group.iloc[:, [0, -1]]
    # g_group.rename(columns={'lineDirection': 'region'}, inplace=True)
    # g_group = pd.merge(delay_df, g_group,on=['zone_id'],how='inner')
    # g_group = g_group[g_group['model'] == 'Grade']
    # g_group = g_group.iloc[:, [-7, -2, -1]]
    # g_group = g_group.loc[g_group.index.repeat(g_group['timeout'])]


    plt.rcParams.update({
        'font.size': 10,  # 设置全局字体大小
        'font.family': 'Times New Roman'  # 设置全局字体类型
    })
    fig, ax = joyplot(b_group2, by='region', column='hours', figsize=(10, 12),
                      # 传入数据，y轴，x轴，设置图片尺寸
                      linecolor="black",  # 山脊线的颜色
                      linewidth=0.5,
                      range_style='all',
                      # ylim='same',
                      colormap=sns.color_palette("coolwarm", as_cmap=True),
                      overlap=2,
                      alpha=0.7,
                      # 设置山脊图的填充色，用seaborn库的色盘，选择离散型颜色，as_cmap参数用来更改显示的颜色范围是离散的还是连续的
                      );
    # 设置背景色
    plt.title("Delay time");  # 添加标题
    plt.xlim([5,28])
    plt.xlabel('Time (hours)', fontsize=10, fontname='Times New Roman')
    plt.xticks(fontsize=10, fontname='Times New Roman')
    plt.subplots_adjust(top=0.95, bottom=0.1)  # 调整图形距离边框位置

    plt.show(block=True)

    # plot_ridgeline(delay_df)
