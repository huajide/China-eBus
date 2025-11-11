import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from pandas.core.interchange.dataframe_protocol import DataFrame
from pyproj import CRS
import contextily as ctx
import pandas as pd
from analysis import basic_stat, find_knee
from test4plot2 import plot_double, clearing
import numpy as np
import matplotlib as mpl
from shapely.geometry import Point
import ast


ax = None
bx= None

def match_cs(cs_gdf: GeoDataFrame,vars:DataFrame,stats:DataFrame,knee_no=None):
    if knee_no is None:
        solu_no = find_knee(stats)
    else:
        solu_no = knee_no
    row_vars = vars[vars.iloc[:,0] == solu_no]
    num = len(cs_gdf)

    this_gdf = cs_gdf.copy(deep=False)

    AA=row_vars.iloc[0, 1:num+1].values
    this_gdf['built'] = AA.T
    AA=row_vars.iloc[0, num+1:2*num+1].values
    this_gdf['fast'] = AA.T
    AA=row_vars.iloc[0, 2*num+1:3*num+1].values
    this_gdf['slow'] = AA.T
    this_gdf.loc[this_gdf['built'] == 0, ['fast', 'slow']] = 0

    return this_gdf


def plot_single_cs(gdf,charger_type,scenario):
    """
    charger_type: 'slow' or 'fast'
    scenario: 'Baseline' or 'Gradient'
    """
    global ax
    gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=gdf[charger_type].to_list())
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels)
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.PositronOnlyLabels)
    ax.set_title(f'{scenario} {charger_type} chargers')
    ax.set_axis_off()

def plot_compared_cs(gdf_b,gdf_g):
    global ax
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221)
    plot_single_cs(gdf_b,'fast','Baseline')
    ax = fig.add_subplot(222)
    plot_single_cs(gdf_g,'fast','Gradient')
    ax = fig.add_subplot(223)
    plot_single_cs(gdf_b,'slow','Baseline')
    ax = fig.add_subplot(224)
    plot_single_cs(gdf_g,'slow','Gradient')

    fig.suptitle('Distribution of Charging Stations', fontsize=12)
    plt.tight_layout()
    plt.show(block=True)


def plot_single_zone(gdf,charger_type,scenario,norm=None):
    global bx
    if norm is not None:
        gdf.plot(column=charger_type, ax=bx, legend=False, norm=norm, cmap='OrRd')  # 使用颜色映射
    else:
        gdf.plot(column=charger_type, ax=bx, legend=True, cmap='OrRd')
    bx.set_title(f'{scenario} {charger_type} chargers')
    bx.set_axis_off()

def plot_zone_cs(gdf_b,gdf_g, zone):
    global bx
    zone['zone_id'] = zone.index

    joined_b = gpd.sjoin(gdf_b, zone, op='within')
    zone_b = joined_b.groupby('zone_id').agg(fast=('fast', 'sum'), slow=('slow', 'sum')).reset_index()
    joined_g = gpd.sjoin(gdf_g, zone, op='within')
    zone_g = joined_g.groupby('zone_id').agg(fast=('fast', 'sum'), slow=('slow', 'sum')).reset_index()

    zone_b = zone.merge(zone_b, left_on='zone_id', right_on='zone_id', how='left')
    zone_g = zone.merge(zone_g, left_on='zone_id', right_on='zone_id', how='left')
    zone_b['fast'] = zone_b['fast'].fillna(0).astype(int)
    zone_b['slow'] = zone_b['slow'].fillna(0).astype(int)
    zone_g['fast'] = zone_g['fast'].fillna(0).astype(int)
    zone_g['slow'] = zone_g['slow'].fillna(0).astype(int)

    v_max = max(zone_b['fast'].max(),zone_b['slow'].max(),zone_g['fast'].max(),zone_g['slow'].max())
    NORM = mpl.colors.Normalize(vmin=0, vmax=v_max)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(10,10))
    bx = fig.add_subplot(221)
    plot_single_zone(zone_b,'fast','Baseline',NORM)
    bx = fig.add_subplot(222)
    plot_single_zone(zone_g,'fast','Gradient',NORM)
    bx = fig.add_subplot(223)
    plot_single_zone(zone_b,'slow','Baseline',NORM)
    bx = fig.add_subplot(224)
    plot_single_zone(zone_g,'slow','Gradient',NORM)

    fig.suptitle('Distribution of Charging Stations by Unit', fontsize=12)
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=NORM)
    sm.set_array([])  # 需要设置数组以避免警告
    cbar = plt.colorbar(sm, ax=bx, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Total count of chargers')  # 设置图例标签
    plt.tight_layout()
    plt.show(block=True)

    return zone_b, zone_g


def plot_delta_zone(zone_b, zone_g):
    global bx
    zone_b.rename(columns={'fast': 'fast_b', 'slow': 'slow_b'}, inplace=True)
    zone_g.rename(columns={'fast': 'fast_g', 'slow': 'slow_g'}, inplace=True)
    zone_b = zone_b.iloc[:, -4:]
    zone_g = zone_g.iloc[:, -3:]
    zone_cs = pd.merge(zone_b, zone_g, on=['zone_id'])
    zone_cs['delta_fast'] = zone_cs['fast_g'] - zone_cs['fast_b']
    zone_cs['delta_slow'] = zone_cs['slow_g'] - zone_cs['slow_b']
    zone_cs['delta'] = zone_cs['delta_fast'] + zone_cs['delta_slow']

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(9,3))
    bx = fig.add_subplot(131)
    plot_single_zone(zone_cs,'delta','')
    bx = fig.add_subplot(132)
    plot_single_zone(zone_cs,'delta_fast','')
    bx = fig.add_subplot(133)
    plot_single_zone(zone_cs,'delta_slow','')

    fig.suptitle('Increased Number of Chargers by Unit', fontsize=12)
    plt.tight_layout()
    plt.show(block=True)

    return zone_cs

if __name__ == '__main__':
    result_path = 'TestingResults0119'
    OBJ_G = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}\{result_path}-Grade\archive_objs.csv")
    OBJ_B = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}\{result_path}-Baseline\archive_objs.csv")
    OBJ_G = clearing(OBJ_G)
    OBJ_B = clearing(OBJ_B)
    VAR_G = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}\{result_path}-Grade\archive_vars.csv")
    VAR_B = pd.read_csv(rf"../../../hkbus\data\output\MOSA\{result_path}\{result_path}-Baseline\archive_vars.csv")

    GDF = gpd.read_file(r'../../../hkbus/data/input/cs_gdf/cs_gdf.shp', crs=CRS.from_epsg(4326))  # 充电站
    GDF = GDF.to_crs(epsg=2326)
    cs_num = len(GDF)

    STAT_G = basic_stat(OBJ_G, VAR_G, cs_num)
    STAT_B = basic_stat(OBJ_B, VAR_B, cs_num)
    GDF_G = match_cs(GDF, VAR_G, STAT_G)
    GDF_B = match_cs(GDF, VAR_B, STAT_B)

    # plot_compared_cs(GDF_B, GDF_G)

    zoneGDF = gpd.read_file(r'../../../../QGIS\HK\HKBoundary\SPU.shp', crs=CRS.from_epsg(2326))  # 充电站

    zone_B, zone_G = plot_zone_cs(GDF_B, GDF_G, zoneGDF)

    "further analysis"
    # zone_B.rename(columns={'fast':'fast_b','slow':'slow_b'},inplace=True)
    # zone_G.rename(columns={'fast': 'fast_g', 'slow': 'slow_g'}, inplace=True)
    # zone_B = zone_B.iloc[:, -4:]
    # zone_G = zone_G.iloc[:, -3:]
    #
    # zone_cs = pd.merge(zone_B,zone_G,on=['zone_id'])
    # vs_parking_df = pd.read_csv(r'../../../hkbus/data/HKsimpreparation/HK_all_vs_parking_nodeid.csv')  # 车辆行程
    # pivot_vs = vs_parking_df.groupby(['lineName', 'lineCo', 'lineDirection', 'destination_coords']).agg(
    #     distance=('distance', 'mean'),
    #     evrange=('evrange', 'mean'),
    #     count=('destination', 'count')
    # ).reset_index()
    # pivot_vs['geometry'] = pivot_vs['destination_coords'].apply(lambda coords: Point(ast.literal_eval(coords)))
    # pivot_vs = gpd.GeoDataFrame(pivot_vs, geometry='geometry',crs=CRS.from_epsg(4326))
    # pivot_vs = pivot_vs.to_crs(epsg=2326)
    # pivot_vs = gpd.sjoin(pivot_vs, zone_cs, how='left', op='intersects')
    # pivot_vs.to_excel(r'aaa.xlsx')
    #
    # zone_cs['delta_fast'] = zone_cs['fast_g'] - zone_cs['fast_b']
    # zone_cs['delta_slow'] = zone_cs['slow_g'] - zone_cs['slow_b']
    # zone_cs['delta'] = zone_cs['delta_fast'] + zone_cs['delta_slow']
    # zone_cs.to_file(r'zone_cs.shp', driver='ESRI Shapefile')



