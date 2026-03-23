import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dateutil import parser
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from test4plot2 import clearing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


def basic_stat(obj_df, var_df, cv_df0, cs_num):
    """
    df: processed by the function 'clearing'
    """
    cv_df = cv_df0.copy()
    for col in range(1, cs_num + 1):
        mask = var_df.iloc[:, col] == 0
        target1 = col + 1
        target2 = col + cs_num + 1
        cv_df.iloc[mask, target1] = np.nan
        cv_df.iloc[mask, target2] = np.nan

    stat_df = obj_df.copy()
    stat_df['bulit_num'] = var_df.iloc[:, 1:cs_num + 1].sum(axis=1)
    stat_df['fast_num_avg'] = cv_df.iloc[:, 2:cs_num + 2].mean(axis=1)
    stat_df['slow_num_avg'] = cv_df.iloc[:, 2 + cs_num:2 * cs_num + 2].mean(axis=1)
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


def find_knee(objs):
    """
    minmax=True: when objective normalization needs to be calculated based on scenarios
    """
    df = objs.copy()
    scaler = MinMaxScaler()
    df[['obj1', 'obj2']] = scaler.fit_transform(df[['obj1', 'obj2']])
    min_values = df[['obj1', 'obj2']].min()

    df['distance'] = ((df[['obj1', 'obj2']] - min_values) ** 2).sum(axis=1) ** 0.5
    closest_row = df.loc[df['distance'].idxmin()]

    # Return the 'solution_no' of this row
    solution_no = closest_row['solution_no']
    return solution_no


def least_station(gdf, max_dis=5000, plot=False):
    if gdf.crs != 'EPSG:2326':
        gdf = gdf.to_crs('EPSG:2326')
    sindex = gdf.sindex

    # 4. Compute neighbors for each station (stations within a distance of 5 km)
    neighbors = []
    for i in range(len(gdf)):
        point = gdf.geometry.iloc[i]
        # Use the spatial index to query stations within 5000 meters (5 km)
        buffer = point.buffer(max_dis)
        nearby_indices = list(sindex.query(buffer, predicate='intersects'))
        neighbors.append(nearby_indices)

    # 5. Use a greedy algorithm to select the minimum number of charging stations
    uncovered = set(range(len(gdf)))  # Set of uncovered stations
    selected = []  # Indices of selected charging stations

    while uncovered:
        max_cover = 0
        best_station = None
        # Iterate over all unselected stations and find the candidate that covers the most uncovered stations
        for i in range(len(gdf)):
            if i not in selected:
                cover = set(neighbors[i]) & uncovered  # Uncovered stations that this station can cover
                if len(cover) > max_cover:
                    max_cover = len(cover)
                    best_station = i
        if best_station is not None:
            selected.append(best_station)
            uncovered -= set(neighbors[best_station])  # Remove covered stations
        else:
            break  # If no station can be selected, exit the loop (theoretically should not happen, because each station covers itself)

    charging_list = [1 if i in selected else 0 for i in range(len(gdf))]

    if plot:
        # 6. Generate the charging station distribution map
        gdf['is_charging'] = 0  # Add a column with default value 0
        for idx in selected:
            gdf.loc[idx, 'is_charging'] = 1  # Mark selected stations as 1

        # Plot the distribution map
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column='is_charging', cmap='cool', legend=True, ax=ax)
        ax.set_title('Charging Stations Distribution in Hong Kong')
        plt.show(block=True)

    return charging_list


if __name__ == '__main__':
    result_path = 'XXX'
    solutions_folder = rf"../../data/output/{result_path}"
