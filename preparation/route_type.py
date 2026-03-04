import geopandas as gpd
from pyproj import CRS
import pandas as pd
import numpy as np

def route_type(ROUTE_GDFs):
    ROUTE_GDF = ROUTE_GDFs.copy()
    ROUTE_GDF['stop_count'] = ROUTE_GDF['route_stop'].str.count(',')
    ROUTE_GDF['stop_spacing'] = ROUTE_GDF['distance'] / (ROUTE_GDF['stop_count'] - 1).clip(lower=1)

    # 新增 vehicle_type 列
    ROUTE_GDF['vehicle_type'] = np.where(
        ROUTE_GDF['distance'] <= 8,
        'small',
        np.where(
            ROUTE_GDF['distance'] <= 12,
            'medium',
            np.where(
                ROUTE_GDF['distance'] >= 15,
                'large',
                np.where(
                    ROUTE_GDF['stop_spacing'] < 0.5,
                    'medium',
                    'large'
                )
            )
        )
    )

    type_priority = {'small': 0, 'medium': 1, 'large': 2}

    # # 为每个name2crawl组分配统一的vehicle_type（取最大类型）
    # def assign_max_type(group):
    #     max_type = group['vehicle_type'].map(type_priority).max()
    #     return pd.Series({'vehicle_type': list(type_priority.keys())[max_type]})

    # 应用分组操作前确保有name2crawl列
    if 'name2crawl' in ROUTE_GDF.columns:
        # 保存原始索引
        original_index = ROUTE_GDF.index

        # 创建新的DataFrame来存储结果
        result_df = pd.DataFrame()

        # 按name2crawl分组
        for name, group in ROUTE_GDF.groupby('name2crawl'):
            # 获取最大类型
            max_type = group['vehicle_type'].map(type_priority).max()
            # 应用到所有行
            group['vehicle_type'] = list(type_priority.keys())[max_type]
            result_df = pd.concat([result_df, group])

        # 恢复原始索引顺序
        result_df = result_df.reindex(original_index)

        ROUTE_GDF = result_df

    ROUTE_GDF.drop(columns=['stop_count', 'stop_spacing'], inplace=True)

    return ROUTE_GDF



if __name__ == '__main__':
    province_name = '山东省'
    city_name = '威海市'
    city_only = city_name.replace('市', '')

    """standardize service hour"""
    route_gdf = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
                              crs=CRS.from_epsg(4326))
    vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')
    route_gdfs = route_type(route_gdf)
