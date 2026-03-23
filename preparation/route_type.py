import geopandas as gpd
from pyproj import CRS
import pandas as pd
import numpy as np

def route_type(ROUTE_GDFs):
    ROUTE_GDF = ROUTE_GDFs.copy()
    ROUTE_GDF['stop_count'] = ROUTE_GDF['route_stop'].str.count(',')
    ROUTE_GDF['stop_spacing'] = ROUTE_GDF['distance'] / (ROUTE_GDF['stop_count'] - 1).clip(lower=1)

    # Add the vehicle_type column.
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

    # def assign_max_type(group):
    #     max_type = group['vehicle_type'].map(type_priority).max()
    #     return pd.Series({'vehicle_type': list(type_priority.keys())[max_type]})

    # Apply grouping only when name2crawl is available.
    if 'name2crawl' in ROUTE_GDF.columns:
        # Keep the original index order.
        original_index = ROUTE_GDF.index

        # Create a temporary DataFrame for grouped assignment.
        result_df = pd.DataFrame()

        # Group by name2crawl.
        for name, group in ROUTE_GDF.groupby('name2crawl'):
            # Use the largest vehicle type within the group.
            max_type = group['vehicle_type'].map(type_priority).max()
            # Apply the group type to all rows.
            group['vehicle_type'] = list(type_priority.keys())[max_type]
            result_df = pd.concat([result_df, group])

        # Restore the original row order.
        result_df = result_df.reindex(original_index)

        ROUTE_GDF = result_df

    ROUTE_GDF.drop(columns=['stop_count', 'stop_spacing'], inplace=True)

    return ROUTE_GDF

if __name__ == '__main__':
    province_name = 'XX'
    city_name = 'XX'

    """standardize service hour"""
    route_gdf = gpd.read_file(rf'../data/cnbusdata/{province_name}/{city_name}/{city_name}_route5.shp',
                              crs=CRS.from_epsg(4326))
    vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}_{province_name}.csv')
    route_gdfs = route_type(route_gdf)
