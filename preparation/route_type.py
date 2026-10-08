import pandas as pd
import numpy as np

def route_type(ROUTE_GDFs):
    ROUTE_GDF = ROUTE_GDFs.copy()
    ROUTE_GDF['stop_count'] = ROUTE_GDF['route_stop'].str.count(',')
    ROUTE_GDF['stop_spacing'] = ROUTE_GDF['distance'] / (ROUTE_GDF['stop_count'] - 1).clip(lower=1)

    
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

    
    
    
    

    
    if 'name2crawl' in ROUTE_GDF.columns:
        
        original_index = ROUTE_GDF.index

        
        result_df = pd.DataFrame()

        
        for name, group in ROUTE_GDF.groupby('name2crawl'):
            
            max_type = group['vehicle_type'].map(type_priority).max()
            
            group['vehicle_type'] = list(type_priority.keys())[max_type]
            result_df = pd.concat([result_df, group])

        
        result_df = result_df.reindex(original_index)

        ROUTE_GDF = result_df

    ROUTE_GDF.drop(columns=['stop_count', 'stop_spacing'], inplace=True)

    return ROUTE_GDF
