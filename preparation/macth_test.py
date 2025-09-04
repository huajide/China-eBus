import geopandas as gpd
from pyproj import CRS
import pandas as pd


def read_space_separated_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取表头
        header = f.readline().strip().split()
        # 读取数据行
        for line in f:
            fields = line.strip().split()
            if len(fields) == len(header):  # 确保字段数量匹配
                data.append(fields)
    df = pd.DataFrame(data, columns=header)
    return df

"""Initialize"""
cities = pd.read_csv(rf'../data/19cities.csv')
province_list = cities['province'].to_list()
city_list = cities['city'].to_list()

all_route_gdf = []
for i in range(len(province_list)):
    province, city = province_list[i], city_list[i]
    route_gdf = gpd.read_file(rf'../data/cnbusdata2024-2/{province}/{city}/{city}_route5.shp',
                              crs=CRS.from_epsg(4326))
    if len(all_route_gdf):
        all_route_gdf = pd.concat([all_route_gdf, route_gdf], ignore_index=True)
    else:
        all_route_gdf = route_gdf

max_columns = 200  # 200 个发车时间
column_names = ['name2crawl', 'status', 'timestamp', 'route_app', 'direction'] + [f'time_{i}' for i in
                                                                                  range(1, max_columns+1)]
all_chelaile = []
for i in range(len(province_list)):
    province, city = province_list[i], city_list[i]
    city_only = city.replace('市', '')
    csv_path = rf'../data/chelaile/{city_only}result.csv'
    try:
        chelaile = pd.read_csv(csv_path, names=column_names,header=None,engine='python')
    except UnicodeDecodeError:
        chelaile = pd.read_csv(csv_path, names=column_names,header=None, sep='\s+', encoding='gbk')
    except:
        chelaile = read_space_separated_csv(csv_path)
    chelaile.insert(loc=0, column='city', value=city)
    if len(all_chelaile):
        all_chelaile = pd.concat([all_chelaile, chelaile],ignore_index=True)
    else:
        all_chelaile = chelaile

all_chelaile['timetable'] = all_chelaile.iloc[:, 6:].apply(lambda row: row.dropna().tolist(),axis=1)
all_chelaile = all_chelaile.drop(all_chelaile.columns[6:-1], axis=1)

all_info = pd.merge(all_route_gdf, all_chelaile, on=['city','name2crawl'],how='left')
all_info['timetable_y'] = all_info['timetable_y'].apply(lambda x: x if isinstance(x, list) else [])

idx = all_info['timetable_y'].apply(len).groupby(all_info['route_id']).idxmax()
all_info = all_info.loc[idx]

all_info.to_excel(rf'../data/19chelaile_stat.xlsx', index=False)


all_timetables = []
for i in range(len(cities)):
    city_name = cities['city'][i]
    timetables = pd.read_csv(rf'../data/timetable/{city_name}.csv')
    if len(all_timetables):
        all_timetables = pd.concat([all_timetables, timetables], ignore_index=True)
    else:
        all_timetables = timetables.copy(deep=False)


