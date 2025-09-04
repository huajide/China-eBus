import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from dateutil import parser
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from test4plot2 import clearing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


city = ['湘潭市','莆田市','珠海市','绵阳市','乌鲁木齐市','昆明市','沈阳市','哈尔滨市']
info = []
for city_name in city:
    cs_gdf = gpd.read_file(rf'../../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4507))
    vs_parking_df = pd.read_csv(rf'../../data/input/vs_parking_nodeid/{city_name}.csv')  # 车辆行程
    info.append([len(cs_gdf),len(vs_parking_df),vs_parking_df['v_name'].max()])
info = pd.DataFrame(info)
print(info)

