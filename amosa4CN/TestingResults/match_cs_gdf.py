import os
import pandas as pd
import geopandas as gpd
from pyproj import CRS

from test4plot2 import clearing
from analysis import basic_stat, find_knee
import warnings

warnings.simplefilter("ignore")

# Define target cities and root directory
target_cities = ['上海市', '厦门市', '拉萨市']
root_name = '251026'
mosa_root_dir = rf"../../data/output/mosa/{root_name}"
cs_gdf_dir = r"../../data/input/cs_gdf"
output_root = rf"../../data/output/mosa/{root_name}"

for city_name in target_cities:
    print(f"Processing {city_name}...")

    # 1. Load the original cs_gdf shapefile
    cs_gdf_path = os.path.join(cs_gdf_dir, f"{city_name}.shp")
    if not os.path.exists(cs_gdf_path):
        print(f"  Error: Could not find shapefile for {city_name} at {cs_gdf_path}")
        continue

    cs_gdf = gpd.read_file(cs_gdf_path, crs=CRS.from_epsg(4547))  # 注意这里恢复为您的原epsg或4547
    cs_gdf = cs_gdf.drop_duplicates(subset=['node_id'], keep='first').reset_index(drop=True)
    cs_num = len(cs_gdf)

    # 2. Load MOSA output files
    city_mosa_dir = os.path.join(mosa_root_dir, f"{city_name}")
    if not os.path.exists(city_mosa_dir):
        print(f"  Error: Could not find MOSA output directory for {city_name}")
        continue

    OBJ_G = pd.read_csv(os.path.join(city_mosa_dir, "inf_archive_objs.csv"))
    VAR_G = pd.read_csv(os.path.join(city_mosa_dir, "inf_archive_vars.csv"))
    CV_G = pd.read_csv(os.path.join(city_mosa_dir, "inf_archive_cvs.csv"))

    # 3. Handle column names for obj
    OBJ_G = clearing(OBJ_G)

    # 4. Calculate statistics and find the knee point
    STAT_G = basic_stat(OBJ_G, VAR_G, CV_G, cs_num)
    knee_no_G = find_knee(STAT_G)

    print(f"  Found knee point solution: {knee_no_G}")

    # 5. Get the row index for the knee solution
    target_idx = STAT_G.loc[STAT_G['solution_no'] == knee_no_G].index[0]

    # Extract decision from VAR_G and charger amounts from CV_G
    knee_vars = VAR_G.iloc[target_idx]
    knee_cvs = CV_G.iloc[target_idx]

    # 6. Extract build decisions and charger counts
    build_vars = []
    fast_chargers = []
    slow_chargers = []

    for i in range(cs_num):
        # 取决于您 VAR_G 的格式，如果第一列不是冗余列（如 index），则 i 就是是否建站
        # 如果包含额外列，请把 i 改为对应的偏移量 (如 i + 1)
        # 我们用原逻辑假设：
        build_decision = knee_vars.iloc[i+1]

        # 充电桩数量在 CV_G 里：假设前 cs_num 个是 fast_c，后 cs_num 个是 slow_c
        # 同样如果 CV_G 有如 "solution_no" 这样的非数据前置列，请 +1 偏移。
        # 如果您的设计确实是 CV_G.iloc[i] 为快充，CV_G.iloc[i + cs_num] 为慢充：
        fast_c = -(knee_cvs.iloc[i+2]) if build_decision == 1 else 0
        slow_c = -(knee_cvs.iloc[i + cs_num+2]) if build_decision == 1 else 0

        build_vars.append(int(build_decision))
        fast_chargers.append(int(fast_c))
        slow_chargers.append(int(slow_c))

    # 7. Append data to the GeoDataFrame
    cs_gdf['is_built'] = build_vars
    cs_gdf['fast_c'] = fast_chargers
    cs_gdf['slow_c'] = slow_chargers

    total_built = sum(build_vars)
    print(f"  Stations to be built: {total_built} out of {cs_num}")

    # 8. Save the new GeoDataFrame to the specific city folder
    output_shapefile_dir = os.path.join(output_root, f"{city_name}")
    os.makedirs(output_shapefile_dir, exist_ok=True)

    output_path = os.path.join(output_shapefile_dir, f"knee_cs.shp")
    cs_gdf.to_file(output_path, encoding='utf-8')

    print(f"  Saved {city_name} knee point configuration to {output_path}\n")

print("Processing complete.")
