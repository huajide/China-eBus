import json
import os
import time
import requests
from requests import RequestException
import pandas as pd
from pyproj import CRS
import ast

import geopandas as gpd
import transbigdata as tbd
from datetime import datetime,timedelta


def request_url_get(url):
    for attempt in range(3):
        try:
            response = requests.get(url=url, timeout=20)
        except RequestException:
            if attempt == 2:
                raise RuntimeError("AMap request failed after three attempts") from None
            time.sleep(3)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"AMap request failed with HTTP {response.status_code}")
        return response.text


def parse_json(content_json):
    try:
        result_json = json.loads(content_json)
    except (TypeError, ValueError):
        raise ValueError("AMap response is not valid JSON") from None
    return result_json


def request_api(url):
    result = request_url_get(url)
    result_json = parse_json(result)
    return result_json


def remove_comma_before_parenthesis(s):
    
    index = s.find(' )')

    
    if index > 0 and s[index - 1] == ',':
        
        return s[:index - 1] + s[index:]

    return s


def get_pt_duration(start_point, end_point, city, date: str, clock: str, key=None):
    """
    :param start_point: WGS84 coordinates tuple or Point
    :param end_point: WGS84 coordinates tuple or Point
    :param city:
    :param date:
    :param clock:
    :param key:
    :return:
    """
    
    key = key or os.environ.get("AMAP_API_KEY")
    if not key:
        raise ValueError("Set AMAP_API_KEY or pass an API key explicitly")
    if hasattr(start_point, 'x') and hasattr(start_point, 'y'):
        start_point = (start_point.x, start_point.y)
    elif isinstance(start_point, (list, tuple)) and len(start_point) >= 2:
        start_point = (start_point[0], start_point[1])

    if hasattr(end_point, 'x') and hasattr(end_point, 'y'):
        end_point = (end_point.x, end_point.y)
    elif isinstance(end_point, (list, tuple)) and len(end_point) >= 2:
        end_point = (end_point[0], end_point[1])

    start_point = tbd.wgs84togcj02(start_point[0], start_point[1])
    end_point = tbd.wgs84togcj02(end_point[0], end_point[1])

    
    start_point = (round(start_point[0], 6), round(start_point[1], 6))
    end_point = (round(end_point[0], 6), round(end_point[1], 6))

    min_walk_dist, duration, min_transfer, vilocity = 200, -1, 99, -1
    pt_path = f'https://restapi.amap.com/v3/direction/transit/integrated?origin={start_point[0]},{start_point[1]}&' \
              f'destination={end_point[0]},{end_point[1]}&city={city}&date={date}&time={clock}&strategy=3&key={key}'
    pt_info = request_api(pt_path)
    try:
        for j in range(len(pt_info['route']['transits'])):
            walk_dist = pt_info['route']['transits'][j]['walking_distance']
            transfer = max(len(pt_info['route']['transits'][j]['segments']) - 2, 0)
            if int(walk_dist) < min_walk_dist and transfer == 0:
                min_walk_dist = int(walk_dist)
                min_transfer = transfer
                duration = pt_info['route']['transits'][j]['segments'][0]['bus']['buslines'][0]['duration']
                distance = pt_info['route']['transits'][j]['segments'][0]['bus']['buslines'][0]['distance']
                vilocity = int(distance) / 1000 / int(duration) * 3600
    except:
        print(f'AMap API request failed while processing {city}')
    time.sleep(0.1)  

    return min_walk_dist, int(duration), min_transfer, vilocity, pt_info


def start_end_point(stops, output='point'):
    """
    :param stops:
    :param output: point or coords
    :return:
    """
    min_index = stops['sequence'].idxmin()
    max_index = stops['sequence'].idxmax()

    if output == 'point':
        start_point = stops.loc[min_index, 'geometry']
        end_point = stops.loc[max_index, 'geometry']
    elif output == 'coords':
        start_point = (stops.loc[min_index, 'lon'], stops.loc[min_index, 'lat'])
        end_point = (stops.loc[max_index, 'lon'], stops.loc[max_index, 'lat'])

    return start_point, end_point


def find_farthest_point(stops, output='point'):
    if output != 'point':
        check_stops(stops)

    start_point, end_point = start_end_point(stops)

    max_sum_distance = 0
    farthest_idx = None
    for IDX, ROW in stops.iterrows():
        current_point = ROW['geometry']
        distance_to_start = current_point.distance(start_point)
        distance_to_end = current_point.distance(end_point)
        sum_distance = distance_to_start + distance_to_end

        if sum_distance >= max_sum_distance:
            max_sum_distance = sum_distance
            farthest_idx = IDX

    if output == 'coords':
        farthest_point = (stops.loc[farthest_idx, 'lon'], stops.loc[farthest_idx, 'lat'])
    else:
        farthest_point = stops.loc[farthest_idx, 'geometry']

    
    if hasattr(farthest_point, 'x') and hasattr(farthest_point, 'y'):
        
        stops_crs = stops.crs
        if stops_crs != CRS.from_epsg(4326):
            
            farthest_gdf = gpd.GeoDataFrame([{'geometry': farthest_point}], crs=stops_crs)
            farthest_gdf = farthest_gdf.to_crs(epsg=4326)
            farthest_point = farthest_gdf.iloc[0]['geometry']

        
        farthest_point = (farthest_point.x, farthest_point.y)

    return farthest_point


def circuity_calculation(stops, path_distance):
    start_point, end_point = start_end_point(stops)
    euc_dist = start_point.distance(end_point) + 1
    circuity_factor = path_distance * 1000 / euc_dist

    return circuity_factor


def check_stops(stops):
    
    if stops.crs != 'EPSG:4547':
        stops = stops.to_crs(epsg=4547)

    
    if 'lon' not in stops.columns or 'lat' not in stops.columns:
        print("Missing lon and lat fields; generating coordinates using EPSG:4326...")
        
        stops_4326 = stops.to_crs(epsg=4326)
        
        stops['lon'] = stops_4326['geometry'].x
        stops['lat'] = stops_4326['geometry'].y


def get_all_pt_duration(timetable,stops, CITY, DATE, KEY):
    """

    :param KEY:
    :param DATE:
    :param CITY:
    :param timetable:
    :param stops: stop_gdf WGS84
    :return: timetable (route_info) with duration
    """
    stops['lon'] = stops['geometry'].x
    stops['lat'] = stops['geometry'].y
    stops = stops.to_crs("EPSG:4547")

    peak_times = ['7:45','17:45','8:45','18:45']
    offpeak_times = ['12:45','10:45','14:45']
    peak_times = [datetime.strptime(t, '%H:%M') for t in peak_times]
    offpeak_times = [datetime.strptime(t, '%H:%M') for t in offpeak_times]

    timetable['walk_dist_peak'] = 0
    timetable['duration_peak'] = 0
    timetable['transfer_peak'] = 0
    timetable['walk_dist_offpeak'] = 0
    timetable['duration_offpeak'] = 0
    timetable['transfer_offpeak'] = 0
    timetable['circuity'] = -1

    PT_INFO_ALL = []
    for idx,row in timetable.iterrows():
        print(f'{idx}/{len(timetable)-1}...')
        
        
        
        
        try:
            start_time, end_time = (datetime.strptime(row['s_time'], '%H:%M:%S'),
                                    datetime.strptime(row['e_time'], '%H:%M:%S')+timedelta(minutes=30))
            for peak_time in peak_times:
                if end_time > peak_time > start_time:
                    peak_time = peak_time.strftime('%H:%M')
                    break
            for offpeak_time in offpeak_times:
                if end_time > offpeak_time > start_time:
                    offpeak_time = offpeak_time.strftime('%H:%M')
                    break
        except ValueError:
            peak_time = peak_times[0].strftime('%H:%M')
            offpeak_time = offpeak_times[0].strftime('%H:%M')
            print(f'NO START TIME')


        stop_slice = stops[(stops['route_name']==row['route_name'])]

        s_point, e_point = start_end_point(stop_slice,'coords')

        circuity = circuity_calculation(stop_slice, row['distance'])
        timetable.loc[idx, "circuity"] = circuity

        if circuity > 2:
            f_point = find_farthest_point(stop_slice)

            WALK_DIST1, DUR1, TF1, V1, PT_DICT1 = get_pt_duration(s_point, f_point, CITY, DATE, peak_time, KEY)
            WALK_DIST2, DUR2, TF2, V2, PT_DICT2 = get_pt_duration(f_point, e_point, CITY, DATE, peak_time, KEY)

            if (V1 > 0 and V2 < 0) or (V1 < 0 and V2 > 0):
                V = V1 if V1 > 0 else V2
                timetable.loc[idx, "walk_dist_peak"] = 0
                timetable.loc[idx, "duration_peak"] = int(row['distance']/V*3600)
                timetable.loc[idx, "transfer_peak"] = 0
            else:
                timetable.loc[idx, "walk_dist_peak"] = WALK_DIST1 + WALK_DIST2
                timetable.loc[idx, "duration_peak"] = DUR1 + DUR2
                timetable.loc[idx, "transfer_peak"] = TF1 + TF2

            PT_DICT1['route_id'] = row['route_id']
            PT_DICT1['period'] = 'peak'
            PT_INFO_ALL.append(PT_DICT1)
            PT_DICT2['route_id'] = row['route_id']
            PT_DICT2['period'] = 'peak'
            PT_INFO_ALL.append(PT_DICT2)

            WALK_DIST1, DUR1, TF1, V1, PT_DICT1 = get_pt_duration(s_point, f_point, CITY, DATE, offpeak_time, KEY)
            WALK_DIST2, DUR2, TF2, V2, PT_DICT2 = get_pt_duration(f_point, e_point, CITY, DATE, offpeak_time, KEY)

            if (V1 > 0 and V2 < 0) or (V1 < 0 and V2 > 0):
                V = V1 if V1 > 0 else V2
                timetable.loc[idx, "walk_dist_offpeak"] = 0
                timetable.loc[idx, "duration_offpeak"] = int(row['distance']/V*3600)
                timetable.loc[idx, "transfer_offpeak"] = 0
            else:
                timetable.loc[idx, "walk_dist_offpeak"] = WALK_DIST1 + WALK_DIST2
                timetable.loc[idx, "duration_offpeak"] = DUR1 + DUR2
                timetable.loc[idx, "transfer_offpeak"] = TF1 + TF2

            PT_DICT1['route_id'] = row['route_id']
            PT_DICT1['period'] = 'offpeak'
            PT_INFO_ALL.append(PT_DICT1)
            PT_DICT2['route_id'] = row['route_id']
            PT_DICT2['period'] = 'offpeak'
            PT_INFO_ALL.append(PT_DICT2)

        else:
            WALK_DIST, DUR, TF, V, PT_DICT = get_pt_duration(s_point, e_point, CITY, DATE, peak_time, KEY)

            timetable.loc[idx, "walk_dist_peak"] = WALK_DIST
            timetable.loc[idx, "duration_peak"] = DUR
            timetable.loc[idx, "transfer_peak"] = TF
            PT_DICT['route_id'] = row['route_id']
            PT_DICT['period'] = 'peak'
            PT_INFO_ALL.append(PT_DICT)

            WALK_DIST, DUR, TF, V, PT_DICT = get_pt_duration(s_point, e_point, CITY, DATE, offpeak_time, KEY)

            timetable.loc[idx, "walk_dist_offpeak"] = WALK_DIST
            timetable.loc[idx, "duration_offpeak"] = DUR
            timetable.loc[idx, "transfer_offpeak"] = TF
            PT_DICT['route_id'] = row['route_id']
            PT_DICT['period'] = 'offpeak'
            PT_INFO_ALL.append(PT_DICT)

    return timetable, PT_INFO_ALL


def get_interval(timetables):
    if  isinstance(timetables, str):
        timetables = ast.literal_eval(timetables)
    if len(timetables) <= 1:
        return 60, 60
    minutes = []
    for t in timetables:
        
        try:
            dt = datetime.strptime(t, '%H:%M')
        except:
            continue
        
        total_minutes = dt.hour * 60 + dt.minute
        minutes.append(total_minutes)

    
    minutes.sort()

    
    intervals = [minutes[i + 1] - minutes[i] for i in range(len(minutes) - 1)]

    
    min_interval = min(intervals)
    max_interval = max(intervals)

    return min_interval, max_interval


def reduce_duration(row):
    min_itv, max_itv = get_interval(row['timetables'])
    
    
    if min_itv > 40:
        row['duration_peak'] = -1
        row['duration_offpeak'] = -1
    else:
        delta_reduction = min(min_itv/2*60,  0.6 * row['duration_peak'], 0.6 * row['duration_offpeak'])
        row['duration_peak'] = row['duration_peak'] - delta_reduction
        row['duration_offpeak'] = row['duration_offpeak'] - delta_reduction

        stop_count = row['route_stop'].count(',')
        est_speed = row['distance']/(row['distance']/50 + stop_count/60)
        actual_speed_peak = row['distance'] / row['duration_peak'] * 3600
        actual_speed_offpeak = row['distance'] / row['duration_offpeak'] * 3600
        if actual_speed_peak < est_speed*0.6 or actual_speed_peak > est_speed*1.5:
            row['duration_peak'] = row['distance'] / est_speed * 3600
        elif actual_speed_peak < 10 or actual_speed_peak > 70:
            row['duration_peak'] = row['distance'] / est_speed * 3600

        if actual_speed_offpeak < est_speed*0.6 or actual_speed_offpeak > est_speed*1.5:
            row['duration_offpeak'] = row['distance'] / est_speed * 3600
        elif actual_speed_offpeak < 10 or actual_speed_offpeak > 70:
            row['duration_offpeak'] = row['distance'] / est_speed * 3600

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    return row


def fill_peak_or_offpeak_blank(row):
    if row['duration_peak'] < 0 < row['duration_offpeak']:
        row['duration_peak'] = row['duration_offpeak']
    elif (row['duration_offpeak'] < 0 < row['duration_peak']) or row['duration_offpeak'] > row['duration_peak']:
        row['duration_offpeak'] = row['duration_peak']
    return row


def fill_peak_and_offpeak_blank(row):
    if row['duration_offpeak'] <= 0 and row['duration_peak'] <= 0:
        stop_count = row['route_stop'].count(',')
        dur = row['distance'] / 50 * 3600 + stop_count*60  
        row['duration_offpeak'] = dur
        row['duration_peak'] = dur

    return row
def update_durations(routes):
    routes = routes.apply(reduce_duration, axis=1)

    routes = routes.apply(fill_peak_or_offpeak_blank, axis=1)

    for idx, row in routes.iterrows():
        
        reverse_mask = (
                (routes['name2crawl'] == row['name2crawl']) &
                (routes['s_stop'] == row['e_stop']) &
                (routes['e_stop'] == row['s_stop']) &
                (routes['duration_peak'] > 0))
        reverse_rows = routes[reverse_mask]


        if not reverse_rows.empty:
            if row['duration_peak'] < 0 and row['duration_offpeak'] < 0:
                
                routes.at[idx, 'duration_peak'] = reverse_rows.iloc[0]['duration_peak']
                routes.at[idx, 'duration_offpeak'] = reverse_rows.iloc[0]['duration_offpeak']
                
            else:
                this_speed = row['distance'] / row['duration_peak'] * 3600
                that_speed = reverse_rows.iloc[0]['distance'] / reverse_rows.iloc[0]['duration_peak'] * 3600

                if max(this_speed, that_speed) / min(this_speed, that_speed) > 1.5:
                    
                    diff_this = abs(this_speed - 20)
                    diff_that = abs(that_speed - 20)
                    if diff_that < diff_this:
                        routes.at[idx, 'duration_peak'] = row['distance']/that_speed*3600
                        routes.at[idx, 'duration_offpeak'] = row['distance']/that_speed*3600

        if row['duration_peak'] < 0 and row['duration_offpeak'] < 0:
            
            reverse_mask = (
                    (routes['name2crawl'] == row['name2crawl']) &
                    (routes['s_stop'] == row['e_stop']) &
                    (routes['e_stop'] == row['s_stop']) &
                    (routes['duration_peak'] > 0))

            reverse_rows = routes[reverse_mask]
            if not reverse_rows.empty:
                
                routes.at[idx, 'duration_peak'] = reverse_rows.iloc[0]['duration_peak']
                routes.at[idx, 'duration_offpeak'] = reverse_rows.iloc[0]['duration_offpeak']
                

    routes = routes.apply(fill_peak_and_offpeak_blank, axis=1)
    routes.drop(columns=['walk_dist_peak', 'walk_dist_offpeak', 'transfer_peak', 'transfer_offpeak'], inplace=True)

    return routes


def trajectory_generation(routes, stop_gdf, date):
    """
    after updating durations
    :param routes:
    :param stop_gdf:
    :return:
    """
    routes['duration_peak'] = routes['duration_peak'] / 60
    routes['duration_offpeak'] = routes['duration_offpeak'] / 60

    routes['timetables'] = routes['timetables'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    routes_expanded = routes.explode('timetables').reset_index(drop=True)
    
    
    routes_expanded['s_time'] = pd.to_datetime(
        date + ' ' + routes_expanded['timetables'],
        format='%Y-%m-%d %H:%M',
        errors='coerce'
    )
    
    routes_expanded = routes_expanded.dropna(subset=['s_time'])

    routes_expanded = routes_expanded.drop(columns=['timetables'])

    routes_expanded['e_time'] = routes_expanded.apply(
        lambda row: row['s_time'] + pd.Timedelta(minutes=row['duration_peak'])
        if (row['s_time'].hour in range(7, 9) or row['s_time'].hour in range(17, 19))
        else row['s_time'] + pd.Timedelta(minutes=row['duration_offpeak']),
        axis=1
    )
    routes_expanded['e_time'] = routes_expanded['e_time'].dt.round('1s')

    routes_expanded['avg_velocity'] = ((routes_expanded['distance'])
                                          / ((routes_expanded['e_time'] - routes_expanded['s_time'])
                                             .dt.total_seconds() / 3600))
    routes_expanded['distance'] = routes_expanded['distance'] * 1000

    stop_gdf_min_max = stop_gdf.groupby('route_name').agg(
        orientation_coords=(
        'geometry', lambda x: tuple(x.iloc[x.index.get_loc(stop_gdf.loc[x.index, 'sequence'].idxmin())].coords[0])),
        destination_coords=(
        'geometry', lambda x: tuple(x.iloc[x.index.get_loc(stop_gdf.loc[x.index, 'sequence'].idxmax())].coords[0]))
    ).reset_index()

    routes_expanded = routes_expanded.merge(
        stop_gdf_min_max[['route_name', 'orientation_coords', 'destination_coords']],
        on='route_name',
        how='left'
    )
    routes_expanded['duration'] = routes_expanded.apply(
        lambda row: row['duration_peak'] if (row['s_time'].hour in range(7, 9) or row['s_time'].hour in range(17, 19)) else
        row['duration_offpeak'],
        axis=1
    )
    routes_expanded = routes_expanded[
        ['route_id', 'route_name', 'name2crawl', 's_stop', 'e_stop', 's_time', 'e_time',
         'distance', 'avg_velocity', 'duration', 'orientation_coords',
         'destination_coords', 'province', 'city']]

    return routes_expanded



