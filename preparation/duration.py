import json
import time
import requests
from requests import RequestException
import pandas as pd

import ast
import transbigdata as tbd
from datetime import datetime,timedelta

def request_url_get(url):
    while True:
        try:
            r = requests.get(url=url, timeout=20)
            if r.status_code == 200:
                return r.text
            return None
        except RequestException:
            print('Request URL returned an error')
            time.sleep(3)

def parse_json(content_json):
    """Parse JSON from an API response string."""
    result_json = json.loads(content_json)
    return result_json

def request_api(url):
    """Request the AMap API and parse the response."""
    result = request_url_get(url)
    result_json = parse_json(result)
    return result_json

def remove_comma_before_parenthesis(s):
    # Find the closing parenthesis.
    index = s.find(' )')

    # Remove a trailing comma before the closing parenthesis.
    if index > 0 and s[index - 1] == ',':

        return s[:index - 1] + s[index:]

    return s

def start_end_point(multilinestring):
    first_line = multilinestring.geoms[0]
    first_point = first_line.coords[0]

    last_line = multilinestring.geoms[-1]
    last_point = last_line.coords[-1]

    return first_point, last_point

def get_pt_duration(start_point, end_point, city, date:str, clock:str,key):
    """

    :param start_point: WGS84
    :param end_point:
    :param city:
    :param date:
    :param clock:
    :param key:
    :return:
    """
    start_point = tbd.wgs84togcj02(start_point[0], start_point[1])
    end_point = tbd.wgs84togcj02(end_point[0], end_point[1])
    min_walk_dist, duration, min_transfer, vilocity = 200, -1, 99, -1
    try:
        pt_path = f'https://restapi.amap.com/v3/direction/transit/integrated?origin={start_point[0]},{start_point[1]}&' \
                  f'destination={end_point[0]},{end_point[1]}&city={city}&date={date}&time={clock}&strategy=3&key={key}'
        pt_info = request_api(pt_path)

        for j in range(len(pt_info['route']['transits'])):
            walk_dist = pt_info['route']['transits'][j]['walking_distance']
            transfer = max(len(pt_info['route']['transits'][j]['segments'])-2,0)
            if int(walk_dist) < min_walk_dist and transfer == 0:
                min_walk_dist = int(walk_dist)
                min_transfer = transfer
                duration = pt_info['route']['transits'][j]['segments'][0]['bus']['buslines'][0]['duration']
                distance = pt_info['route']['transits'][j]['segments'][0]['bus']['buslines'][0]['distance']
                vilocity = int(distance) / 1000 / int(duration) * 3600
    except:
        None
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

def find_farthest_point(stops,output='point'):
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
    return farthest_point

def circuity_calculation(stops, path_distance):
    """
    :param stops:
    :param path_distance: kilometers
    :return:
    """
    start_point, end_point = start_end_point(stops)
    euc_dist = start_point.distance(end_point) + 1
    circuity_factor = path_distance * 1000 / euc_dist

    return circuity_factor

def check_stops(stops):
    # Check and convert the coordinate reference system if needed.
    if stops.crs != 'EPSG:4547':
        stops = stops.to_crs(epsg=4547)

    # Create lon/lat fields if they are missing.
    if 'lon' not in stops.columns or 'lat' not in stops.columns:
        print("lon and lat fields were not found; generating them from EPSG:4326...")
        # Convert to EPSG:4326 to extract longitude and latitude.
        stops_4326 = stops.to_crs(epsg=4326)
        # Extract longitude and latitude.
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

    """Use the result directly when walking distance is short and no transfer is needed."""
    """Keep all crawled results and preserve time categories."""
    """Check ring routes or metro-direct cases later using average speed."""
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
        # if idx>5:  # for test
        #     break
        # if row['route_id'] != 900000054965:  # for test
        #     continue
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
        # Parse timetable strings into datetime objects.
        try:
            dt = datetime.strptime(t, '%H:%M')
        except:
            continue
        # Convert time values to minutes.
        total_minutes = dt.hour * 60 + dt.minute
        minutes.append(total_minutes)

    # Ensure the list is sorted.
    minutes.sort()

    # Compute adjacent intervals.
    intervals = [minutes[i + 1] - minutes[i] for i in range(len(minutes) - 1)]

    # Get the minimum and maximum interval.
    min_interval = min(intervals)
    max_interval = max(intervals)

    return min_interval, max_interval

def reduce_duration(row):
    min_itv, max_itv = get_interval(row['timetables'])
    # if row['name2crawl'] == '62':
    #     print('aaa')
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
    """
    In addition to filling the gap, the duration also includes waiting time, so half of the headway should be subtracted.
    :param routes: timetable after the duration crawling
    :return: updated route info
    """
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
                # print(f'{row["route_name"]} & {reverse_rows.iloc[0]['route_name']}') # for testing
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
                # print(f'{row["route_name"]} & {reverse_rows.iloc[0]['route_name']}') # for testing

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
    # routes_expanded['s_time'] = pd.to_datetime(date + ' ' + routes_expanded['timetables'],
    #                                               format='%Y-%m-%d %H:%M')
    routes_expanded['s_time'] = pd.to_datetime(
        date + ' ' + routes_expanded['timetables'],
        format='%Y-%m-%d %H:%M',
        errors='coerce'
    )
    # Drop rows with NaT in s_time.
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

if __name__ == '__main__':
    '''data crawling from amap api'''
    key = 'XXXXXXXX'
    date = '2025-1-20'