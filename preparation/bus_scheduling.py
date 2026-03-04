"""
Author: Zili Tian
Date: 2024/12/10
Description:
    This script is used to generate the bus schedule as key input data of test4mosa.py.
    it can calculate the driving range considering road grade
    it can not only use ETA data, but also generate by using frequency and operating time
"""

import numpy as np
import pandas as pd
from functools import partial
from datetime import datetime, timedelta
import math
import ast
from haversine import haversine, Unit
from itertools import islice
import pickle


class Line:
    def __init__(self,
                 lineId: int,
                 lineName: str,
                 lineCo: str,
                 lineDirection: str,
                 lineLength: float = None,  # km
                 lineEVRange: float = None,  # km
                 lineStops: list = None,
                 lineCoords: list = None,
                 lineBusPairShortestPath: list = None,  # km
                 lineHeadway: list = None,  # min
                 lineVelocity: list = None,  # km/h
                 lineDuration: list = None  # min
                 ):
        if lineVelocity is None:
            lineVelocity = [25] * 24
        elif len(lineVelocity) != 24:
            lineVelocity = [25] * 24

        # compulsory input
        self.lineId = lineId
        self.lineName = lineName
        self.lineCo = lineCo
        self.lineDirection = lineDirection
        self.lineStops = lineStops
        self.lineCoords = lineCoords  # convert string to list

        # other inputs using for calculating the duration of every section between two adjoined stops
        self.lineLength = lineLength
        self.lineEVRange = lineEVRange
        self.lineBusPairShortestPath = lineBusPairShortestPath
        if lineHeadway is not None:
            if lineHeadway == ['Single departure']:
                self.lineHeadway = 'Single departure'
            else:
                lineHeadway = [int(num) for num in lineHeadway]
                if len(lineHeadway) == 1:
                    self.lineHeadway = lineHeadway * 24
                elif len(lineHeadway) == 2:
                    self.lineHeadway = [lineHeadway[1] if i in [6, 7, 16, 17] else lineHeadway[0] for i in range(24)]
                elif len(lineHeadway) != 24:
                    lineHeadway = [120] * 5 + [15] * 18 + [120]
        else:
            lineHeadway = [3600] * 10 + [45] * 5 + [3600] * 9  # for small scale testing

        self.lineVelocity = lineVelocity
        if lineDuration is not None:
            if len(lineDuration) == 1:
                self.lineDuration = lineDuration * 24
            elif len(lineDuration) != 24:
                lineDuration = None
        if lineDuration is not None:
            self.lineDuration = [math.ceil(num) for num in self.lineDuration]
        elif lineLength is not None:
            length_with_conversion = lineLength * 60
            self.lineDuration = [math.ceil(length_with_conversion / value) for value in lineVelocity]
        else:
            self.lineDuration = [30] * 24
            print(f"【{self.lineCo} {self.lineName} to {self.lineDirection}】's duration has been assumed")

        # need to be generated
        self.schedule = pd.DataFrame()

    def timetable_scheduling(self, year: int, month: int, day: int, ETA_schedule=None, openhour=None):
        """for example: 2024 7 15
        openhour format str： 8:10:00 or 5:35 AM - 11:40 PM"""
        if ETA_schedule is not None:
            ETA_schedule['avg_velocity'] = self.lineLength / ETA_schedule['duration'] * 60
            ETA_schedule['duration'] = round(ETA_schedule['duration']).astype(int)
            ETA_schedule['avg_velocity'] = round(ETA_schedule['avg_velocity']).astype(int)
            schedule = ETA_schedule[['s_time', 'e_time', 'avg_velocity', 'duration']]
        elif openhour is not None and self.lineHeadway == 'Single departure':
            schedule = []
            try:
                c_start_time = datetime.strptime(openhour, '%H:%M:%S')
                c_start_time = c_start_time.replace(year=year, month=month, day=day)
                c_end_time = c_start_time + timedelta(minutes=self.lineDuration[0])
                schedule.append([c_start_time, c_end_time, self.lineLength/(self.lineDuration[0]/60),
                                 self.lineDuration[0]])
                schedule = pd.DataFrame(schedule, columns=['s_time', 'e_time', 'avg_velocity', 'duration'])
            except:
                return False
        else:
            if openhour is not None:
                start_time_str, end_time_str = openhour.split(' - ')
                start_datetime = datetime.strptime(start_time_str, '%I:%M %p')
                end_datetime = datetime.strptime(end_time_str, '%I:%M %p')
                start_datetime = start_datetime.replace(year=year, month=month, day=day)
                end_datetime = end_datetime.replace(year=year, month=month, day=day)
            else:
                start_datetime = datetime(year, month, day)
                end_datetime = start_datetime + timedelta(days=1)

            c_start_time = start_datetime  # current s_time
            c_hour = c_start_time.hour  # current hour
            schedule = []  # store the start time, end time, velocity and duration
            while c_hour<24:  # break if circle to the next day
                if self.lineHeadway[c_hour] <= 3600:
                    schedule.append([c_start_time, c_start_time + timedelta(minutes=self.lineDuration[c_hour]),
                                     self.lineLength/(self.lineDuration[c_hour]/60), self.lineDuration[c_hour]])
                    c_start_time = c_start_time + timedelta(minutes=self.lineHeadway[c_hour])
                else:
                    c_start_time = c_start_time + timedelta(minutes=60)
                if c_start_time > end_datetime:
                    break
                c_hour = c_start_time.hour
            schedule = pd.DataFrame(schedule, columns=['s_time', 'e_time', 'avg_velocity', 'duration'])

        if self.lineLength is not None:
            schedule['distance'] = self.lineLength
            schedule['evrange'] = self.lineEVRange
        else:
            schedule['distance'] = ''
        schedule['lineName'] = self.lineName
        schedule['lineCo'] = self.lineCo
        schedule['lineDirection'] = self.lineDirection
        schedule['destination_coords'] = [self.lineCoords[-1]] * len(schedule)
        schedule['orientation_coords'] = [self.lineCoords[0]] * len(schedule)
        self.schedule = schedule


def find_duplicates(lineDict):
    duplicates_by_key = {}
    for key, line in lineDict.items():
        identifier = (line.lineCo, line.lineName)
        if identifier not in duplicates_by_key:
            duplicates_by_key[identifier] = [key]
        else:
            duplicates_by_key[identifier].append(key)
    return list(duplicates_by_key.values())


def swap_tuple(t):
    return t[1], t[0]


def find_and_remove_nearest(o, d):
    """找到o和d中最接近的一对点并移除，返回距离和这对点"""
    if not o or not d:
        return None, None, None  # 如果任一列表为空，则返回None
    # 计算所有可能的点对距离
    distances = [(haversine(swap_tuple(o_i), swap_tuple(d_j)), o_i, d_j) for o_i in o for d_j in d]  # km

    # 找到最小距离的点对
    min_distance, o_nearest, d_nearest = min(distances)

    # 从列表中移除已匹配的点
    o.remove(o_nearest)
    d.remove(d_nearest)

    return min_distance, o_nearest, d_nearest


def vehicle_scheduling(schedule_df, minInterval=5, speed=25, line_name=None, s_coords=None, e_coords=None,
                       s_time=None, e_time=None, dispatch_distance=999999):
    """分配车辆，不考虑跨线调度，到达和下次出发时间间隔超过t但最短的即可匹配为一辆车
    可能存在单向、双向线路和三向（A to B to C to A）
    一般能闭环，无法闭环就按速度V回到首站
    line_name: column name to indentify same route (allow different directions)
    s_coords, e_coords (essential): column name to identify coordinates of stop
    s_time, e_time (essential): column name to identify start and end time of each trip, datetime format
    speed: km/h speed with no passengers"""
    VSParkingDF = []
    vehicle_no = 0
    dispatch_distance /= 1000
    for name, route_schedule in schedule_df.groupby(line_name):
        route_schedule.sort_values(by=s_time, ascending=True, inplace=True)
        route_schedule.reset_index(drop=True, inplace=True)

        od_site = route_schedule.drop_duplicates(subset=[s_coords, e_coords])
        direction_num =len(od_site)
        d_site = od_site[e_coords].tolist()
        o_site = od_site[s_coords].tolist()

        d2o = {}
        while o_site and d_site:
            distance, o_point, d_point = find_and_remove_nearest(o_site, d_site)
            if distance < dispatch_distance:  # meters
                d2o[d_point] = [o_point, round(distance*1.2/speed*60)]  # 空载duration，后面用最短路替换
            else:
                break

        route_schedule['vehicle_no'] = np.nan
        route_schedule['trip_no'] = np.nan
        for m in range(len(route_schedule)):
            if np.isnan(route_schedule['vehicle_no'].iloc[m]):
                vehicle_no += 1
                trip_no = 1
                route_schedule.loc[m, 'trip_no'] = trip_no
                route_schedule.loc[m, 'vehicle_no'] = vehicle_no
                c_e_time = route_schedule[e_time].iloc[m]
                c_e_loc = route_schedule[e_coords].iloc[m]
                for n in range(m+1, len(route_schedule)):
                    if (route_schedule[s_coords].iloc[n] == d2o[c_e_loc][0] and
                            route_schedule[s_time].iloc[n] - c_e_time >= timedelta(minutes=minInterval+d2o[c_e_loc][1])):
                        trip_no += 1
                        route_schedule.loc[n, 'trip_no'] = trip_no
                        route_schedule.loc[n, 'vehicle_no'] = vehicle_no
                        c_e_time = route_schedule[e_time].iloc[n]
                        c_e_loc = route_schedule[e_coords].iloc[n]
                        continue
        route_schedule['vehicle_no'] = route_schedule['vehicle_no'].astype(int)
        route_schedule['trip_no'] = route_schedule['trip_no'].astype(int)

        if len(VSParkingDF):
            VSParkingDF = pd.concat([VSParkingDF, route_schedule], axis=0)
        else:
            VSParkingDF = route_schedule.copy(deep=False)

    # VSParkingDF.sort_values(by=e_time, ascending=True, inplace=True)
    VSParkingDF.reset_index(drop=True, inplace=True)

    return VSParkingDF


if __name__ == '__main__':
    YEAR, MONTH, DAY = 2024, 9, 18
    # # input file
    # line_info is from match_stop2network, containing evrange
    line_info = pd.read_csv(r"E:\Manufacture\Python\hkbus\data\HKsimpreparation\past\202412-2\line_info.csv")  # without minibus

    # ETA_data is the output of eta_preproscessing.py
    # ETA_data = pd.read_csv(r"E:\STUDY\TIP\HKbus\data\HKsimpreparation\fbus_eta240918.csv",nrows=9547) # for test
    ETA_data = pd.read_csv(r"E:\STUDY\TIP\HKbus\data\HKsimpreparation\fbus_eta240918.csv")

    '''step 1: route with ETA'''
    ETA_route_list = ETA_data.drop_duplicates(subset=['co', 'route', 'dir', 'dest'])
    ETA_route_list = ETA_route_list[['co', 'route', 'dir', 'dest']]
    ETA_route_list = ETA_route_list.reset_index(drop=True)

    ETA_data.drop_duplicates(inplace=True)
    ETA_data['eta'] = pd.to_datetime(ETA_data['eta'])
    ETA_data['timestamp'] = pd.to_datetime(ETA_data['timestamp'])

    eta_lines = []
    failed_eta_lineinfo = []
    for idx, row in ETA_route_list.iterrows():
        print(f"Index: {idx}, Row data: {row.to_dict()}")
        sub_ETA_data = ETA_data[(ETA_data['co']==row['co'])&(ETA_data['route']==row['route'])&
                                (ETA_data['dir']==row['dir'])&(ETA_data['dest']==row['dest'])]

        # 令站点顺序连续
        unique_sorted_values = sub_ETA_data['seq'].unique()
        unique_sorted_values.sort()
        value_to_rank = {value: rank + 1 for rank, value in enumerate(unique_sorted_values)}
        sub_ETA_data['seq'] = sub_ETA_data['seq'].map(value_to_rank)

        sub_line_info = line_info[(line_info['co']==row['co'])&(line_info['lineName']==row['route'])&
                                (line_info['lineDirection']==row['dir'])].reset_index(drop=True)

        if len(sub_line_info) == 0:
            continue

        sub_ETA_data = sub_ETA_data.reset_index(drop=True)
        sub_schedule = eta2schedule(sub_ETA_data, retain_unmatched=False)

        if len(sub_schedule) == 0:
            continue

        # 发车时刻(seq=1时的eta)相同的保留scraptime大的
        start_eta = sub_schedule[sub_schedule['seq']==1]

        start_eta = start_eta.sort_values(by='scrapped_time', ascending=False)
        start_eta = start_eta.drop_duplicates(subset='eta', keep='first')
        start_eta = start_eta['route_id'].to_list()

        sub_schedule = sub_schedule[sub_schedule['route_id'].isin(start_eta)]

        sub_schedule = sub_schedule.groupby('route_id').agg(
            min_eta=('eta', 'min'),
            max_eta=('eta', 'max')
        ).reset_index()
        sub_schedule['duration'] = (sub_schedule['max_eta'] - sub_schedule['min_eta']).dt.total_seconds() / 60
        sub_schedule.rename(columns={'min_eta': 's_time', 'max_eta': 'e_time'}, inplace=True)

        sub_schedule = sub_schedule[sub_schedule['duration'] > 0]
        if len(sub_schedule) == 0:
            continue

        i=0
        _line = Line(lineId=idx,  # No lineId in the raw data
                     lineName=sub_line_info['lineName'].iloc[i],
                     lineCo=sub_line_info['co'].iloc[i],
                     lineDirection=sub_line_info['lineDirection'].iloc[i],
                     lineLength=sub_line_info['lineLength'].iloc[i],
                     lineEVRange=sub_line_info['evrange'].iloc[i],
                     lineStops=sub_line_info['lineStops'].iloc[i],
                     lineCoords=ast.literal_eval(sub_line_info['lineCoords'].iloc[i]),
                     lineBusPairShortestPath=ast.literal_eval(sub_line_info['lineBusPairShortestPath'].iloc[i]))


        _line.timetable_scheduling(YEAR, MONTH, DAY, ETA_schedule=sub_schedule)

        eta_lines.append(_line)

        matched_list = []
        for l in eta_lines:
            matched_list.append([l.lineCo,l.lineName,l.lineDirection])
        matched_list = pd.DataFrame(matched_list, columns=['co','lineName','lineDirection'])
        matched_list['match'] = 1
        unmatched_list = pd.merge(line_info, matched_list,how='left',on=['co','lineName','lineDirection'])
        unmatched_list = unmatched_list[unmatched_list['match']!=unmatched_list['match']]

    with open('eta_lines_fbus.pkl', 'wb') as file:
        pickle.dump(eta_lines, file)

    '''step 2: process route without eta or unmatched ones'''
    # # 从文件中读取列表
    # with open('eta_lines.pkl', 'rb') as file:
    #     loaded_lines = pickle.load(file)
    #
    # # no_eta_route_freq includes routes with freq but without ETA from lineinfo,
    # # which is extracted from RoutestopWithETA.xlsx
    # # unmacthed routes in former eta process should also be included
    # no_eta_route_freq = pd.read_csv(r"E:\Manufacture\Python\hkbus\data\HKsimpreparation\no_eta_route_freq.csv")
    # no_eta_route_list = pd.merge(no_eta_route_freq, line_info, how='left', on=['co', 'lineName', 'lineDirection'])
    # no_eta_lines = []
    # for idx, row in no_eta_route_list.iterrows():
    #     _line = Line(lineId=idx+len(loaded_lines),  # No lineId in the raw data
    #                  lineName=row['lineName'],
    #                  lineCo=row['co'],
    #                  lineDirection=row['lineDirection'],
    #                  lineLength=row['lineLength'],
    #                  lineEVRange=row['evrange'],
    #                  lineStops=row['lineStops'],
    #                  lineCoords=ast.literal_eval(row['lineCoords']),
    #                  lineBusPairShortestPath=ast.literal_eval(row['lineBusPairShortestPath']),
    #                  lineHeadway=[row['FreqMin']],
    #                  lineDuration=[row['DuraMin']])
    #     _line.timetable_scheduling(YEAR, MONTH, DAY, openhour=row['OperHour'])
    #
    #     no_eta_lines.append(_line)
    #
    # lines = loaded_lines + no_eta_lines
    # all_schedule = []
    # for _line in lines:
    #     if len(all_schedule):
    #         all_schedule = pd.concat([all_schedule, _line.schedule], axis=0)
    #     else:
    #         all_schedule = _line.schedule.copy(deep=False)
    #
    # line_dict = {_line.lineId: _line for _line in lines}  # convert line_info list into dict
    #
    # vs_parking_df = vehicle_scheduling(line_dict, minInterval=5, hold_unknown=False)
    # vs_parking_df.to_csv(r"E:\STUDY\TIP\HKbus\data\HKsimpreparation\HK_all_vs_parking.csv", index=False)




