import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from haversine import haversine, Unit


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
    """Find and remove the closest point pair from o and d."""
    if not o or not d:
        return None, None, None
    # Compute all candidate point-pair distances.
    distances = [(haversine(swap_tuple(o_i), swap_tuple(d_j)), o_i, d_j) for o_i in o for d_j in d]  # km

    # Find the closest pair.
    min_distance, o_nearest, d_nearest = min(distances)

    # Remove the matched points from the lists.
    o.remove(o_nearest)
    d.remove(d_nearest)

    return min_distance, o_nearest, d_nearest

def vehicle_scheduling(schedule_df, minInterval=5, speed=25, line_name=None, s_coords=None, e_coords=None,
                       s_time=None, e_time=None, dispatch_distance=999999):
    """
    line_name: column name to indentify same route (allow different directions)
    s_coords, e_coords (essential): column name to identify coordinates of stop
    s_time, e_time (essential): column name to identify start and end time of each trip, datetime format
    speed: km/h speed with no passengers
    """
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
                d2o[d_point] = [o_point, round(distance*1.2/speed*60)]
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
    var = None
