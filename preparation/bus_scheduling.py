import argparse
import ast
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from haversine import haversine


class Line:
    def __init__(self,
                 lineId: int,
                 lineName: str,
                 lineCo: str,
                 lineDirection: str,
                 lineLength: float = None,  
                 lineEVRange: float = None,  
                 lineStops: list = None,
                 lineCoords: list = None,
                 lineBusPairShortestPath: list = None,  
                 lineHeadway: list = None,  
                 lineVelocity: list = None,  
                 lineDuration: list = None  
                 ):
        if lineVelocity is None:
            lineVelocity = [25] * 24
        elif len(lineVelocity) != 24:
            lineVelocity = [25] * 24

        
        self.lineId = lineId
        self.lineName = lineName
        self.lineCo = lineCo
        self.lineDirection = lineDirection
        self.lineStops = lineStops
        self.lineCoords = lineCoords  

        
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
            lineHeadway = [3600] * 10 + [45] * 5 + [3600] * 9  

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

        
        self.schedule = pd.DataFrame()

    def timetable_scheduling(self, year: int, month: int, day: int, ETA_schedule=None, openhour=None):
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

            c_start_time = start_datetime  
            c_hour = c_start_time.hour  
            schedule = []  
            while c_hour<24:  
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
    if not o or not d:
        return None, None, None  
    
    distances = [(haversine(swap_tuple(o_i), swap_tuple(d_j)), o_i, d_j) for o_i in o for d_j in d]  

    
    min_distance, o_nearest, d_nearest = min(distances)

    
    o.remove(o_nearest)
    d.remove(d_nearest)

    return min_distance, o_nearest, d_nearest


def vehicle_scheduling(schedule_df, minInterval=5, speed=25, line_name=None, s_coords=None, e_coords=None,
                       s_time=None, e_time=None, dispatch_distance=999999):
    VSParkingDF = []
    vehicle_no = 0
    dispatch_distance /= 1000
    for name, route_schedule in schedule_df.groupby(line_name, observed=True):
        route_schedule.sort_values(by=s_time, ascending=True, inplace=True)
        route_schedule.reset_index(drop=True, inplace=True)

        od_site = route_schedule.drop_duplicates(subset=[s_coords, e_coords])
        direction_num =len(od_site)
        d_site = od_site[e_coords].tolist()
        o_site = od_site[s_coords].tolist()

        d2o = {}
        while o_site and d_site:
            distance, o_point, d_point = find_and_remove_nearest(o_site, d_site)
            if distance < dispatch_distance:  
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
                    if (np.isnan(route_schedule['vehicle_no'].iloc[n]) and
                            route_schedule[s_coords].iloc[n] == d2o[c_e_loc][0] and
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

    
    VSParkingDF.reset_index(drop=True, inplace=True)

    return VSParkingDF


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "amosa4CN"
SOURCE_ROOT = PROJECT_ROOT.parent / "data" / "input" / "vs_parking_nodeid_original"
TARGET_ROOT = PROJECT_ROOT.parent / "data" / "input" / "vs_parking_nodeid"


def prepare_city(source: Path, target: Path) -> tuple[int, int]:
    with source.open("rb") as handle:
        encoding = "utf-8-sig" if handle.read(3) == b"\xef\xbb\xbf" else "utf-8"
    with source.open(encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        original_rows = list(reader)
    trips = pd.read_csv(source, dtype={"name2crawl": "string"}, low_memory=False)
    if trips.empty:
        raise ValueError(f"{source.name}: no trips")
    trips["_source_row"] = range(len(trips))
    for name in ("s_time", "e_time"):
        trips[name] = pd.to_datetime(trips[name], format="mixed")
    for name in ("orientation_coords", "destination_coords"):
        trips[name] = trips[name].map(ast.literal_eval)

    scheduled = vehicle_scheduling(
        trips,
        minInterval=5,
        speed=40,
        line_name="name2crawl",
        s_coords="orientation_coords",
        e_coords="destination_coords",
        s_time="s_time",
        e_time="e_time",
    )
    if len(scheduled) != len(trips):
        raise ValueError(f"{source.name}: trip count changed during scheduling")
    assignments = scheduled.set_index("_source_row")[["vehicle_no", "trip_no"]].sort_index()
    if not assignments.index.equals(pd.RangeIndex(len(original_rows))):
        raise ValueError(f"{source.name}: trip identity changed during scheduling")
    for row, (vehicle, trip) in zip(original_rows, assignments.itertuples(index=False, name=None)):
        row["v_name"] = str(int(vehicle))
        row["trip"] = str(int(trip))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(original_rows)
    temporary.replace(target)
    return len(trips), scheduled["vehicle_no"].nunique()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--cities", nargs="+")
    args = parser.parse_args()
    cities = args.cities or [path.stem for path in sorted(args.source_root.glob("*.csv"))]
    for city in cities:
        trip_count, vehicle_count = prepare_city(
            args.source_root / f"{city}.csv", args.target_root / f"{city}.csv"
        )
        print(f"{city}: {trip_count} trips, {vehicle_count} vehicles", flush=True)


if __name__ == "__main__":
    main()




