import pandas as pd
import re

def get_earlier_time(time_str):
    times = time_str.split('|')
    if len(times) == 1:
        return times[0]
    else:
        return min(times)

def get_later_time(time_str):
    times = time_str.split('|')
    if len(times) == 1:
        return times[0]
    else:
        return max(times)

def fill_empty_s_time(row, df):
    if row['s_time']=='[]':
        # Find rows with the same name2crawl value.
        matching_rows = df[df['name2crawl'] == row['name2crawl']]
        # Keep rows with non-empty s_time.
        non_empty_s_time = matching_rows['s_time'].dropna().unique()
        if len(non_empty_s_time) > 0:
            if non_empty_s_time[0]!='[]':
                return non_empty_s_time[0]
            else:
                # Match routes with different names but the same line.
                matching_rows = df[(df['s_stop'] == row['e_stop'])&(df['e_stop'] == row['s_stop'])&
                                   (abs(df['route_id']-row['route_id'])==1)]
                non_empty_s_time = matching_rows['s_time'].dropna().unique()
                if len(non_empty_s_time) > 0:
                    if non_empty_s_time[0]!='[]':
                        return non_empty_s_time[0]
    return row['s_time']

def fill_empty_e_time(row, df):
    if row['e_time']=='[]':
        # Find rows with the same name2crawl value.
        matching_rows = df[df['name2crawl'] == row['name2crawl']]
        # Keep rows with non-empty e_time.
        non_empty_e_time = matching_rows['e_time'].dropna().unique()
        if len(non_empty_e_time) > 0:
            if non_empty_e_time[0]!='[]':
                return non_empty_e_time[0]
            else:
                # Match routes with different names but the same line.
                matching_rows = df[(df['s_stop'] == row['e_stop'])&(df['e_stop'] == row['s_stop'])&
                                   (abs(df['route_id']-row['route_id'])==1)]
                non_empty_e_time = matching_rows['e_time'].dropna().unique()
                if len(non_empty_e_time) > 0:
                    if non_empty_e_time[0]!='[]':
                        return non_empty_e_time[0]
    return row['e_time']

def clean_service_hour(ROUTE_GDF):
    """
    :param ROUTE_GDF: route5.shp  key columns: s_time, e_time, route_id, s_stop, e_stop
    :return: update route_gdf directly
    """
    ROUTE_GDF['s_time'] = ROUTE_GDF['s_time'].apply(get_earlier_time)
    ROUTE_GDF['e_time'] = ROUTE_GDF['e_time'].apply(get_later_time)
    ROUTE_GDF['s_time'] = ROUTE_GDF.apply(lambda row: fill_empty_s_time(row, ROUTE_GDF), axis=1)
    ROUTE_GDF['e_time'] = ROUTE_GDF.apply(lambda row: fill_empty_e_time(row, ROUTE_GDF), axis=1)

def read_chelaile(path):
    max_columns = 300
    column_names = ['name2crawl', 'status', 'timestamp', 'route_app', 'direction'] + [f'time_{i}' for i in
                                                                                      range(1, max_columns + 1)]
    try:
        chelaile = pd.read_csv(path, names=column_names,header=None,engine='python')
    except UnicodeDecodeError:
        chelaile = pd.read_csv(path, names=column_names,header=None, sep=r'(?<!#)\s+(?!#)', encoding='gbk')
    chelaile['s_stop_app'] = chelaile['direction'].str.extract(r'^(.*?) # ')[0]
    chelaile['e_stop_app'] = chelaile['direction'].str.extract(r' # (.*)$')[0]
    chelaile['timetable'] = chelaile.iloc[:, 5:-2].apply(lambda row: row.dropna().tolist(), axis=1)
    chelaile = chelaile.drop(chelaile.columns[5:-3], axis=1)

    return chelaile

def read_space_separated_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read the header.
        header = f.readline().strip().split()
        # Read data rows.
        for line in f:
            fields = line.strip().split()
            if len(fields) == len(header):
                data.append(fields)
    df = pd.DataFrame(data, columns=header)
    return df

def match_chelaile(routes, chelaile):
    dnmc_info = []
    valid_values = ['No result', 'No information']
    route_list = routes['name2crawl'].drop_duplicates().tolist()
    for route in route_list:
        routes_slice = routes[routes['name2crawl'] == route]
        chelaile_slice = chelaile[chelaile['name2crawl'] == route]
        if len(chelaile_slice) == 0 or chelaile_slice['status'].isin(valid_values).all():
            for row in routes_slice.iterrows():
                dnmc_info.append([row[1]['route_id'], 'NoInfo', [], None, None])
        else:  # match by similarity
            candidates = []
            # Iterate over all candidate pairs.
            for idx1 in routes_slice.index:
                for idx2 in chelaile_slice.index:
                    # Compute start-stop similarity.
                    s_sim = common_substring_ratio(
                        str(routes_slice.loc[idx1, 's_stop']),
                        str(chelaile_slice.loc[idx2, 's_stop_app'])
                    )
                    # Compute terminal-stop similarity.
                    e_sim = common_substring_ratio(
                        str(routes_slice.loc[idx1, 'e_stop']),
                        str(chelaile_slice.loc[idx2, 'e_stop_app'])
                    )
                    # Combined similarity score.
                    total_sim = s_sim + e_sim

                    candidates.append((-total_sim, idx1, idx2))

            # Sort by similarity in descending order.
            candidates.sort()

            # Initialize matched record sets.
            matched_routes = set()
            matched_chelaile = set()

            # Run greedy matching.
            for sim, idx1, idx2 in candidates:
                if idx1 not in matched_routes and idx2 not in matched_chelaile:
                    dnmc_info.append([routes_slice.loc[idx1,'route_id'], 'Success',chelaile_slice.loc[idx2,'timetable'],
                                      chelaile_slice.loc[idx2,'route_app'],chelaile_slice.loc[idx2,'direction']])
                    matched_routes.add(idx1)
                    matched_chelaile.add(idx2)

                # Early stopping condition.
                if len(matched_chelaile) == len(chelaile_slice):
                    break

            if len(matched_routes) < len(routes_slice):
                for idx1 in routes_slice.index:
                    if idx1 not in matched_routes:
                        dnmc_info.append([routes_slice.loc[idx1,'route_id'], 'NoInfo', [], None, None])

    dnmc_info = pd.DataFrame(dnmc_info, columns=['route_id', 'status', 'timetables', 'route_app', 'direction'])
    dnmc_routes = pd.merge(routes, dnmc_info, on='route_id', how='left')
    return dnmc_routes

def common_substring_ratio(s1, s2):
    """Return the longest common substring ratio relative to s1."""
    max_length = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            length = 0
            while (i + length < len(s1) and j + length < len(s2) and
                   s1[i + length] == s2[j + length]):
                length += 1
            max_length = max(max_length, length)
    return max_length / len(s1)

def extract_fixed_timetable(text):
    """Extract fixed departure times from more timetable formats."""
    match = re.search(r'Fixed schedule([\d:\s、,，]+)', text)
    if not match:
        return []

    time_str = match.group(1)
    time_matches = re.findall(r'(\d{1,2})[:：](\d{2})', time_str.replace(' ', ''))

    timetable = []
    for hour, minute in time_matches:
        hour = hour.lstrip('0') or '0'
        timetable.append(f"{hour}:{minute}")

    return timetable

def multi_period_timetable(text: str) -> list[str]:
    """Extract departure times from interval-based service text."""
    # Extract time windows and intervals with regex.
    pattern = r"""
        (\d{1,2})[:：](\d{2})
        -
        (\d{1,2})[:：](\d{2})
        \D*?
        (\d+)
        (?:min)
    """
    matches = re.findall(pattern, text, re.X)

    timetable = []

    for match in matches:
        # Parse each time window.
        start_h, start_m, end_h, end_m, interval = match
        interval = int(interval)

        # Convert to minutes.
        start = int(start_h) * 60 + int(start_m)
        end = int(end_h) * 60 + int(end_m)

        # Generate departures for the current window.
        current = start
        while current <= end:
            # Convert back to time strings.
            hours = current // 60
            mins = current % 60

            # Normalize the output format.
            time_str = f"{hours}:{mins:02d}"
            timetable.append(time_str)

            current += interval

    # Deduplicate, sort, and return.
    return sorted(list(set(timetable)), key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])))

def parse_bus_interval(s: str):
    """Extract the first peak and off-peak intervals from text."""

    peak = None
    off_peak = None

    # Match peak and off-peak intervals from text.
    pattern = re.compile(r'(Peak|Off-peak)(\d+)min')
    matches = pattern.findall(s)

    for m in matches:
        word, num_str = m
        num = int(num_str)

        # Use the first peak interval found.
        if (word == 'Peak') and (peak is None):
            peak = num

        # Use the first off-peak interval found.
        if (word == 'Off-peak') and (off_peak is None):
            off_peak = num

    # Fill missing values according to the rules.
    if peak is None and off_peak is None:
        # If neither is found, use 9999 as a fallback.
        peak, off_peak = 9999, 9999
    elif peak is None and off_peak is not None:
        # If peak is missing, reuse the off-peak interval.

        peak = off_peak
    elif peak is not None and off_peak is None:
        # If off-peak is missing, use 9999.

        off_peak = 9999

    return peak, off_peak

def general_timetable(
    start_time_str: str,
    end_time_str: str,
    off_peak_interval: int,
    peak_interval: int
    ) -> list:
    """
    Based on start_time_str and end_time_str (formatted like '06:45:00' and '18:30:00'),
    use off_peak_interval (off-peak headway, in minutes) and peak_interval (peak headway, in minutes)
    to return a departure timetable (list), formatted like ['6:45', '7:15', '8:25'].

    Note: The default peak periods are set to [7:00-9:00) and [17:00-19:00).
    """

    from datetime import datetime, timedelta

    time_format = "%H:%M:%S"
    start_dt = datetime.strptime(start_time_str, time_format)
    try:
        end_dt = datetime.strptime(end_time_str, time_format)
    except ValueError:
        end_dt = datetime.strptime('23:59:59', time_format)

    schedule = []
    current_dt = start_dt

    while current_dt <= end_dt:
        h = current_dt.hour
        # Check whether the current time falls in peak hours.
        if (7 <= h < 9) or (17 <= h < 19):
            interval = peak_interval
        else:
            interval = off_peak_interval

        # Append the current departure time.
        schedule.append(f"{h}:{current_dt.minute:02d}")

        # Move to the next departure time.
        try:
            current_dt += timedelta(minutes=interval)
        except TypeError:
            print(interval)

    return schedule

def extract_interval(text: str) -> int:
    """
    Extract the interval value preceding the first occurrence of "min" from the given text.
    Return None if no match is found.
    """
    match = re.search(r'(\d+)\s*min', text)
    if match:
        return int(match.group(1))
    else:
        return None

workdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Weekday']
weekends = ['Saturday', 'Sunday', 'Weekend']
periods = ['Peak','Off-peak']

def static_timetable(dnmc_routes):
    """
    should be done after 'match_chelaile'
    :param dnmc_routes: timetables, timetable columns are needed
    :return: dnmc_routes with timetable updated
    """
    global workdays, weekends, periods

    for idx, row in dnmc_routes.iterrows():
        static_info = row['timetable']
        if not isinstance(static_info, str):
            continue
        if len(row['timetables']) or ('min' not in static_info and 'Fixed schedule' not in static_info):
            continue

        # distinguish the weekend and workdays
        if any(day in static_info for day in weekends):
            if any(day in static_info for day in workdays):
                positions = [(static_info.find(day), day) for day in workdays if day in static_info]
                # Find the earliest workday keyword.
                first_pos, first_day = min(positions, key=lambda x: x[0])
                # Keep content starting from the first workday keyword.
                static_info = static_info[first_pos:]
            else:
                continue  # routes only weekend are excluded

        # distinguish the fixed frequency and fixed time
        if 'min' in static_info:
            freq_count= static_info.count('min')

            if any(p in static_info for p in periods):
                pk_freq, opk_freq = parse_bus_interval(static_info)
                static_tt = general_timetable(
                    start_time_str=row['s_time'],
                    end_time_str=row['e_time'],
                    off_peak_interval=opk_freq,
                    peak_interval=pk_freq
                )
            elif freq_count > 1:
                static_tt = multi_period_timetable(static_info)
            else:
                freq = extract_interval(static_info)
                static_tt = general_timetable(
                    start_time_str=row['s_time'],
                    end_time_str=row['e_time'],
                    off_peak_interval=freq,
                    peak_interval=freq
                )

        elif 'Fixed schedule' in static_info:
            static_tt = extract_fixed_timetable(static_info)
        else:
            static_tt = []

        dnmc_routes.at[idx, 'timetables'] = static_tt

def fill_twin_timetable(dnmc_routes):
    """
    Assign the same timetable to the best-matching route within each name2crawl category.

    df: dnmc_routes
    If only one route in a given name2crawl category has timetable information,
    find the best-matching route (or assign it directly if there are only two routes in total)
    and copy the same timetable.
    """
    route_list = dnmc_routes['name2crawl'].drop_duplicates().tolist()
    for route in route_list:
        routes_slice = dnmc_routes[dnmc_routes['name2crawl'] == route]
        if (len(routes_slice) <= 1 or all(routes_slice['timetables'].apply(len) > 0) or
                all(routes_slice['timetables'].apply(len) == 0)):
            continue

        if len(routes_slice) == 2:
            idx_not_empty = routes_slice[routes_slice['timetables'].apply(len) > 0].index[0]
            idx_empty = routes_slice[routes_slice['timetables'].apply(len) == 0].index[0]
            dnmc_routes.at[idx_empty, 'timetables'] = dnmc_routes.at[idx_not_empty, 'timetables']
        else:
            valid_pairs = []
            for idx1, row1 in routes_slice.iterrows():
                for idx2, row2 in routes_slice.iterrows():
                    if idx1 != idx2 and row1['s_stop'] == row2['e_stop'] and row1['e_stop'] == row2['s_stop']:
                        if len(row1['timetables']) > 0 and len(row2['timetables']) == 0:
                            valid_pairs.append((idx1, idx2))
                            # print(idx1, idx2)
            for idx1, idx2 in valid_pairs:
                dnmc_routes.at[idx2, 'timetables'] = dnmc_routes.at[idx1, 'timetables']

            count_not_empty = routes_slice[routes_slice['timetables'].apply(len) > 0].shape[0]
            if len(valid_pairs) == 0 and count_not_empty == 1:
                idx_not_empty = routes_slice[routes_slice['timetables'].apply(len) > 0].index[0]
                candidates = []
                # Iterate over all candidate pairs.
                for idx2 in routes_slice.index:
                    # Compute start-stop similarity.
                    s_sim = common_substring_ratio(
                        str(routes_slice.loc[idx_not_empty, 's_stop']),
                        str(routes_slice.loc[idx2, 'e_stop'])
                    )
                    # Compute terminal-stop similarity.
                    e_sim = common_substring_ratio(
                        str(routes_slice.loc[idx_not_empty, 'e_stop']),
                        str(routes_slice.loc[idx2, 's_stop'])
                    )
                    # Combined similarity score.
                    total_sim = s_sim + e_sim

                    candidates.append((-total_sim, idx2))

                # Sort by similarity in descending order.
                candidates.sort()
                idx2 = candidates[0][1]
                dnmc_routes.at[idx2, 'timetables'] = dnmc_routes.at[idx_not_empty, 'timetables']

def check_length_diff(row):
    if '(' not in row['route_name']:
        return False
    left_part = row['route_name'].split('(')[0]
    left_length = len(left_part)
    name2crawl_length = len(row['name2crawl'])
    return left_length - name2crawl_length > 1

def clean_no_timetables(dnmc_routes):
    """
    Remove routes that do not have timetable information.

    This includes the following cases:
    Type 1: No app information, no timetables, other routes with the same name2crawl have information, and the route is not a main line.
    Type 2: Empty timetables, and the timetable contains only weekend service with no weekday service.
    Type 3: No timetables and the route is an airport shuttle.
    Type 4: No app information, no timetables, and stop spacing is greater than 5 km.
    Type 5: No timetables and no service hours.

    :param dnmc_routes:
    :return:
    """
    reserved_routes = dnmc_routes.copy()
    global workdays, weekends

    # type 1
    reserved_routes['tt_length'] = reserved_routes['timetables'].apply(len)
    cond1_2 = (reserved_routes['tt_length'] == 0) & (reserved_routes['status'] == 'NoInfo')

    non_empty_counts = reserved_routes[reserved_routes['tt_length'] > 0].groupby('name2crawl').size()
    cond3 = reserved_routes['name2crawl'].map(non_empty_counts).fillna(0) >= 2

    cond4 = reserved_routes['route_name'].str.count(r'\(') == 1
    cond5 = reserved_routes.apply(check_length_diff, axis=1)

    rows_to_drop = reserved_routes[cond1_2 & cond3 & cond4 & cond5].index
    reserved_routes = reserved_routes.drop(rows_to_drop).drop(columns=['tt_length'])

    # type 2
    reserved_routes['tt_length'] = reserved_routes['timetables'].apply(len)
    cond1 = reserved_routes['tt_length'] == 0

    cond2_weekends = reserved_routes['route_stop'].apply(
        lambda x: any(w in str(x) for w in weekends) if pd.notnull(x) else False
    )
    cond3_no_workdays = ~reserved_routes['route_stop'].apply(
        lambda x: any(w in str(x) for w in workdays) if pd.notnull(x) else False
    )

    # Condition for routes with weekend-only stop text.
    rows_to_drop = reserved_routes[cond1 & cond2_weekends & cond3_no_workdays].index
    reserved_routes = reserved_routes.drop(rows_to_drop).drop(columns=['tt_length'])

    # type 3
    reserved_routes = reserved_routes[~((reserved_routes['timetables'].apply(len) == 0) &
                                        (reserved_routes['route_type'] == 'Airport'))]

    # type 4
    reserved_routes['tt_length'] = reserved_routes['timetables'].apply(len)
    cond1 = reserved_routes['tt_length'] == 0

    reserved_routes['stop_count'] = reserved_routes['route_stop'].str.count(',') + 1
    reserved_routes['avg_distance'] = reserved_routes['distance'] / (reserved_routes['stop_count'] - 1)
    cond2 = reserved_routes['avg_distance'] > 5
    cond2 = cond2.fillna(False)

    rows_to_drop = reserved_routes[cond1 & cond2].index
    reserved_routes = reserved_routes.drop(rows_to_drop).drop(columns=['tt_length', 'stop_count', 'avg_distance'])

    # type 5
    reserved_routes = reserved_routes[~((reserved_routes['timetables'].apply(len) == 0) &
                                        ((reserved_routes['s_time'] == '[]') | (reserved_routes['e_time'] == '[]')))]

    return reserved_routes

def assume_timetable(dnmc_routes, off_peak_interval=120, peak_interval=60):
    for idx, row in dnmc_routes.iterrows():
        if len(row['timetables']) == 0:
            static_tt = general_timetable(
                start_time_str=row['s_time'],
                end_time_str=row['e_time'],
                off_peak_interval=off_peak_interval,
                peak_interval=peak_interval
            )
            dnmc_routes.at[idx, 'timetables'] = static_tt

def extract_avg_intervals(timetables):
    """Compute average peak and off-peak headways from a timetable list."""
    from datetime import datetime, timedelta

    # Keep valid HH:mm strings only.
    valid_time_pattern = re.compile(r'^\d{1,2}:\d{2}$')
    valid_times = [t for t in timetables if valid_time_pattern.match(t)]

    # Convert strings to datetime values and handle overnight service.
    time_objects = []
    base_date = datetime(2024, 1, 1)

    # Sort by the original timetable order.
    time_points = []
    for time_str in valid_times:
        hour, minute = map(int, time_str.split(':'))
        time_points.append((hour, minute, time_str))

    # Sort by hour and minute.
    time_points.sort(key=lambda x: (x[0], x[1]))

    # Handle cross-day schedules.
    current_date = base_date
    previous_time = None

    for hour, minute, time_str in time_points:
        # Create the current datetime object.
        if hour > 23:
            hour -= 24
        time_obj = current_date.replace(hour=hour, minute=minute)

        # If time decreases, treat it as the next day.
        if previous_time is not None and time_obj < previous_time:
            current_date += timedelta(days=1)
            time_obj = current_date.replace(hour=hour, minute=minute)

        time_objects.append((time_obj, hour))
        previous_time = time_obj

    # Split times into peak and off-peak groups.
    peak_times = []
    offpeak_times = []

    for time_obj, hour in time_objects:
        # Use 24-hour time to classify periods.
        actual_hour = time_obj.hour
        if (7 <= actual_hour < 9) or (17 <= actual_hour < 19):
            peak_times.append(time_obj)
        elif 9 <= actual_hour < 17:
            offpeak_times.append(time_obj)

    # Compute the average peak interval.
    peak_interval = None
    if len(peak_times) > 1:
        # Compute adjacent intervals while excluding long gaps between peak windows.
        intervals = []
        for i in range(len(peak_times) - 1):
            # Check whether the interval crosses peak windows.
            current_hour = peak_times[i].hour
            next_hour = peak_times[i + 1].hour

            # Skip long gaps between morning and evening peak windows.
            time_diff = (peak_times[i + 1] - peak_times[i]).total_seconds()
            if (7 <= current_hour < 9) and (17 <= next_hour < 19) and time_diff > 6 * 3600:
                continue

            interval = time_diff / 60
            intervals.append(interval)

        if intervals:
            peak_interval = sum(intervals) / len(intervals)
    elif len(peak_times) == 1:
        # Use the window length divided by departures plus one when only one trip is available.
        peak_interval = 240 / (len(peak_times) + 1)

    # Compute the average off-peak interval.
    offpeak_interval = None
    if len(offpeak_times) > 1:
        # Compute adjacent intervals.
        intervals = []
        for i in range(len(offpeak_times) - 1):
            interval = (offpeak_times[i + 1] - offpeak_times[i]).total_seconds() / 60
            intervals.append(interval)

        if intervals:
            offpeak_interval = sum(intervals) / len(intervals)
    elif len(offpeak_times) == 1:
        # Use the window length divided by departures plus one when only one trip is available.
        offpeak_interval = 480 / (len(offpeak_times) + 1)

    return peak_interval, offpeak_interval

if __name__ == '__main__':
    province_name = 'XXX'
    city_name = 'XXX'