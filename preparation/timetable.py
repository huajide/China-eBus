import geopandas as gpd
from pyproj import CRS
import pandas as pd
import re
from datetime import datetime, timedelta
from sympy.printing.cxx import reserved


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
        # 找到name2crawl相同的行
        matching_rows = df[df['name2crawl'] == row['name2crawl']]
        # 找到这些行中s_time不为空的行
        non_empty_s_time = matching_rows['s_time'].dropna().unique()
        if len(non_empty_s_time) > 0:
            if non_empty_s_time[0]!='[]':
                return non_empty_s_time[0]
            else:
                # 找到名字不同但是一条线的
                matching_rows = df[(df['s_stop'] == row['e_stop'])&(df['e_stop'] == row['s_stop'])&
                                   (abs(df['route_id']-row['route_id'])==1)]
                non_empty_s_time = matching_rows['s_time'].dropna().unique()
                if len(non_empty_s_time) > 0:
                    if non_empty_s_time[0]!='[]':
                        return non_empty_s_time[0]
    return row['s_time']


def fill_empty_e_time(row, df):
    if row['e_time']=='[]':
        # 找到name2crawl相同的行
        matching_rows = df[df['name2crawl'] == row['name2crawl']]
        # 找到这些行中e_time不为空的行
        non_empty_e_time = matching_rows['e_time'].dropna().unique()
        if len(non_empty_e_time) > 0:
            if non_empty_e_time[0]!='[]':
                return non_empty_e_time[0]
            else:
                # 找到名字不同但是一条线的
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
    max_columns = 300  # 200 个发车时间
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
        # 读取表头
        header = f.readline().strip().split()
        # 读取数据行
        for line in f:
            fields = line.strip().split()
            if len(fields) == len(header):  # 确保字段数量匹配
                data.append(fields)
    df = pd.DataFrame(data, columns=header)
    return df


def match_chelaile(routes, chelaile):
    dnmc_info = []
    valid_values = ['无搜索结果', '无发车信息']
    route_list = routes['name2crawl'].drop_duplicates().tolist()
    for route in route_list:
        routes_slice = routes[routes['name2crawl'] == route]
        chelaile_slice = chelaile[chelaile['name2crawl'] == route]
        if len(chelaile_slice) == 0 or chelaile_slice['status'].isin(valid_values).all():
            for row in routes_slice.iterrows():
                dnmc_info.append([row[1]['route_id'], 'NoInfo', [], None, None])
        else:  # match by similarity
            candidates = []
            # 遍历所有可能的组合
            for idx1 in routes_slice.index:
                for idx2 in chelaile_slice.index:
                    # 计算起始站相似度
                    s_sim = common_substring_ratio(
                        str(routes_slice.loc[idx1, 's_stop']),
                        str(chelaile_slice.loc[idx2, 's_stop_app'])
                    )
                    # 计算终点站相似度
                    e_sim = common_substring_ratio(
                        str(routes_slice.loc[idx1, 'e_stop']),
                        str(chelaile_slice.loc[idx2, 'e_stop_app'])
                    )
                    # 综合相似度（可调整权重）
                    total_sim = s_sim + e_sim

                    candidates.append((-total_sim, idx1, idx2))  # 负号用于降序排序

            # 按相似度降序排序
            candidates.sort()

            # 初始化记录集合
            matched_routes = set()
            matched_chelaile = set()

            # 执行贪心匹配
            for sim, idx1, idx2 in candidates:
                if idx1 not in matched_routes and idx2 not in matched_chelaile:
                    dnmc_info.append([routes_slice.loc[idx1,'route_id'], 'Success',chelaile_slice.loc[idx2,'timetable'],
                                      chelaile_slice.loc[idx2,'route_app'],chelaile_slice.loc[idx2,'direction']])
                    matched_routes.add(idx1)
                    matched_chelaile.add(idx2)

                # 提前终止条件
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
    """计算两个字符串的最长公共子串长度占s1总长度的比例"""
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
    """更健壮的时间表提取，处理更多格式"""
    # 匹配第一个"定点发车"后的所有时间
    match = re.search(r'定点发车([\d:\s、,，]+)', text)
    if not match:
        return []

    # 提取所有可能的时间格式
    time_str = match.group(1)
    time_matches = re.findall(r'(\d{1,2})[:：](\d{2})', time_str.replace(' ', ''))

    # 标准化时间格式
    timetable = []
    for hour, minute in time_matches:
        hour = hour.lstrip('0') or '0'  # 处理0点的情况
        timetable.append(f"{hour}:{minute}")

    return timetable


def multi_period_timetable(text: str) -> list[str]:
    """
    从文本中提取时段信息并生成时刻表

    参数:
        text: 包含时段信息的字符串，格式示例：
            '06:00-13:00 35分钟/趟 13:00-15:00 40分钟/趟'

    返回:
        标准化格式的时刻表列表，如 ['6:00', '6:35', '7:10', ...]
    """
    # 使用正则表达式提取时间段和间隔
    pattern = r"""
        (\d{1,2})[:：](\d{2})    # 开始时间 (HH:mm)
        -                        # 分隔符
        (\d{1,2})[:：](\d{2})    # 结束时间 (HH:mm)
        \D*?                     # 非数字分隔
        (\d+)                    # 间隔分钟数
        (?:分钟/趟|min)          # 间隔单位
    """
    matches = re.findall(pattern, text, re.X)

    timetable = []

    for match in matches:
        # 解析时间段参数
        start_h, start_m, end_h, end_m, interval = match
        interval = int(interval)

        # 转换为总分钟数
        start = int(start_h) * 60 + int(start_m)
        end = int(end_h) * 60 + int(end_m)

        # 生成该时段时刻表
        current = start
        while current <= end:
            # 转换回时间格式
            hours = current // 60
            mins = current % 60

            # 标准化格式（去掉前导零）
            time_str = f"{hours}:{mins:02d}"
            timetable.append(time_str)

            current += interval

    # 去重、排序并返回
    return sorted(list(set(timetable)), key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])))


def parse_bus_interval(s: str):
    """
    根据题目要求：
    1. 如果出现“高峰X分钟/趟”，则取首次出现的X作为高峰间隔。
    2. 如果出现“平峰X分钟/趟”或“低峰X分钟/趟”，则取首次出现的X作为平峰间隔。
    （有时“平峰”会被写成“低峰”，在此视为同一类）
    3. 如果没有高峰，则高峰间隔与平峰相同。
    4. 如果没有平峰，则平峰间隔设为9999。
    5. 如果出现多个相同峰，只取第一次出现。
    """


    peak = None  # 高峰间隔
    off_peak = None  # 平峰/低峰间隔

    # 使用正则表达式，匹配 "高峰X分钟/趟"、"平峰X分钟/趟" 或 "低峰X分钟/趟"
    pattern = re.compile(r'(高峰|平峰|低峰)(\d+)分钟/趟')
    matches = pattern.findall(s)

    for m in matches:
        word, num_str = m
        num = int(num_str)

        # 首次出现高峰
        if (word == '高峰') and (peak is None):
            peak = num

        # 首次出现平峰或低峰
        if (word in ['平峰', '低峰']) and (off_peak is None):
            off_peak = num

    # 根据规则补全结果
    if peak is None and off_peak is None:
        # 两者都没找到，可根据业务需要自行处理，以下示例设为9999
        peak, off_peak = 9999, 9999
    elif peak is None and off_peak is not None:
        # 没有高峰，但有平峰
        # “如果没高峰，那高峰间隔与平峰相同”
        peak = off_peak
    elif peak is not None and off_peak is None:
        # 有高峰，但没有平峰
        # “如果有高峰没平峰，输出的时候平峰间隔为9999”
        off_peak = 9999

    return peak, off_peak


def general_timetable(
    start_time_str: str,
    end_time_str: str,
    off_peak_interval: int,
    peak_interval: int
    ) -> list:
    """
    根据 start_time_str 和 end_time_str (格式例如'06:45:00'、'18:30:00')，
    利用 off_peak_interval(平峰间隔, 单位: 分钟) 与 peak_interval(高峰间隔, 单位: 分钟)，
    返回发车时刻表(列表)，格式如 ['6:45', '7:15', '8:25']。


    注：高峰时段默认设为[7:00-9:00)与[17:00-19:00)。
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
        # 判定是否在高峰时段
        if (7 <= h < 9) or (17 <= h < 19):
            interval = peak_interval
        else:
            interval = off_peak_interval

        # 将当前时刻加入结果列表，格式去掉小时的前导 0
        schedule.append(f"{h}:{current_dt.minute:02d}")

        # 跳到下一趟车的时刻
        try:
            current_dt += timedelta(minutes=interval)
        except TypeError:
            print(interval)

    return schedule


def extract_interval(text: str) -> int:
    """
    从给定的文本中提取第一次出现的“分钟/趟”前面的间隔数字。
    如果未找到，返回 None。


    示例：
    1) "06:00-22:00 高峰19分钟/趟 低峰29分钟/趟" -> 19
    2) "高峰10分钟/趟 低峰15分钟/趟" -> 10
    3) "abcxyz" -> None
    """
    match = re.search(r'(\d+)\s*分钟/趟', text)
    if match:
        return int(match.group(1))
    else:
        return None


workdays = ['周一', '周二', '周三', '周四', '周五', '工作日']
weekends = ['周六', '周日', '周末', '节假日']
periods = ['高峰','平峰','低峰']


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
        if len(row['timetables']) or ('分钟/趟' not in static_info and '定点发车' not in static_info):
            continue

        # distinguish the weekend and workdays
        if any(day in static_info for day in weekends):
            if any(day in static_info for day in workdays):
                positions = [(static_info.find(day), day) for day in workdays if day in static_info]
                # 获取最早出现的工作日词及其位置
                first_pos, first_day = min(positions, key=lambda x: x[0])
                # 返回从第一个工作日词开始的内容
                static_info = static_info[first_pos:]
            else:
                continue  # routes only weekend are excluded

        # distinguish the fixed frequency and fixed time
        if '分钟/趟' in static_info:
            freq_count= static_info.count('分钟/趟')

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

        elif '定点发车' in static_info:
            static_tt = extract_fixed_timetable(static_info)
        else:
            static_tt = []

        dnmc_routes.at[idx, 'timetables'] = static_tt  # 用loc会报错，loc不能给一个unit赋值list，会长度不符


def fill_twin_timetable(dnmc_routes):
    """
    df: dnmc_routes
    如果某类name2crawl有信息的只有一个，找到最match的那个（如果一共两条直接认定），赋予相同的时刻表。
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
                # 遍历所有可能的组合
                for idx2 in routes_slice.index:
                    # 计算起始站相似度
                    s_sim = common_substring_ratio(
                        str(routes_slice.loc[idx_not_empty, 's_stop']),
                        str(routes_slice.loc[idx2, 'e_stop'])
                    )
                    # 计算终点站相似度
                    e_sim = common_substring_ratio(
                        str(routes_slice.loc[idx_not_empty, 'e_stop']),
                        str(routes_slice.loc[idx2, 's_stop'])
                    )
                    # 综合相似度（可调整权重）
                    total_sim = s_sim + e_sim

                    candidates.append((-total_sim, idx2))  # 负号用于降序排序

                # 按相似度降序排序
                candidates.sort()
                idx2 = candidates[0][1]
                dnmc_routes.at[idx2, 'timetables'] = dnmc_routes.at[idx_not_empty, 'timetables']


def check_length_diff(row):
    if '(' not in row['route_name']:
        return False
    left_part = row['route_name'].split('(')[0]  # 取左括号左边的部分
    left_length = len(left_part)  # 左括号左边的字符数
    name2crawl_length = len(row['name2crawl'])  # name2crawl 的字符数
    return left_length - name2crawl_length > 1


def clean_no_timetables(dnmc_routes):
    """
    剔除线路（共同点是无时刻表）：
    一类：app无信息+无timetables+同name2crawl有的线路有信息，并且非正线
    二类：timetables为空+timetable只包含周末无工作日
    三类：无timetables+机场大巴
    四类：app无信息+无timetables+站间距大于5公里
    五类：无timetables+无服务时间
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

    # 合并条件：timetables 长度为 0 且 route_stop 包含 weekends 字符但不含 workdays 字符
    rows_to_drop = reserved_routes[cond1 & cond2_weekends & cond3_no_workdays].index
    reserved_routes = reserved_routes.drop(rows_to_drop).drop(columns=['tt_length'])

    # type 3
    reserved_routes = reserved_routes[~((reserved_routes['timetables'].apply(len) == 0) &
                                        (reserved_routes['route_type'] == '机场大巴'))]

    # type 4
    reserved_routes['tt_length'] = reserved_routes['timetables'].apply(len)
    cond1 = reserved_routes['tt_length'] == 0

    reserved_routes['stop_count'] = reserved_routes['route_stop'].str.count(',') + 1
    reserved_routes['avg_distance'] = reserved_routes['distance'] / (reserved_routes['stop_count'] - 1)
    cond2 = reserved_routes['avg_distance'] > 5
    cond2 = cond2.fillna(False)  # 如果 stop_count == 1，avg_distance 为 inf，替换为 False

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
    """
    计算高峰时段和平峰时段的平均发车间隔

    参数:
        timetables: 时间表列表，包含格式为 'H:MM' 或 'HH:MM' 的时间字符串

    返回:
        tuple: (高峰平均间隔, 平峰平均间隔)，单位为分钟
    """
    from datetime import datetime, timedelta

    # 只保留符合 HH:mm 格式的字符串
    valid_time_pattern = re.compile(r'^\d{1,2}:\d{2}$')
    valid_times = [t for t in timetables if valid_time_pattern.match(t)]

    # 将时间字符串转换为 datetime 对象（处理跨天情况）
    time_objects = []
    base_date = datetime(2024, 1, 1)

    # 先按原始时间排序
    time_points = []
    for time_str in valid_times:
        hour, minute = map(int, time_str.split(':'))
        time_points.append((hour, minute, time_str))

    # 按小时和分钟排序
    time_points.sort(key=lambda x: (x[0], x[1]))

    # 处理跨天情况
    current_date = base_date
    previous_time = None

    for hour, minute, time_str in time_points:
        # 创建当前时间对象
        if hour > 23:
            hour -= 24
        time_obj = current_date.replace(hour=hour, minute=minute)

        # 如果当前时间比前一个时间小，说明已经跨天了
        if previous_time is not None and time_obj < previous_time:
            current_date += timedelta(days=1)
            time_obj = current_date.replace(hour=hour, minute=minute)

        time_objects.append((time_obj, hour))
        previous_time = time_obj

    # 分类时间点
    peak_times = []
    offpeak_times = []

    for time_obj, hour in time_objects:
        # 使用24小时制来判断时段
        actual_hour = time_obj.hour
        if (7 <= actual_hour < 9) or (17 <= actual_hour < 19):
            peak_times.append(time_obj)
        elif 9 <= actual_hour < 17:
            offpeak_times.append(time_obj)

    # 计算高峰平均间隔
    peak_interval = None
    if len(peak_times) > 1:
        # 计算相邻班次之间的时间间隔，但排除早高峰到晚高峰的间隔
        intervals = []
        for i in range(len(peak_times) - 1):
            # 检查是否跨越了早高峰到晚高峰的边界
            current_hour = peak_times[i].hour
            next_hour = peak_times[i + 1].hour

            # 如果当前是早高峰(7-9点)且下一个是晚高峰(17-19点)，且时间差超过6小时，则跳过这个间隔
            time_diff = (peak_times[i + 1] - peak_times[i]).total_seconds()
            if (7 <= current_hour < 9) and (17 <= next_hour < 19) and time_diff > 6 * 3600:
                continue

            interval = time_diff / 60
            intervals.append(interval)

        if intervals:  # 确保有有效的间隔
            peak_interval = sum(intervals) / len(intervals)
    elif len(peak_times) == 1:
        # 只有一班车时，使用班次数+1除以时段长度的方法
        peak_interval = 240 / (len(peak_times) + 1)  # 高峰时段总长为240分钟(4小时)

    # 计算平峰平均间隔
    offpeak_interval = None
    if len(offpeak_times) > 1:
        # 计算相邻班次之间的时间间隔
        intervals = []
        for i in range(len(offpeak_times) - 1):
            interval = (offpeak_times[i + 1] - offpeak_times[i]).total_seconds() / 60
            intervals.append(interval)

        if intervals:  # 确保有有效的间隔
            offpeak_interval = sum(intervals) / len(intervals)
    elif len(offpeak_times) == 1:
        # 只有一班车时，使用班次数+1除以时段长度的方法
        offpeak_interval = 480 / (len(offpeak_times) + 1)  # 平峰时段总长为14小时(840分钟)

    return peak_interval, offpeak_interval


if __name__ == '__main__':
    province_name = '山东省'
    city_name = '威海市'
    city_only = city_name.replace('市', '')

    """standardize service hour"""
    route_gdf = gpd.read_file(rf'../data/cnbusdata2024-2/{province_name}/{city_name}/{city_name}_route5.shp',
                              crs=CRS.from_epsg(4326))
    clean_service_hour(route_gdf)

    """match chelaile data"""
    chelaile_path = rf'../data/chelaile/{city_only}result.csv'
    chelaile_data = read_chelaile(chelaile_path)

    route_dnmc = match_chelaile(route_gdf,chelaile_data)

    """get the raw data to match the completed timetable info"""
    raw_path = f'../data/cnbusdata2024/{province_name}/{city_name}/{city_name}_线路.csv'
    raw_data = pd.read_csv(raw_path, encoding='gbk')
    raw_data.rename(columns={'公交id': 'route_id','运营时刻': 'timetable', '路过的公交站': 'route_stop'}, inplace=True)
    raw_data = raw_data[['route_id', 'timetable', 'route_stop']]

    route_dnmc = route_dnmc.drop(columns=['timetable', 'route_stop'])
    route_dnmc = pd.merge(route_dnmc, raw_data, on='route_id', how='left')
    print(f"Number of routes with dynamic timetables: {route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0]}")

    """generate static timetables (should be done after matching chelaile data)"""
    static_timetable(route_dnmc)
    print(f"Number of routes with timetables: {route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0]}")
    fill_twin_timetable(route_dnmc)
    print(f"Number of routes with timetables (adjusted): {route_dnmc[route_dnmc['timetables'].apply(len) > 0].shape[0]}")

    """delete 4 types of route"""
    route_reserved = clean_no_timetables(route_dnmc)

    """assume the timetables for those info-lacked routes"""
    route_reserved[['peak_interval', 'offpeak_interval']] = route_reserved['timetables'].apply(
        lambda x: pd.Series(extract_avg_intervals(x))
    )
    peak_avg = route_reserved['peak_interval'].mean(skipna=True)
    offpeak_avg = route_reserved['offpeak_interval'].mean(skipna=True)
    peak_avg_rounded = round(peak_avg)
    offpeak_avg_rounded = round(offpeak_avg)
    assume_timetable(route_reserved,off_peak_interval=offpeak_avg, peak_interval=peak_avg)
    route_reserved.drop(columns=['peak_interval', 'offpeak_interval'], inplace=True)