"""
@Project ：HKbus
@File    ：station.py
@IDE     ：PyCharm 
@Author  ：Zili Tian
@Date    ：12/22/2024
"""
from datetime import datetime

import pandas as pd


def extra_timesave(v_name_list, dist_list, timeout_list, v_type_list, range_large, range_medium, range_small):
    """
    Vehicle scheduling based on parking duration and timeout conditions

    Args:
    vs_parking_df (pd.DataFrame): DataFrame containing vehicle parking information
    timeout_list (List[datetime]): List of datetime objects representing timeout conditions
    extra_bus (int): Number of extra buses to be scheduled
    extra_minibus (int): Number of extra minibuses to be scheduled
    distance:m ; range: km
    Returns:
    saved_time
    """
    '''从timeout由高到低选。每选一个，这辆车后来不超过续航里程的班次都可以被新车代替'''
    survived_indices = [i for i, t in enumerate(timeout_list) if t > 0]
    v_name_list = [v_name_list[i] for i in survived_indices]
    dist_list = [dist_list[i] for i in survived_indices]
    timeout_list = [timeout_list[i] for i in survived_indices]
    v_type_list = [v_type_list[i] for i in survived_indices]

    sorted_indices = sorted(range(len(timeout_list)), key=lambda i: timeout_list[i], reverse=True)
    saved_time_all = 0
    sub_idx = []
    extra_large, extra_medium, extra_small = 0,0,0
    for rank in range(len(sorted_indices)):
        idx = sorted_indices.index(rank)
        if v_name_list[idx]==-1:
            continue
        if timeout_list[idx]:
            if v_type_list[idx] == 'large':
                extra_large += 1
                range_state = range_large * 1000
            elif v_type_list[idx] == 'medium':
                extra_medium += 1
                range_state = range_medium * 1000
            else:
                extra_small += 1
                range_state = range_small * 1000
            # 找到 v_name[idx:] 中等于 v_name[idx] 的所有索引
            samevehi_indices = [i for i in range(idx, len(v_name_list)) if v_name_list[i] == v_name_list[idx]]

            for idx2 in samevehi_indices:
                if dist_list[idx2]<=range_state:
                    range_state-=dist_list[idx2]
                    v_name_list[idx2] = -1
                    saved_time_all += timeout_list[idx2]
                    timeout_list[idx2]=0
                    sub_idx.append(survived_indices[idx2])
                else:
                    break
        if all(t <= 0.05 for t in timeout_list):  # all timeout < 3mins
            break

    return saved_time_all, sub_idx, extra_large, extra_medium, extra_small

if __name__ == '__main__':
    test = pd.read_csv(r"E:\Manufacture\Python\hkbus\data\input\formosa\test4extra.csv")
    test_timeout_list = test['timeout'].to_list()
    v_name_list = test['v_name'].to_list()
    dist_list = test['distance'].to_list()

    saved_time, subs_idx, extra_bus_num = extra_timesave(v_name_list,dist_list,test_timeout_list,500)
    print(sum(test_timeout_list)-saved_time)
    print(extra_bus_num)
