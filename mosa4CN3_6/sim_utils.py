"""
@Project ：DissertationPY 
@File    ：sim_utils.py
@IDE     ：PyCharm 
@Author  ：Xander PENG
@Date    ：11/8/2022 01:55 
"""
import numpy as np
from treelib import Node
import random
from geopandas import GeoDataFrame
from pandas import DataFrame


def find_keys(dict_input: dict, **kwargs):
    """
    Find the max_value or min_value items' keys in a dict

    :param dict_input: the input dict
    :param kwargs: specify max or min
    :return: A list of keys
    """
    keys = []

    if kwargs.get('value') == 'min':
        min_value = dict_input.get(min(dict_input))
        for k, v in dict_input.items():
            if v == min_value:
                keys.append(k)

        return keys

    elif kwargs.get('value') == 'max':
        max_value = dict_input.get(max(dict_input))
        for k, v in dict_input.items():
            if v == max_value:
                keys.append(k)

        return keys

    else:
        raise ValueError('Plz specify value type')


def set_node(last_node: Node, jth_trip: int, cs_id: int = 0, cd=0, **kwargs):
    if cd == 0:  # When there is no charging demand and behavior

        if kwargs.get('para') == 'tag':
            current_tag = 't' + str(jth_trip) + 's0'

            return current_tag

        elif kwargs.get('para') == 'id':
            last_id = last_node.identifier
            current_id = [i for i in last_id]
            current_id.append(0)
            current_id = tuple(current_id)

            return current_id

        elif kwargs.get('para') == 'parent':
            parent_id = last_node.identifier

            return parent_id

        else:
            raise ValueError('Plz specify para value')

    elif cd == 1:

        if kwargs.get('para') == 'tag':
            current_tag = 't' + str(jth_trip) + last_node.tag[2:] + str(cs_id)

            return current_tag

        elif kwargs.get('para') == 'id':
            last_id = last_node.identifier
            current_id = [i for i in last_id]
            current_id.append(cs_id)
            current_id = tuple(current_id)

            return current_id

        elif kwargs.get('para') == 'parent':
            parent_id = last_node.identifier

            return parent_id

        else:
            raise ValueError('Plz specify para value')

    else:
        raise ValueError("Parameter 'cd' or 'trip' is incorrect or missed!")


def get_var_value(v_name,  # The SimV's name
                  trip_idx,  # start from 1
                  sim_v_info,
                  cs_num,  # The quantity of charging stations
                  solutions  # Only a single individual of the population
                  ):
    """
    Find the 'Z' value list of the certain simV's trip
    :param v_name: the name of SimV
    :param trip_idx: the index of trip
    :param sim_v_info: a dataframe containing all sim_info grouped by simV
    :param cs_num: the quantity of cs
    :param solutions: Vars
    :return: a slice of vars
    """
    # Get the start loc of the whole "Z" variables
    z_loc = cs_num * 3
    # Get the loc of this simV's trip
    loc = sim_v_info.loc[v_name, 'var_loc'] + (trip_idx - 1) * 3 + z_loc
    # Vars slice
    return solutions[loc:loc+3]


def set_cv1(cs_num: int, vars_):
    """
    Calculate and set the cv_1 that limits the amount of stations to be set

    :param cs_num: Candidate charging stations amount
    :param vars_: solution
    :return: A numpy.array cv
    """
    return np.array([sum(vars_[0: cs_num]) - cs_num])


def set_cv3(cs_num: int, vars_, vs_cs: DataFrame):
    """
    Set cv3 (For each trip, this sim_v can only get charging in an opened station) for each trip; need vs_cs;

    :param cs_num:
    :param vars_:
    :param vs_cs:
    """

    cv3_list = []
    for v, row in vs_cs.iterrows():
        for outer_count, stations in enumerate(row.destination):
            for inner_count, s in enumerate(stations):
                cv3 = vars_[row.var_loc + outer_count * 3 + inner_count + cs_num * 3] - vars_[s]
                cv3_list.append(cv3)

    return np.array(cv3_list)


def set_refer_vars(cs_num: int, refer_num: int = 1):
    """
    Generate and set referred_vars

    :param cs_num: candidate charging stations amount
    :param num_vars: vars amount
    :param interval: interval of the "Z" for each trip, which is corresponding to @sim_cs in the simulation
    :param refer_num: num of referred solutions needed
    :return: 2-dim numpy array of solution
    """

    if refer_num == 1:
        X = [1] * cs_num  # Station
        VH = [0,0,0]  # vehicle type
        refer_vars = np.array([X + VH])
        assert refer_vars.ndim == 2, f"Invalid 'refer_vars' format'"
        return refer_vars
    else:  # When more refer_vars are needed
        refer_vars_list = []  # Reserve the first one
        for j in range(refer_num):
            X = [random.randint(0, 1) for _ in range(cs_num)]
            # X = [1] * cs_num
            VH = [0,0,0]

            current_refer_vars = np.array(X + VH)

            assert current_refer_vars.ndim == 1, f"Invalid 'refer_vars_list' format'"
            refer_vars_list.append(current_refer_vars)

        refer_vars = np.hstack([refer_vars_list])
        assert refer_vars.ndim == 2, f"Invalid 'refer_vars_list' format'"

        return refer_vars


def choose_staion(SIM_N, is_random=False):
    """select a station from the SIM_Nth nearest station list randomly, or select nearest one"""
    station_list = [0] * SIM_N
    if is_random:
        # 随机选择一个位置将其设置为1
        random_index = random.randint(0, SIM_N - 1)
        station_list[random_index] = 1
    else:
        station_list[0] = 1
    return station_list