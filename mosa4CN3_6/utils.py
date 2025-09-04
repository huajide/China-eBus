# @Time : 2022-09-15 16:51
# @Author : Xander PENG
# @File : utils.py
# @Software: PyCharm
# @Description:

import numpy as np
import pandas as pd


def convert_str2array(str_l: str):
    s = str_l.replace('[', '')
    s = s.replace(']', '')
    s = s.split()
    s = [float(i) for i in s]
    s = np.array(s)
    return s


def read_archive(path):
    archive = pd.read_csv(path, index_col=0)
    archive = archive.applymap(convert_str2array)
    vars_ = np.array(archive['Vars'].to_list())
    ObjV = np.array(archive['ObjVs'].to_list())
    CVs = np.array(archive['CVs'].to_list())
    return vars_, ObjV, CVs

