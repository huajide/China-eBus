"""
@Project ：DissertationPY 
@File    ：problem.py
@IDE     ：PyCharm 
@Author  ：Xander PENG
@Date    ：4/9/2022 01:46 
"""
from typing import List, Tuple


class Problem:
    def __init__(self,
                 vars_num: int,  # The amount of variables
                 vars_type: List[int],  # The type of vars: 0 for Integer and 1 for Real
                 lb: List[int or float],  # lower bounds of vars
                 ub: List[int or float],  # upper bounds of vars
                 obj_num: int,
                 cv_num: int,
                 **kwargs
                 ):
        """ Check the validity of inputs """
        assert vars_num == len(vars_type) == len(lb) == len(ub), 'Plz check vars setting'

        # Static attrs
        self.vars_num = vars_num
        self.vars_type = vars_type
        self.lb = lb
        self.ub = ub
        self.obj_num = obj_num
        self.cv_num = cv_num
        self.num_return = 2  # the number of objects returned after @eval_vars
        self.adl_names: Tuple[str] = tuple()  # The names of additional output for the sake of output file name

        # Dynamic attrs
        self.archive = []
        self.adl_returns = []  # A container for additional returns storing

    def eval_vars(self, vars_, *args):
        """
        Define the model and outputs;
        There should be two outputs: 1. ObjV: [f1, f2]; 2. CV: [cv1, cv2, cv3, cv4, ..., cvn]
        :param args: Receive if only calculate and return CVs
        :param vars_: variables
        """
        pass


