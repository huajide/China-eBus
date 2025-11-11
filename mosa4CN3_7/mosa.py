"""
@Project ：DissertationPY 
@File    ：mosa.py
@IDE     ：PyCharm 
@Author  ：Xander PENG
@Date    ：4/9/2022 00:27 
"""
import copy
import os.path
import random
import math
from typing import List
from problem import Problem
import numpy as np
import time
import logging
import pandas as pd
import utils
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from vehicle_type import vehi_type_num

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class MOSAConfig:
    def __init__(self,
                 problem: Problem,
                 initial_temperature: int or float = 1000,
                 end_temperature: int or float = 200,
                 cooling_alpha: float = 0.98,  # within (0,1)
                 annealing_iters: int = 300,
                 annealing_strength: float = 0.2,  # perturb strength; range between (0, 1)
                 early_termination: dict = None,  # dict{'max_iters': 1000, 'max_duration': 15} //unit: hour
                 min_delta_hv: float = 0.01,  # 0.01 by default, with the max_no_eliminated of 10000
                 multiprocess=False,
                 what_if=False,
                 sim_n: int = 3,
                 is_tested=False
                 ):
        """ Check the validity of inputs """
        # assert sl >= hl > 0, f'Invalid sl and hl setting'
        assert 0 < cooling_alpha < 1, f'Invalid cooling_factor'

        # Static attrs

        self.problem = problem
        self.initial_temperature = initial_temperature
        self.end_temperature = end_temperature
        self.cooling_alpha = cooling_alpha
        self.annealing_iters = annealing_iters
        self.annealing_strength = annealing_strength
        self.early_termination = early_termination
        self.min_delta_hv = min_delta_hv
        self.multiprocess = multiprocess
        self.what_if = what_if
        self.sim_n = sim_n
        # self.is_tested = is_tested


class MOSA(MOSAConfig):
    def __init__(self, problem, hl=20, sl=50, clustering_iters=50, hill_climbing_num=100,
                 initial_temperature=1000, end_temperature=1e-7,
                 cooling_alpha=0.98, annealing_iters=200, annealing_strength=0.2,
                 early_termination={}, multiprocess=False, sim_n=3
                 ):
        # Inherit from config
        super(MOSA, self).__init__(problem, initial_temperature, end_temperature, cooling_alpha,
                                   annealing_iters, annealing_strength, early_termination, multiprocess, sim_n)
        # dynamic attrs
        self.archive = {}  # Store the solutions
        self.problem_records = {}  # Store output for specific problem
        self.start_time = time.perf_counter()  # Record the time when the class has been instantiation
        self.current_temperature = 0
        self.fitness = []
        self.archive_history = []

    def load_refer_solutions(self, refer_vars: np.ndarray = None, refer_obj=None, path=None):
        """
        load the referred solutions and objectives values into the archive

        :param refer_vars: A 2-dimension numpy ndarray
        :param refer_obj: numpy ndarray accordance to refer_vars
        :param path: The archive.csv path
        """

        if path is not None:
            logging.info('Start to read archive stored in the local disk.')
            refer_vars, refer_obj, refer_cvs = utils.read_archive(path)
            logging.info("Read archive successfully!")
        # Check refer_vars
        assert isinstance(refer_vars, np.ndarray), f"'refer_vars should be np.ndarray'"
        assert refer_vars.ndim == 2, f"'refer_vars' should be a 2-dim ndarray"
        assert refer_vars.shape[1] == self.problem.vars_num, f"Plz check the length of 'refer_vars'"
        # Check refer_obj
        if refer_obj is not None:
            assert isinstance(refer_obj, np.ndarray), f"'refer_obj' should be np.ndarray"
            assert refer_vars.shape[0] == refer_obj.shape[0], f"'refer_vars' & 'refer_obj' should have the same length"
            assert refer_obj.shape[1] == self.problem.obj_num[0], f"Plz check the number of 'refer_obj' columns"

        # If there is no given refer_obj, calculate them
        if refer_obj is None:
            logging.info("There is no refer_obj. Start calculate the objectives of refer_vars")
            if self.multiprocess:  # Use Multiprocessing for the sake of acceleration
                all_vars_list = list(map(lambda x: [x], refer_vars))  # Store each group of vars separately in a list

                res_list = []
                with Pool(cpu_count()) as pool:
                    res = pool.starmap_async(self.problem.eval_vars, all_vars_list)
                    res_list.append(res.get())

                for idx, v in enumerate(res_list[0]):
                    self.archive.update({idx: [all_vars_list[0][0], v[0], v[1]]})
                    if self.problem.num_return > 2:  # Add additional problem return if it has
                        self.problem_records.update({idx: v[2]})
                logging.info("The initialization of archive has finished")
            else:
                # Calculate the objective values of refer_solutions
                for idx, v in enumerate(refer_vars):
                    obj_v, cv_v = self.problem.eval_vars(v)[0:2]
                    # Add current refer_var and objectives values into archive
                    self.archive.update({idx: [v, obj_v, cv_v]})  # [vars: [int or float], objV:[f1, f2, f3], cv:[0]]
                    if self.problem.num_return > 2:
                        self.problem_records.update({idx: self.problem.adl_returns})
                logging.info("The initialization of archive has finished")
        # When refer_obj is given, Store them into the archive directly
        else:
            for idx, v in enumerate(tuple(zip(refer_vars, refer_obj, refer_cvs))):
                self.archive.update({idx: [v[0], v[1], v[2]]})  # [vars: [int or float], objV:[f1, f2, f3], cv:[0]]
            logging.info("The initialization of archive has finished")

    def is_archive(self, vars_: List[int or float]):
        """
        Judge if the input solution has already stored in the archive
        :param vars_: input solution
        :return: False or True
        """
        return False if len(list(filter(lambda x: all(x == vars_), map(lambda y: y[0],
                                                                       self.archive.values())))) == 0 else True

    def check_perturb_cv(self, new_var):

        cv = self.problem.eval_vars(new_var, 'CV')
        return True if all(cv <= 0) else False

    def perturb(self, old_vars, is_cv=True, is_h_cv=False):
        """
        The perturbation operation for producing new solution;
        A "max_attempts" is adopted to protect resource utilization, once the "new_vars" is also in the archive,
        continue perturbing to produce new vars but the loop is limited.

        :param is_h_cv: Take cv into account in the hydrogen scenario
        :param old_vars: The "old_vars" that need to be perturbed
        :param is_cv: take cv into account
        :return: the new vars
        """

        new_vars = copy.deepcopy(old_vars)
        num_exit = 1  # Max attempts for avoiding same vars in the archive
        cs_num = self.problem.cs_num
        if self.what_if:
            shrink_rate = 0.1
        else:
            shrink_rate = self.current_temperature / self.initial_temperature
        # s_time = time.perf_counter()
        while num_exit > 0:
            # Select which vars should be perturbed randomly
            if is_cv:
                indexes_x = random.sample(range(cs_num),
                                          random.randint(1, math.ceil(max(shrink_rate,0.01) *
                                                                      self.annealing_strength * cs_num)))
                for idx_x in indexes_x:  # perturb stations
                    new_vars[idx_x] = random.randint(0, 1)

                # perturb the vehicle type
                if random.random() < 0.5*(shrink_rate+0.1):
                    new_vars[cs_num] =  random.randint(0,vehi_type_num('large')-1)
                    new_vars[cs_num+1] = random.randint(0, vehi_type_num('medium') - 1)
                    new_vars[cs_num+2] = random.randint(0, vehi_type_num('small') - 1)

                # 确保至少有一个充电站被选中
                if all(new_vars[:cs_num] == 0):
                    # 如果所有充电站都未被选中，随机选择一个充电站设为1
                    random_idx = random.randint(0, cs_num - 1)
                    new_vars[random_idx] = 1

                num_exit -= 1

            elif is_h_cv:

                cs_num = self.problem.cs_num
                sim_info_num = len(self.problem.sim_info)
                sim_trip_num = int(sim_info_num / 12)

                indexes_y = random.sample(range(sim_trip_num),
                                          random.randint(1, math.ceil(self.annealing_strength * sim_trip_num)))
                for idx_y in indexes_y:  # Perturb the y value
                    parse_var = [0] * 11
                    parse_var.append(1)
                    random.shuffle(parse_var)
                    for iy, v in enumerate(parse_var):  # replace
                        new_vars[cs_num + idx_y * 12 + iy] = v

                ''' Set for the X based on the y value '''
                # Get the var_loc of which idx_y = 1
                sel_y = np.where(new_vars[cs_num:] == 1)[0]
                # The list containing all selected stations' osmid
                sel_stations = self.problem.sim_info.query("var_loc in @sel_y").drop_duplicates('cs_id')[
                    'cs_id'].to_list()
                # The var_loc of stations should be deployed
                sel_x = self.problem.cs_gdf.query("osmid in @sel_stations").index.to_list()
                for x in sel_x:
                    if x in sel_x:
                        new_vars[x] = 1
                    else:
                        new_vars[x] = 0

                num_exit -= 1

            else:
                indexes = random.sample(range(self.problem.vars_num),
                                        random.randint(1, math.ceil(self.annealing_strength * self.problem.vars_num)))
                feasible = False
                num_cv_exit = 2  # Max attempts for deriving feasible vars
                while not feasible and num_cv_exit > 0:
                    for idx in indexes:
                        idx_lb = self.problem.lb[idx]
                        idx_ub = self.problem.ub[idx]
                        idx_type = self.problem.vars_type[idx]
                        # Perturb the current index var
                        new_vars[idx] = random.randint(idx_lb, idx_ub) if idx_type == 0 else random.uniform(idx_lb,
                                                                                                            idx_ub)
                    # Check if the new_vars fulfill CVs
                    feasible = self.check_perturb_cv(new_vars)
                    num_cv_exit -= 1

                num_exit -= 1
                if feasible and num_cv_exit > 0:
                    logging.info("Perturb feasible solutions successfully and the iters are: ", num_cv_exit)
        # print('perturb time: ', time.perf_counter() - s_time)
        return new_vars

    def cal_fitness_range(self, old_obj: np.ndarray, new_obj: np.ndarray) -> np.ndarray:
        """
        Calculate the fitness range using old, new solutions and all solutions in the archive

        :param old_obj: The old vars' obj
        :param new_obj: The new vars' obj
        :return: The fitness range
        """
        # Use a list to store all objectives values including the old, new obj and archive
        objs = [old_obj, new_obj] + [i[1] for i in self.archive.values()]
        return np.nanmax(objs, 0) - np.nanmin(objs, 0)

    def cal_domination_amount(self, old_obj: np.ndarray, new_obj: np.ndarray, ranges=None):
        """
        The basic function for calculating the domination amount

        :param old_obj: The old vars' obj
        :param new_obj: The new vars' obj
        :param ranges: R
        :return: The domination amount
        """
        if ranges is None:
            ranges = self.cal_fitness_range(old_obj, new_obj)
        # Process if there is only one objective
        if all(np.array([ranges[1], old_obj[1], new_obj[1]]) == 0):
            return np.prod([abs(new_obj[0] - old_obj[0]) / ranges[0]])

        elif all(np.array([ranges[0], old_obj[0], new_obj[0]]) == 0):
            return np.prod([abs(new_obj[1] - old_obj[1]) / ranges[1]])
        else:
            return np.prod([abs(i - j) / r for i, j, r in zip(old_obj, new_obj, ranges)])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(np.array(-x, dtype=np.float64)))

    def dominates(self, old_obj: np.ndarray, new_obj: np.ndarray):
        """
        Compare the status of domination between the old_obj and the new_obj;
        There are 3 outputs:
            1. Which solution should be set as "current_var" -> "old" or "new" or a 'key' pointing to archive solution
            2. If the "new_vars" is selected, whether it should be added into the archive -> True or False
            3. If the "new_vars" is selected, which solutions in the archive should be deleted -> [keys] or []

        :param old_obj: The old vars' obj
        :param new_obj: The new vars' obj
        """
        # Check if the old_obj and new_obj is instance of np.array
        assert isinstance(old_obj, np.ndarray), f'The old_obj is not an instance of numpy ndarray'
        assert isinstance(new_obj, np.ndarray), f'The new_obj is not an instance of numpy ndarray'
        '''
        There are 3 situations when compare the old and new vars
        '''
        if all(new_obj >= old_obj):  # 1. When new_vars dominates old_vars （越大越好）

            # Derive the solutions in the archive dominated by the new_vars
            # x[0]和x[1][1]分别是archive的键和双目标值的元组，二者再组成元组即为y，所以这行代码是筛选双目标函数值均小于新解的元组，
            # dominated_solutions_idx是旧解不好的键
            dominated_solutions = list(filter(lambda y: all(new_obj >= y[1]),
                                              map(lambda x: (x[0], x[1][1]), self.archive.items())))  # 筛选目标函数均更差的以前的解
            dominated_solutions_idx = [i[0] for i in dominated_solutions]  # The dict keys for these solutions
            # Derive the solutions in the archive dominates the new_vars
            dominate_solutions = list(filter(lambda y: all(y[1] >= new_obj),
                                             map(lambda x: (x[0], x[1][1]), self.archive.items())))
            dominate_solutions_idx = [i[0] for i in dominate_solutions]

            # 1) When new_vars dominates k(k >= 1) solutions in the archive
            if len(dominated_solutions_idx) >= 1:
                # print("new dom >=1 old")
                return "new", True, dominated_solutions_idx
            # 2) When new_vars is dominated by k(k >= 1) solutions in the archive
            elif len(dominate_solutions_idx) >= 1:
                # Calculate the domination amount between the new solution and 'k' solutions
                ranges = self.cal_fitness_range(old_obj, new_obj)  # Calculate the fitness range
                domination_amounts = [self.cal_domination_amount(k, new_obj, ranges) for k in
                                      [i[1] for i in dominate_solutions]]
                dom_min = min(domination_amounts)
                dom_min_idx = dominate_solutions_idx[np.argmin(domination_amounts)]
                probability = self.sigmoid(dom_min)  # Accept probability
                if random.random() <= probability:  # Set solution that in the archive corresponding to the dom_min
                    # print('return archive 1.2.1')
                    return dom_min_idx, False, []
                else:  # Set new_solution as current one without adding
                    # print('return new 1.2.2')
                    return "new", False, []
            else:  # 3) There is no dominate relations between the new solution and any solutions in the archive
                # Judge whether old solution is in the archive
                is_archive_parse = list(filter(lambda y: all(y[1] == old_obj),
                                               map(lambda x: (x[0], x[1][1]),
                                                   self.archive.items())))
                is_archive = True if len(is_archive_parse) != 0 else False
                # print('return new, 1.3')
                if is_archive:
                    return "new", True, [is_archive_parse[0][0]]
                else:
                    return "new", True, []

        elif all(old_obj >= new_obj):  # 2. When the old solution dominates the new one
            # Check which solutions in the archive dominates the new_solution
            dominate_solutions = list(filter(lambda y: all(y[1] >= new_obj),
                                             map(lambda x: (x[0], x[1][1]), self.archive.items())))
            dominate_solutions_idx = [i[0] for i in dominate_solutions]

            # Calculate the domination amount
            ranges = self.cal_fitness_range(old_obj, new_obj)
            domination_amount_avg = (np.nansum([self.cal_domination_amount(k, new_obj, ranges) for k in
                                                [i[1] for i in dominate_solutions]]) +
                                     self.cal_domination_amount(old_obj, new_obj, ranges)) / (
                                            len(dominate_solutions) + 1)
            probability = self.sigmoid(-domination_amount_avg * self.current_temperature)
            if random.random() <= probability:
                # print('return new, 2.1')
                return "new", False, []
            else:
                # print('return old, 2.2')
                return "old", False, []
        # 3. When the new solution and old solution are non-dominated by each other
        elif not all(old_obj >= new_obj) and not all(new_obj >= old_obj):

            # Derive the solutions in the archive dominated by the new_vars
            dominated_solutions = list(filter(lambda y: all(new_obj >= y[1]),
                                              map(lambda x: (x[0], x[1][1]), self.archive.items())))
            dominated_solutions_idx = [i[0] for i in dominated_solutions]  # The dict keys for these solutions
            # Derive the solutions in the archive dominates the new_vars
            dominate_solutions = list(filter(lambda y: all(y[1] >= new_obj),
                                             map(lambda x: (x[0], x[1][1]), self.archive.items())))
            dominate_solutions_idx = [i[0] for i in dominate_solutions]

            if len(dominated_solutions) > 0:  # 1) When the new solution dominates 'k' solutions in the archive
                # print('return new, 3.1')
                return "new", True, dominated_solutions_idx
            elif len(dominate_solutions) > 0:  # 2) The new solution is dominated by 'k' solutions in the archive
                ranges = self.cal_fitness_range(old_obj, new_obj)
                domination_amount_avg = np.nansum([self.cal_domination_amount(k, new_obj, ranges) for k in
                                                   [i[1] for i in dominate_solutions]]) / len(dominate_solutions)
                probability = self.sigmoid(-domination_amount_avg * self.current_temperature)
                if random.random() <= probability:
                    # print('return new, 3.2.1')
                    return "new", False, []
                else:
                    # print('return old, 3.2.2')
                    return "old", False, []
            else:  # 3) There is no domination relations between the new solution and all the solutions in archive
                # print('return new, 3.3')
                return "new", True, []
        else:
            raise ValueError('Cannot determine the status of domination. Plz check the solutions')

    def print_statistics(self, current_obj: np.ndarray, current_cv: np.arange, iter_num: int):
        # is_feasible = all(current_cv <= 0)  # Check whether the solution meet the cv requirement
        print(" | {:<10.2f} | {:<8d} | {:<.2e} {:<.2e} | {:^9} | {:^8.0f} |".
              format(self.current_temperature,
                     iter_num,
                     *current_obj,
                     len(self.archive),
                     (time.perf_counter() - self.start_time) / 60
                     ))

    @staticmethod
    def print_header():
        print(" | {:^8} | {:^8} | {:^18} | {:^9} | {:^} |".format('temp', 'iter', 'ObjV', 'feasible', 'duration'))

    def plot_results(self, final_results=None, save=False, path=None):

        plt.figure(figsize=(6, 6), dpi=230)  # Set figure size
        plt.title('Pareto results',
                  fontdict={'fontsize': 12, 'fontweight': 'bold', 'color': 'darkorange'})  # Title setting
        # Hide the top and right axes
        plt.gca().spines['right'].set_color('None')
        plt.gca().spines['top'].set_color('None')
        # axes settings
        plt.xlabel('ObjV1', labelpad=6, color='orangered', fontsize=12)
        plt.xticks(fontsize=8)
        plt.ylabel('ObjV2', labelpad=6, color='orangered', fontsize=12)
        plt.yticks(fontsize=8)

        # if final_results is None:  # Plot the self.archive
        #     ObjV = np.array([i[1] for i in self.archive.values()])
        #     plt.scatter(ObjV[:, [0]], ObjV[:, [1]], color='coral', marker='X', linewidths=1, alpha=0.8)
        #     plt.show(block=True)
        #     if save:
        #         plt.savefig(path + '\\' + 'pareto_results.png')
        # else:
        #     ObjV = np.array([i[1] for i in final_results.values()])
        #     plt.scatter(ObjV[:, [0]], ObjV[:, [1]], color='coral', marker='X', linewidths=1, alpha=0.8)
        #     plt.show(block=True)
        #     if save:
        #         plt.savefig(path + '\\' + 'pareto_results.png')

    def check_cv(self):
        """
        Filter the solutions that fulfill the CV requirement in the archive
        :return: Filtered archive
        """

        # Delete solutions that do not fulfill the CV requirement in the archive
        final_archive = dict(filter(lambda x: all(x[1][2] <= 0), self.archive.items()))
        return final_archive

    @staticmethod
    def out2csv(results_df: pd.DataFrame, path, *args):
        assert len(results_df) > 0, f'No results can be output!'

        num = len(results_df)
        for s_idx in range(num):
            if s_idx == 0:
                # Vars
                vars_df = results_df.iloc[:1, :1].explode('Vars', ignore_index=True).T
                vars_df.index = [results_df.index[s_idx]]
                # ObjVs
                objs_df = results_df.iloc[:1, 1:2].explode('ObjVs', ignore_index=True).T
                objs_df.index = [results_df.index[s_idx]]
                # CVs
                cvs_df = results_df.iloc[:1, 2:].explode('CVs', ignore_index=True).T
                cvs_df.index = [results_df.index[s_idx]]
            else:
                # Var
                var_df = results_df.iloc[s_idx:s_idx + 1, :1].explode('Vars', ignore_index=True).T
                var_df.index = [results_df.index[s_idx]]
                # ObjV
                obj_df = results_df.iloc[s_idx:s_idx + 1, 1:2].explode('ObjVs', ignore_index=True).T
                obj_df.index = [results_df.index[s_idx]]
                # CV
                cv_df = results_df.iloc[s_idx:s_idx + 1, 2:].explode('CVs', ignore_index=True).T
                cv_df.index = [results_df.index[s_idx]]
                # Concat
                vars_df = pd.concat([vars_df, var_df])
                objs_df = pd.concat([objs_df, obj_df])
                cvs_df = pd.concat([cvs_df, cv_df])
        vars_df.to_csv(path + '\\' + args[0] + '_vars.csv')
        objs_df.to_csv(path + '\\' + args[0] + '_objs.csv')
        cvs_df.to_csv(path + '\\' + args[0] + '_cvs.csv')

    def output_adl2csv(self, path):
        # Create a directory to store additional output
        is_exist = os.path.exists(path + '\\' + 'adl_output\\')
        if not is_exist:
            os.makedirs(path + '\\' + 'adl_output\\')

        for idx, add_v in self.problem_records.items():
            for iv, v in enumerate(add_v):
                v.to_csv(path + '\\' + 'adl_output\\' + str(idx) + '_' + self.problem.adl_names[iv] + '.csv')


    # ---------- 2-D hyper-volume（最小化问题） ----------
    @staticmethod
    def _hv_2d_min(points: np.ndarray, ref_point=(1., 1.)) -> float:
        """
        计算 2 维最小化问题的 Hyper-volume。
        points:  已在 [0,1] 内；ref_point 在 (1,1) 右上角。
        """
        if points.size == 0:
            return 0.0

        # 去掉 reference point 以外的点
        pts = points[(points[:, 0] <= ref_point[0]) & (points[:, 1] <= ref_point[1])]

        # 去掉被支配的点(只保留 Pareto 前沿)
        keep_idx = []
        for i, p in enumerate(pts):
            dominated = np.any((pts <= p).all(axis=1) & (pts < p).any(axis=1))
            if not dominated:
                keep_idx.append(i)
        pts = pts[keep_idx]

        # 按第 0 维从小到大排序
        pts = pts[np.argsort(pts[:, 0])]
        hv, prev_y = 0.0, ref_point[1]

        for x, y in pts:
            if y < prev_y:
                hv += (ref_point[0] - x) * (prev_y - y)
                prev_y = y
        return hv

    # ---------- 绘制超体积收敛曲线 ----------
    def hyper_volume(self, points: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> float:
        """
        计算给定目标值点集的超体积指标

        :param points: 目标值点集，形状为 (n_points, n_objectives)
        :param min_vals: 每个目标的最小值，用于归一化
        :param max_vals: 每个目标的最大值，用于归一化
        :return: 超体积值 (0-1之间)
        """
        if points.size == 0:
            return 0.0

        # 使用全局 min-max 进行归一化
        diff = max_vals - min_vals
        diff[diff == 0] = 1.0  # 防止除0

        # 归一化到 0~1 (越大越好)
        norm_objs = (points - min_vals) / diff
        # 转换为最小化问题
        norm_objs = 1.0 - norm_objs

        # 调用已有的2D超体积计算方法
        ref_point = (1.0, 1.0)
        return self._hv_2d_min(norm_objs, ref_point)


    def output_fitness(self, min_vals=None,  max_vals=None):
        """
        根据 self.archive_history (只含 ObjV) 计算并绘制超体积收敛曲线。
        适用于 2 目标、且“越大越好”的问题。
        """
        if len(self.archive_history) == 0:
            print("archive_history 为空，无法计算超体积。")
            return []

        # 1. 收集所有阶段的目标值，用于全局 min-max 归一化
        all_objs = np.vstack([objs for objs in self.archive_history if objs.size != 0])
        if all_objs.size == 0:
            print("目标值为空，无法计算超体积。")
            return []

        if min_vals is None or max_vals is None:
            min_vals = np.min(all_objs, axis=0)
            max_vals = np.max(all_objs, axis=0)

        hv_values = []

        # 2. 逐阶段计算 HV
        for objs in self.archive_history:
            if objs.size == 0:
                hv_values.append(0.0)
                continue

            hv = self.hyper_volume(objs, min_vals, max_vals)
            hv_values.append(hv)

        # 3. 绘制
        # plt.figure(figsize=(9, 5))
        # plt.plot(range(1, len(hv_values) + 1), hv_values,
        #          marker='o', linestyle='-', color='steelblue')
        # plt.xlabel('Cooling Step')
        # plt.ylabel('Hyper-volume (0-1)')
        # plt.title('Hyper-volume Convergence Curve')
        # plt.grid(alpha=0.3)
        # plt.tight_layout()
        # plt.show(block=True)

        return hv_values


    def output(self, path=None, *args):

        print("{:^2} solutions are found without CVs consideration".format(len(self.archive)))
        results = self.check_cv()
        print("{:^2} solutions are found with CVs taken into account".format(len(results)))

        if 'infeasible' in args:  # Retain the whole archive including infeasible ones
            # Plot the results
            try:
                self.plot_results(results, False, path)
            except IndexError or ValueError:
                self.plot_results(self.archive, False, path)

            if 'store' in args:
                assert path is not None, f"Plz specify a path for storing files"
                assert len(self.archive) > 0, f'No results found'
                inf_archive_df = pd.DataFrame.from_dict(self.archive, orient='index', columns=['Vars', 'ObjVs', 'CVs'])
                self.out2csv(inf_archive_df, path, 'inf_archive')
                self.output_adl2csv(path)
                # inf_archive_df.to_csv(path + '\\' + 'inf_archive.csv')
                if len(results) != 0:
                    archive_df = pd.DataFrame.from_dict(results, orient='index', columns=['Vars', 'ObjVs', 'CVs'])
                    self.out2csv(archive_df, path, 'archive')
                    self.output_adl2csv(path)
                # archive_df.to_csv(path + '\\' + 'archive.csv')
                logging.info('The result has been saved!')
            else:
                print('Infeasible Archive:')
                print(self.archive)
                print('Filtered Archive:')
                print(results)
        else:
            # Plot the results
            try:
                self.plot_results(results, False, path)
            except IndexError or ValueError:
                self.plot_results(self.archive, False, path)

            if 'store' in args:
                assert path is not None, f"Plz specify a path for storing files"
                assert len(self.archive) > 0, f'No results found'
                archive_df = pd.DataFrame.from_dict(results, orient='index', columns=['Vars', 'ObjVs', 'CVs'])
                self.out2csv(archive_df, path, 'archive')
                self.output_adl2csv(path)
                # archive_df.to_csv(path + '\\' + 'archive.csv')
                logging.info('The result has been saved!')
            else:
                print('Filtered Archive:')
                print(results)


    def multi_perturb(self, p: Problem, old_var, old_obj):
        """
        Perform the perturbation and result calculation by multiprocessing method

        :param p: Problem
        :param old_var: Old variables
        :param old_obj: Old objectives
        :return: A tuple containing 2 list; ([new_var, new_obj, new_cv], [var_choice, archive_choice, del_keys])
        """

        assert self.multiprocess, f"Invalid MultiProcessing status"  # Check the multiprocess setting

        new_var = self.perturb(old_var, is_cv=True)
        new_obj, new_cv = p.eval_vars(new_var)[0:2]
        var_choice, archive_choice, del_keys = self.dominates(old_obj, new_obj)

        return [new_var, new_obj, new_cv], [var_choice, archive_choice, del_keys], p.adl_returns

    @staticmethod
    def multi_test_func(p: Problem, var_):
        return p.eval_vars(var_)

    def run(self, path=None, store=None, inf=None, is_early_terminate=False, multi_tasks=8,
            is_cv=False, is_h_cv=False):
        # Check if there is an available archive
        assert len(self.archive) > 0, f"No existing archive can be found"
        # Read the conditions of early termination
        is_early_terminate = False if len(self.early_termination) == 0 else True
        if is_early_terminate:  # 如果初始化了提前结束的温度，需要如下设置
            max_iters = self.early_termination.get('max_iters')
            max_duration = self.early_termination.get('max_duration') * 3600  # unit: second
            max_no_eliminated = self.early_termination.get('max_no_eliminated')

            if not isinstance(max_iters, int) or not isinstance(max_duration, int):
                raise ValueError('Max_iters or Max_duration cannot be None')

        # 初始化
        self.current_temperature = self.initial_temperature  # Initialize the temperature
        # Select a solution randomly from the archive as current_var
        current_var, current_obj, current_cv = self.archive.get(np.random.randint(0, len(self.archive)))
        all_iter_counts = 0
        no_eliminated_times = 0
        archive_choice = False
        self.print_header()

        # 用于存储HV历史记录的变量
        hv_history = []

        # Start the outer loop for temperature annealing
        break_outer = False
        while self.current_temperature >= self.end_temperature:
            for i in range(self.annealing_iters):  # The inner loop for looking for new solutions under a certain temp
                '''Once there is a new solution is non-dominated to any solutions in the archive, 
                   Select a solution from the archive randomly to start iterations and finding '''
                if archive_choice:
                    current_var, current_obj, current_cv = self.archive.get(random.choice(list(self.archive.keys())))
                # enable multiprocessing
                if self.multiprocess:
                    all_iter_counts += multi_tasks
                    no_eliminated_times += multi_tasks
                    multi_args = [[self.problem, current_var, current_obj] for _ in range(multi_tasks)]
                    res_list = []
                    with Pool(multi_tasks) as pool:
                        res = pool.starmap_async(self.multi_perturb, multi_args)
                        res_list.append(res.get())
                    # Filter those new_vars are selected and needed to store in the archive
                    res_best = list(filter(lambda x: x[1][1] is True and x[1][0] == 'new', res_list[0]))
                    # # 测试：保存变量到文件
                    # with open('data_res_best.pkl', 'wb') as f:
                    #     pickle.dump(res_best, f)
                    # with open('data_res_list.pkl', 'wb') as f:
                    #     pickle.dump(res_list, f)

                    if len(res_best) == 0:  # cannot get the best new_vars with "new" and "True"
                        # Filter those new_vars are selected but not to be stored in the archive
                        res_inferior = list(filter(lambda x: x[1][0] == 'new' or x[1][1] is True, res_list[0]))
                        if len(res_inferior) > 0:  # Find inferior new_vars
                            # Choose one of them randomly as the current one
                            eval_list, dominates_list = random.sample(res_inferior, 1)[0][0:2]
                            new_var, new_obj, new_cv = eval_list
                            var_choice, archive_choice, del_keys = dominates_list
                        else:  # find nothing
                            var_choice = 'old'
                            del_keys = []
                    else:  # Find res_best
                        # Store all of them and Select one of them as the new one
                        all_del_keys = []
                        for idx, rb in enumerate(res_best):
                            self.archive.update({"".join(['SA', str(all_iter_counts), 'M', str(idx)]): rb[0]})
                            if self.problem.num_return > 2:  # When there are additional returns for the problem
                                self.problem_records.update({"".join(['SA', str(all_iter_counts), 'M', str(idx)]):
                                                                 rb[2]})
                            all_del_keys.extend(rb[1][2])

                        eval_list, dominates_list = random.sample(res_best, 1)[0][0:2]
                        new_var, new_obj, new_cv = eval_list
                        var_choice, archive_choice, del_keys = dominates_list
                        del_keys = list(set(all_del_keys))
                        # print(f"del_keys: {del_keys}")
                        no_eliminated_times = 0

                else:  # Without Multiprocess setting
                    all_iter_counts += 1
                    no_eliminated_times += 1
                    new_var = self.perturb(current_var, is_h_cv=is_h_cv, is_cv=is_cv)  # Get a new vars via perturbation
                    new_obj, new_cv = self.problem.eval_vars(new_var)[0:2]  # Get new_vars' ObjVs and CVs
                    # Judge the status of domination between the old and new solutions
                    var_choice, archive_choice, del_keys = self.dominates(current_obj, new_obj)
                    if var_choice == "new" and archive_choice:
                        no_eliminated_times = 0

                # Update the current solution info
                if var_choice == "new":  # When the new_vars is selected
                    current_var, current_obj, current_cv = new_var, new_obj, new_cv
                elif var_choice == "old":
                    pass
                else:  # When the "key" of solution in the archive is passed by
                    current_var, current_obj, current_cv = self.archive.get(var_choice)

                if archive_choice and not self.multiprocess:  # Add the new solution into the archive
                    self.archive.update({"".join(['SA', str(all_iter_counts)]): [current_var, current_obj, current_cv]})
                    self.problem_records.update({"".join(['SA', str(all_iter_counts)]):
                                                     self.problem.adl_returns})
                if len(del_keys) > 0:  # Delete dominated solutions in the archive
                    [self.archive.pop(k) for k in del_keys]
                    try:  # 增加
                        [self.problem_records.pop(k) for k in del_keys]
                    except:
                        None
                # logging.info("Eval_iter: " + str(all_iter_counts))
                # logging.info("No eliminated times: " + str(no_eliminated_times))

            # 在每次温度下降后记录archive状态
            self.archive_history.append(
                np.array([v[1] for v in self.archive.values()], dtype=float)
            )

            # 检查是否满足HV提前终止条件
            if is_early_terminate and len(self.archive_history) > int(max_no_eliminated / self.annealing_iters):
                # 收集所有目标值用于归一化
                all_objs = np.vstack([objs for objs in self.archive_history if objs.size != 0])
                if all_objs.size != 0:
                    min_vals = np.min(all_objs, axis=0)
                    max_vals = np.max(all_objs, axis=0)

                    # 计算当前HV和之前HV
                    current_points = self.archive_history[-1]
                    previous_points = self.archive_history[-int(max_no_eliminated / self.annealing_iters) - 1]

                    if current_points.size != 0 and previous_points.size != 0:
                        current_hv = self.hyper_volume(current_points, min_vals, max_vals)
                        previous_hv = self.hyper_volume(previous_points, min_vals, max_vals)
                        print(f"Delta HV: {(current_hv - previous_hv):.4f}")
                        # 如果HV变化小于min_delta_hv，则提前终止
                        if current_hv - previous_hv < self.min_delta_hv:
                            break_outer = True
                            break

            # Once finishing a for-loop, cooling the temperature
            self.current_temperature = self.cooling_alpha * self.current_temperature
            self.print_statistics(current_obj, current_cv, all_iter_counts)
            if is_early_terminate:  # When the early termination is set, judge whether it should stop
                if all_iter_counts >= max_iters or (
                        time.perf_counter() - self.start_time) >= max_duration or break_outer:
                    break

        # Once the "current_temp < end_temp", finish the algorithm and output all results
        self.output(path, store, inf)

