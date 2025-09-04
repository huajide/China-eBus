import random
import numpy as np
import pickle


def dominates(archive, old_obj: np.ndarray, new_obj: np.ndarray):
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
    current_temperature = 900
    if all(new_obj >= old_obj):  # 1. When new_vars dominates old_vars （越大越好）

        # Derive the solutions in the archive dominated by the new_vars
        # x[0]和x[1][1]分别是archive的键和双目标值的元组，二者再组成元组即为y，所以这行代码是筛选双目标函数值均小于新解的元组，
        # dominated_solutions_idx是旧解不好的键
        dominated_solutions = list(filter(lambda y: all(new_obj >= y[1]),
                                          map(lambda x: (x[0], x[1][1]), archive.items())))  # 筛选目标函数均更差的以前的解
        dominated_solutions_idx = [i[0] for i in dominated_solutions]  # The dict keys for these solutions
        # Derive the solutions in the archive dominates the new_vars
        dominate_solutions = list(filter(lambda y: all(y[1] >= new_obj),
                                         map(lambda x: (x[0], x[1][1]), archive.items())))
        dominate_solutions_idx = [i[0] for i in dominate_solutions]

        # 1) When new_vars dominates k(k >= 1) solutions in the archive
        if len(dominated_solutions_idx) >= 1:
            print("new dom >=1 old")
            return "new", True, dominated_solutions_idx
        # 2) When new_vars is dominated by k(k >= 1) solutions in the archive
        elif len(dominate_solutions_idx) >= 1:
            # Calculate the domination amount between the new solution and 'k' solutions
            ranges = cal_fitness_range(archive, old_obj, new_obj)  # Calculate the fitness range
            domination_amounts = [cal_domination_amount(k, new_obj, ranges) for k in
                                  [i[1] for i in dominate_solutions]]
            dom_min = min(domination_amounts)
            dom_min_idx = dominate_solutions_idx[np.argmin(domination_amounts)]
            probability = sigmoid(dom_min)  # Accept probability
            if random.random() <= probability:  # Set solution that in the archive corresponding to the dom_min
                print('return archive 1.2.1')
                return dom_min_idx, False, []
            else:  # Set new_solution as current one without adding
                print('return new 1.2.2')
                return "new", False, []
        else:  # 3) There is no dominate relations between the new solution and any solutions in the archive
            # Judge whether old solution is in the archive
            is_archive_parse = list(filter(lambda y: all(y[1] == old_obj),
                                           map(lambda x: (x[0], x[1][1]),
                                               archive.items())))
            is_archive = True if len(is_archive_parse) != 0 else False
            print('return new, 1.3')
            if is_archive:
                return "new", True, [is_archive_parse[0][0]]
            else:
                return "new", True, []

    elif all(old_obj >= new_obj):  # 2. When the old solution dominates the new one
        # Check which solutions in the archive dominates the new_solution
        dominate_solutions = list(filter(lambda y: all(y[1] >= new_obj),
                                         map(lambda x: (x[0], x[1][1]), archive.items())))
        dominate_solutions_idx = [i[0] for i in dominate_solutions]

        # Calculate the domination amount
        ranges = cal_fitness_range(archive, old_obj, new_obj)
        domination_amount_avg = (np.nansum([cal_domination_amount(k, new_obj, ranges) for k in
                                            [i[1] for i in dominate_solutions]]) +
                                 cal_domination_amount(old_obj, new_obj, ranges)) / (
                                        len(dominate_solutions) + 1)
        probability = sigmoid(-domination_amount_avg * current_temperature)
        if random.random() <= probability:
            print('return new, 2.1')
            return "new", False, []
        else:
            print('return old, 2.2')
            return "old", False, []
    # 3. When the new solution and old solution are non-dominated by each other
    elif not all(old_obj >= new_obj) and not all(new_obj >= old_obj):

        # Derive the solutions in the archive dominated by the new_vars
        dominated_solutions = list(filter(lambda y: all(new_obj >= y[1]),
                                          map(lambda x: (x[0], x[1][1]), archive.items())))
        dominated_solutions_idx = [i[0] for i in dominated_solutions]  # The dict keys for these solutions
        # Derive the solutions in the archive dominates the new_vars
        dominate_solutions = list(filter(lambda y: all(y[1] >= new_obj),
                                         map(lambda x: (x[0], x[1][1]), archive.items())))
        dominate_solutions_idx = [i[0] for i in dominate_solutions]

        if len(dominated_solutions) > 0:  # 1) When the new solution dominates 'k' solutions in the archive
            print('return new, 3.1')
            return "new", True, dominated_solutions_idx
        elif len(dominate_solutions) > 0:  # 2) The new solution is dominated by 'k' solutions in the archive
            ranges = cal_fitness_range(archive, old_obj, new_obj)
            domination_amount_avg = np.nansum([cal_domination_amount(k, new_obj, ranges) for k in
                                               [i[1] for i in dominate_solutions]]) / len(dominate_solutions)
            probability = sigmoid(-domination_amount_avg * current_temperature)
            if random.random() <= probability:
                print('return new, 3.2.1')
                return "new", False, []
            else:
                print('return old, 3.2.2')
                return "old", False, []
        else:  # 3) There is no domination relations between the new solution and all the solutions in archive
            print('return new, 3.3')
            return "new", True, []
    else:
        raise ValueError('Cannot determine the status of domination. Plz check the solutions')


def cal_fitness_range(archive, old_obj: np.ndarray, new_obj: np.ndarray) -> np.ndarray:
    """
    Calculate the fitness range using old, new solutions and all solutions in the archive

    :param old_obj: The old vars' obj
    :param new_obj: The new vars' obj
    :return: The fitness range
    """
    # Use a list to store all objectives values including the old, new obj and archive
    objs = [old_obj, new_obj] + [i[1] for i in archive.values()]
    return np.nanmax(objs, 0) - np.nanmin(objs, 0)


def cal_domination_amount(old_obj: np.ndarray, new_obj: np.ndarray, ranges=None):
    """
    The basic function for calculating the domination amount

    :param old_obj: The old vars' obj
    :param new_obj: The new vars' obj
    :param ranges: R
    :return: The domination amount
    """
    if ranges is None:
        ranges = cal_fitness_range(old_obj, new_obj)
    # Process if there is only one objective
    if all(np.array([ranges[1], old_obj[1], new_obj[1]]) == 0):
        return np.prod([abs(new_obj[0] - old_obj[0]) / ranges[0]])

    elif all(np.array([ranges[0], old_obj[0], new_obj[0]]) == 0):
        return np.prod([abs(new_obj[1] - old_obj[1]) / ranges[1]])
    else:
        return np.prod([abs(i - j) / r for i, j, r in zip(old_obj, new_obj, ranges)])


def sigmoid(x):
    return 1 / (1 + np.exp(np.array(-x, dtype=np.float64)))


if __name__ == '__main__':
    # with open('archive.pkl', 'rb') as file:
    #     archive = pickle.load(file)
    archive = {1: [np.array([0]), np.array([1,2,1])], 2: [np.array([0]), np.array([2,1,2])]}
    old_obj = np.array([1,2,1])
    new_obj = np.array([1,2,2])
    result = dominates(archive, old_obj, new_obj)
    print(result)