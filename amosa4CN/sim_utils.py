import numpy as np
from treelib import Node
import random
import zlib


DEFAULT_RANDOM_SEED = 42


def derive_seed(base_seed=DEFAULT_RANDOM_SEED, key=""):
    offset = seed_offset(str(key))
    if offset is None:
        offset = zlib.crc32(str(key).encode("utf-8"))
    return (int(base_seed) + offset) % (2 ** 32)


def set_random_seed(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    return seed


def find_keys(dict_input: dict, **kwargs):
    keys = []

    if kwargs.get('value') == 'first_key':
        reference_value = dict_input.get(min(dict_input))
        for k, v in dict_input.items():
            if v == reference_value:
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


def set_refer_vars(cs_num: int, refer_num: int = 1):
    """Return reference decisions with station flags and three model indices."""

    if refer_num == 1:
        X = [1] * cs_num  # Station
        VH = [0,0,0]  # vehicle type
        refer_vars = np.array([X + VH])
        assert refer_vars.ndim == 2, f"Invalid 'refer_vars' format'"
        return refer_vars
    else:  # When more refer_vars are needed
        refer_vars_list = [np.array([1] * cs_num + [0, 0, 0])]
        for _ in range(refer_num - 1):
            X = [random.randint(0, 1) for _ in range(cs_num)]
            if all(x == 0 for x in X):
                
                random_index = random.randint(0, cs_num - 1)
                X[random_index] = 1
            VH = [0,0,0]

            current_refer_vars = np.array(X + VH)

            assert current_refer_vars.ndim == 1, f"Invalid 'refer_vars_list' format'"
            refer_vars_list.append(current_refer_vars)

        refer_vars = np.hstack([refer_vars_list])
        assert refer_vars.ndim == 2, f"Invalid 'refer_vars_list' format'"

        return refer_vars

def canonical_city(city):
    from paths import SHARED_DATA_ROOT
    import json
    records = json.loads((SHARED_DATA_ROOT / "city_registry.json").read_text(encoding="utf-8"))
    if city not in {row["release_city"] for row in records}:
        raise KeyError(f"Unknown city: {city}")
    return city


def seed_offset(city):
    from paths import SHARED_DATA_ROOT
    import json
    records = json.loads((SHARED_DATA_ROOT / "city_registry.json").read_text(encoding="utf-8"))
    for row in records:
        if row["release_city"] == city:
            return int(row["seed_offset"])
    return None
