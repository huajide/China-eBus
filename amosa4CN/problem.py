"""Optimisation problem definition and city input loading."""

from __future__ import annotations

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
        Outputs: ObjV: [f1, f2]; CV: [cv1, cv2, cv3, cv4, ..., cvn]
        :param args: Receive if only calculate and return CVs
        :param vars_: variables
        """
        pass



import pickle

import geopandas as gpd
import pandas as pd
from dateutil import parser
from pyproj import CRS

import data_utils
from paths import SHARED_DATA_ROOT, VS_PARKING_ROOT
from simulation import SimVehicle


def validate_vs_parking_df(vs_parking_df, city_name):
    required = [
        'v_name', 'trip', 's_time', 'e_time', 'destination', 'distance',
        'avg_velocity', 'vehicle_type', 'base_energy',
    ]
    missing_columns = [name for name in required if name not in vs_parking_df.columns]
    if missing_columns:
        raise ValueError(f"{city_name}: missing columns {missing_columns}")
    null_columns = vs_parking_df[required].columns[vs_parking_df[required].isna().any()].tolist()
    if null_columns:
        raise ValueError(f"{city_name}: null values in {null_columns}")

    ordered = vs_parking_df.sort_values(['v_name', 'trip'])
    previous_end = ordered.groupby('v_name')['e_time'].shift()
    negative_gap = previous_end.notna() & (ordered['s_time'] < previous_end)
    if negative_gap.any():
        bad = ordered.loc[negative_gap, ['v_name', 'trip']].iloc[0].to_dict()
        raise ValueError(f"{city_name}: overlapping trips detected at {bad}")


class Location(Problem):
    def __init__(self, **kwargs):
        Problem.__init__(self,
                         kwargs.get('num_vars'),  # var_num
                         [0] * kwargs.get('num_vars'),  # Integer or Real
                         [0] * len(kwargs.get('cs_gdf')) + [0,0,0],  # lb
                         [1] * len(kwargs.get('cs_gdf')) + [9999,9999,9999],  # ub
                         2,  # num of f
                         1,  # num of cv
                         **kwargs
                         )
        self.num_vars = kwargs.get('num_vars')
        self.sim_v_info = kwargs.get('sim_v_info')
        self.sim_v_dict = kwargs.get('sim_v_dict')
        self.cs_gdf = kwargs.get('cs_gdf')
        self.cs_num = len(self.cs_gdf)
        self.vs_parking_df = kwargs.get('vs_parking_df')
        self.all_d2s_dict = kwargs.get('all_d2s_dict')
        self.cs_dict = kwargs.get('cs_dict')

        self.v_name = self.vs_parking_df['v_name'].to_list()
        self.trip = self.vs_parking_df['trip'].to_list()
        self.s_time = self.vs_parking_df['s_time'].to_list()
        self.e_time = self.vs_parking_df['e_time'].to_list()
        self.destination = self.vs_parking_df['destination'].to_list()
        self.distance = self.vs_parking_df['distance'].to_list()
        self.transition_distance = data_utils.derive_transition_distances_m(
            self.vs_parking_df
        ).to_list()
        self.avg_velocity = self.vs_parking_df['avg_velocity'].to_list()
        self.v_type = self.vs_parking_df['vehicle_type'].to_list()
        self.base_energy = self.vs_parking_df['base_energy'].to_list()

        self.simv_v_name = self.sim_v_info.index.to_list()
        self.simv_v_type = self.sim_v_info['vehicle_type'].to_list()

        self.large_num = self.simv_v_type.count('large')
        self.medium_num = self.simv_v_type.count('medium')
        self.small_num = self.simv_v_type.count('small')

        self.e_price = kwargs.get('e_price')
        self.reserve_energy = kwargs.get('reserve_energy', 20)
        self.default_chargers = 20


def build_city_problem(city: str, reserve_energy: float = 20, vs_root=None) -> Location:
    from sim_utils import canonical_city
    requested = city
    city = canonical_city(city)
    vs_path = (VS_PARKING_ROOT if vs_root is None else vs_root) / f"{city}.csv"
    if vs_root is not None and not vs_path.exists():
        vs_path = vs_root / f"{requested}.csv"
    if not vs_path.exists():
        raise FileNotFoundError(f"Prepared operational input is missing: {vs_path}")

    vs = pd.read_csv(vs_path)
    vs["s_time"] = vs["s_time"].apply(parser.parse)
    vs["e_time"] = vs["e_time"].apply(parser.parse)
    validate_vs_parking_df(vs, city)
    vs.sort_values("e_time", inplace=True, ignore_index=True)

    cs = gpd.read_file(
        SHARED_DATA_ROOT / "input" / "cs_gdf" / f"{city}.shp",
        crs=CRS.from_epsg(4547),
    )
    cs["lon"] = cs.geometry.x
    cs["lat"] = cs.geometry.y
    if not cs.index.equals(pd.RangeIndex(len(cs))):
        raise ValueError(f"{city}: candidate-station index must be 0..{len(cs) - 1}")

    grouped = vs.groupby("v_name").apply(lambda frame: data_utils.derive_simV_info(frame))
    sim_v_info = pd.DataFrame.from_dict(grouped.to_dict(), orient="index")
    sim_v_dict = {
        index: SimVehicle(
            index,
            row.trip,
            row.s_time,
            row.e_time,
            row.destination,
            row.distance,
            row.base_energy,
            row.avg_velocity,
        )
        for index, row in sim_v_info.iterrows()
    }

    with (SHARED_DATA_ROOT / "input" / "all_d2s_dict" / f"{city}.pkl").open("rb") as handle:
        all_d2s = pickle.load(handle)

    city_rows = pd.read_csv(SHARED_DATA_ROOT / "224cities.csv")
    match = city_rows.loc[city_rows["city"] == city, "eprice"]
    if len(match) != 1:
        raise ValueError(f"{city}: expected exactly one electricity-price row, found {len(match)}")

    cs_dict = {node_id: index for index, node_id in zip(cs.index, cs["node_id"])}
    return Location(
        num_vars=data_utils.set_var_num(cs),
        sim_v_info=sim_v_info,
        sim_v_dict=sim_v_dict,
        cs_gdf=cs,
        vs_parking_df=vs,
        all_d2s_dict=all_d2s,
        cs_dict=cs_dict,
        e_price=float(match.iloc[0]),
        reserve_energy=reserve_energy,
    )


