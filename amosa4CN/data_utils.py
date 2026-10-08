import numpy as np
import pandas as pd
from ast import literal_eval


def derive_simV_info(df: pd.DataFrame):
    # Sort
    df.sort_values('e_time', inplace=True)
    max_trip = df['trip'].max()
    trips = [i + 1 for i in range(max_trip)]
    s_time_list = df['s_time'].to_list()
    e_time_list = df['e_time'].to_list()
    destinations = df['destination'].to_list()
    distances = df['distance'].to_list()
    velocity = df['avg_velocity'].to_list()
    base_energy = df['base_energy'].to_list()

    vehicle_types = df['vehicle_type'].unique().tolist()
    priority_order = {'large': 2, 'medium': 1, 'small': 0}
    most_important_type = max(vehicle_types, key=lambda x: priority_order[x])

    return {'trip': trips,
            's_time': s_time_list,
            'e_time': e_time_list,
            'destination': destinations,
            'distance': distances,
            'avg_velocity': velocity,
            'vehicle_type': most_important_type,
            'base_energy': base_energy
            }


def derive_transition_distances_m(vs_parking: pd.DataFrame, road_factor: float = 1.2) -> pd.Series:
    """Estimate each trip-end to next-trip-origin deadhead."""
    required = {"v_name", "trip", "orientation_coords", "destination_coords"}
    missing = required.difference(vs_parking.columns)
    if missing:
        raise ValueError(f"Missing transition-distance columns: {sorted(missing)}")

    ordered = vs_parking.sort_values(["v_name", "trip"])
    origins = np.asarray(
        [literal_eval(value) if isinstance(value, str) else value
         for value in ordered["orientation_coords"]],
        dtype=np.float64,
    )
    destinations = np.asarray(
        [literal_eval(value) if isinstance(value, str) else value
         for value in ordered["destination_coords"]],
        dtype=np.float64,
    )
    if origins.shape[1:] != (2,) or destinations.shape[1:] != (2,):
        raise ValueError("Trip endpoint coordinates must be (longitude, latitude) pairs")

    distances = np.zeros(len(ordered), dtype=np.float64)
    same_vehicle = (
        ordered["v_name"].to_numpy()[:-1]
        == ordered["v_name"].to_numpy()[1:]
    )
    rows = np.flatnonzero(same_vehicle)
    if len(rows):
        start = np.radians(destinations[rows])
        end = np.radians(origins[rows + 1])
        d_lon = end[:, 0] - start[:, 0]
        d_lat = end[:, 1] - start[:, 1]
        hav = (
            np.sin(d_lat / 2.0) ** 2
            + np.cos(start[:, 1]) * np.cos(end[:, 1])
            * np.sin(d_lon / 2.0) ** 2
        )
        distances[rows] = (
            6_371_008.8 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
            * road_factor
        )

    result = pd.Series(distances, index=ordered.index, dtype=np.float64)
    return result.reindex(vs_parking.index)


def set_var_num(cs_dict: dict):
    """
    Get how many vars should be set; Note: This function will manipulate the "sim_v_info"

    :param sim_n:
    :param sim_v_info:
    :param cs_dict:
    :return: Number of variables
    """

    # The num of charging stations
    x_num = len(cs_dict)

    vh_num = 3  # vehicle types
    return x_num + vh_num



