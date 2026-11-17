"""Numba operations evaluator with annual vehicle/fuel-cycle objectives."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Keep compiled kernels reusable even when the source tree is read-only.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "cn_reuse_numba_cache"),
)

import numpy as np
import pandas as pd
from numba import config, get_num_threads, njit, prange, set_num_threads

config.CACHE_DIR = os.environ["NUMBA_CACHE_DIR"]

from vehicle_type import (
    VehicleTypes,
    baseline_mass,
    beta_HVAC,
    beta_P_l,
    beta_SOC,
    people_mass,
    vehi_type_num,
)
from vehicle_type import (
    CHARGING_EFFICIENCY, GRID_G_CO2E_PER_KWH, STATION_T_CO2E_PER_YEAR,
    VEHICLE_CYCLE_T_PER_YEAR,
)


_DAY_US = 86_400_000_000
_HOUR_US = 3_600_000_000.0
_MAX_NUMBA_THREADS = get_num_threads()


@dataclass(frozen=True)
class CNFastData:
    num_stations: int
    num_destinations: int
    num_trips: int
    num_vehicles: int
    vehicle_counts: np.ndarray
    vehicle_idx: np.ndarray
    vehicle_type_idx: np.ndarray
    vehicle_type_by_vehicle: np.ndarray
    vehicle_route_idx: np.ndarray
    trip_route_idx: np.ndarray
    trip_idx: np.ndarray
    s_us: np.ndarray
    e_us: np.ndarray
    first_s_us: np.ndarray
    next_pos: np.ndarray
    interval_min: np.ndarray
    distance_m: np.ndarray
    distance_km: np.ndarray
    transition_distance_m: np.ndarray
    transition_distance_km: np.ndarray
    base_energy_rate_kwh_km: np.ndarray
    base_energy_kwh: np.ndarray
    destination_idx: np.ndarray
    station_candidate_idx: np.ndarray
    station_candidate_distance_m: np.ndarray
    model_counts: np.ndarray
    battery_kwh: np.ndarray
    mass_kg: np.ndarray
    mass_energy_kwh_km: np.ndarray
    per_emission_g_km: np.ndarray
    purchase_price_yuan: np.ndarray
    fix_cost_yuan_year: np.ndarray


@dataclass(frozen=True)
class CNFastResult:
    objectives: np.ndarray
    diagnostics: np.ndarray
    metrics: np.ndarray
    timeouts: np.ndarray | None = None
    tree_initializations: np.ndarray | None = None


METRIC_COLUMNS = (
    "station_count",
    "fast_chargers",
    "slow_chargers",
    "extra_large",
    "extra_medium",
    "extra_small",
    "trip_energy_kwh",
    "d2s_energy_kwh",
    "wait_time_h",
    "remaining_timeout_h",
    "operation_emission_g_day",
    "vehicle_cost_yuan_year",
    "station_cost_yuan_year",
    "charger_cost_yuan_year",
    "energy_cost_yuan_year",
    "vehicle_cycle_emission_t_year",
    "fuel_cycle_emission_t_year",
    "station_emission_t_year",
)


def build_fast_data(problem) -> CNFastData:
    """Encode a city's trip, vehicle and station inputs as arrays."""
    vehicle_names = list(problem.simv_v_name)
    vehicle_to_idx = {name: i for i, name in enumerate(vehicle_names)}
    vehicle_idx = np.asarray([vehicle_to_idx[name] for name in problem.v_name], dtype=np.int32)
    type_map = {"large": 0, "medium": 1, "small": 2}
    vehicle_type_idx = np.asarray([type_map[value] for value in problem.v_type], dtype=np.int8)
    vehicle_type_by_vehicle = np.full(len(vehicle_names), -1, dtype=np.int8)
    vehicle_type_by_vehicle[vehicle_idx] = vehicle_type_idx
    trip_idx = np.asarray(problem.trip, dtype=np.int32)
    routes = pd.Series(problem.vs_parking_df["route_id"]).astype(str)
    trip_route_idx = pd.factorize(routes, sort=False)[0].astype(np.int32)
    vehicle_route_idx = np.full(len(vehicle_names), -1, dtype=np.int32)
    for vehicle, route in zip(vehicle_idx, trip_route_idx):
        if vehicle_route_idx[vehicle] == -1:
            vehicle_route_idx[vehicle] = route

    s_time = pd.to_datetime(pd.Series(problem.s_time))
    e_time = pd.to_datetime(pd.Series(problem.e_time))
    s_us = s_time.to_numpy(dtype="datetime64[us]").astype(np.int64)
    e_us = e_time.to_numpy(dtype="datetime64[us]").astype(np.int64)

    by_vehicle_trip: dict[tuple[int, int], int] = {}
    first_s_us = np.zeros(len(vehicle_names), dtype=np.int64)
    for pos, (vehicle, local_trip) in enumerate(zip(vehicle_idx, trip_idx)):
        key = (int(vehicle), int(local_trip))
        if key in by_vehicle_trip:
            raise ValueError(f"Duplicate vehicle/trip key: {key}")
        by_vehicle_trip[key] = pos
        if int(local_trip) == 1:
            first_s_us[int(vehicle)] = s_us[pos]
    if np.any(first_s_us == 0):
        raise ValueError("Every simulated vehicle must have a trip numbered 1")

    next_pos = np.full(len(vehicle_idx), -1, dtype=np.int32)
    for pos, (vehicle, local_trip) in enumerate(zip(vehicle_idx, trip_idx)):
        next_pos[pos] = by_vehicle_trip.get((int(vehicle), int(local_trip) + 1), -1)
    interval_min = np.full(len(vehicle_idx), 9999.0, dtype=np.float64)
    has_next = next_pos >= 0
    interval_min[has_next] = (
        s_us[next_pos[has_next]] - e_us[has_next]
    ) / 60_000_000.0
    if np.any(interval_min < 0):
        raise ValueError("Negative trip intervals are not supported")

    destinations = np.asarray(problem.destination, dtype=object)
    unique_destinations = pd.unique(pd.Series(destinations))
    destination_to_idx = {value: i for i, value in enumerate(unique_destinations)}
    destination_idx = np.asarray(
        [destination_to_idx[value] for value in destinations], dtype=np.int32
    )
    max_candidates = max(
        (len(problem.all_d2s_dict.get(value, [])) for value in unique_destinations),
        default=0,
    )
    if max_candidates == 0:
        raise ValueError("No charging-station candidates are available")
    station_candidate_idx = np.full(
        (len(unique_destinations), max_candidates), -1, dtype=np.int32
    )
    station_candidate_distance_m = np.full(
        (len(unique_destinations), max_candidates), 99_000.0, dtype=np.float64
    )
    for destination, dest_idx in destination_to_idx.items():
        for write, record in enumerate(problem.all_d2s_dict.get(destination, [])):
            station_idx = problem.cs_dict.get(record["station"])
            if station_idx is None:
                raise KeyError(f"Unknown candidate station: {record['station']}")
            station_candidate_idx[dest_idx, write] = int(station_idx)
            station_candidate_distance_m[dest_idx, write] = float(record["distance"])

    model_counts = np.asarray(
        [vehi_type_num("large"), vehi_type_num("medium"), vehi_type_num("small")],
        dtype=np.int32,
    )
    max_models = int(model_counts.max())
    battery_kwh = np.zeros((3, max_models), dtype=np.float64)
    mass_kg = np.zeros((3, max_models), dtype=np.float64)
    per_emission_g_km = np.zeros((3, max_models), dtype=np.float64)
    purchase_price_yuan = np.zeros((3, max_models), dtype=np.float64)
    fix_cost_yuan_year = np.zeros((3, max_models), dtype=np.float64)
    type_names = ("large", "medium", "small")
    for vehicle_type, type_name in enumerate(type_names):
        for model in range(model_counts[vehicle_type]):
            vehicle = VehicleTypes(model, type_name)
            battery_kwh[vehicle_type, model] = vehicle.battery
            mass_kg[vehicle_type, model] = vehicle.mass
            per_emission_g_km[vehicle_type, model] = vehicle.per_emission
            purchase_price_yuan[vehicle_type, model] = vehicle.price
            fix_cost_yuan_year[vehicle_type, model] = vehicle.fix_cost

    distance_m = pd.to_numeric(
        pd.Series(problem.distance), errors="raise"
    ).to_numpy(dtype=np.float64)
    if np.any(distance_m <= 0.0):
        raise ValueError("Trip distances must be positive for exact energy evaluation")
    distance_km = distance_m / 1000.0
    transition_distance_m = np.asarray(
        problem.transition_distance, dtype=np.float64
    )
    if np.any(transition_distance_m < 0.0):
        raise ValueError("Transition distances cannot be negative")
    transition_distance_km = transition_distance_m / 1000.0
    base_energy_kwh = pd.to_numeric(
        pd.Series(problem.base_energy), errors="raise"
    ).to_numpy(dtype=np.float64)
    mass_energy_kwh_km = (
        (mass_kg - baseline_mass) / people_mass * beta_P_l
    )

    return CNFastData(
        num_stations=int(problem.cs_num),
        num_destinations=len(unique_destinations),
        num_trips=len(vehicle_idx),
        num_vehicles=len(vehicle_names),
        vehicle_counts=np.asarray(
            [problem.large_num, problem.medium_num, problem.small_num], dtype=np.int32
        ),
        vehicle_idx=vehicle_idx,
        vehicle_type_idx=vehicle_type_idx,
        vehicle_type_by_vehicle=vehicle_type_by_vehicle,
        vehicle_route_idx=vehicle_route_idx,
        trip_route_idx=trip_route_idx,
        trip_idx=trip_idx,
        s_us=s_us,
        e_us=e_us,
        first_s_us=first_s_us,
        next_pos=next_pos,
        interval_min=interval_min,
        distance_m=distance_m,
        distance_km=distance_km,
        transition_distance_m=transition_distance_m,
        transition_distance_km=transition_distance_km,
        base_energy_rate_kwh_km=base_energy_kwh / distance_m * 1000.0,
        base_energy_kwh=base_energy_kwh,
        destination_idx=destination_idx,
        station_candidate_idx=station_candidate_idx,
        station_candidate_distance_m=station_candidate_distance_m,
        model_counts=model_counts,
        battery_kwh=battery_kwh,
        mass_kg=mass_kg,
        mass_energy_kwh_km=mass_energy_kwh_km,
        per_emission_g_km=per_emission_g_km,
        purchase_price_yuan=purchase_price_yuan,
        fix_cost_yuan_year=fix_cost_yuan_year,
    )


def build_degradation_masks(data: CNFastData, share: float, seed: int,
                            capacity_factor: float = 0.9, *,
                            hvac_power: float = -1.0,
                            energy_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Select the requested fleet share by type, excluding routes needing new batteries."""
    if not 0.0 <= share <= 1.0 or not 0.0 < capacity_factor <= 1.0:
        raise ValueError("Invalid battery degradation setting")
    masks = np.zeros((3, data.battery_kwh.shape[1], data.num_vehicles), dtype=np.uint8)
    feasible = np.ones((3, data.battery_kwh.shape[1]), dtype=np.uint8)
    if share == 0.0:
        return masks, feasible
    priority = np.random.default_rng(seed).random(data.num_vehicles)
    base_energy = data.base_energy_kwh
    if hvac_power >= 0:
        base_energy = base_energy + beta_HVAC * (hvac_power - 20.0) * data.distance_km
    for vehicle_type in range(3):
        vehicles = np.flatnonzero(data.vehicle_type_by_vehicle == vehicle_type)
        target = int(np.floor(len(vehicles) * share + 0.5))
        for model in range(data.model_counts[vehicle_type]):
            battery = data.battery_kwh[vehicle_type, model] * capacity_factor
            trip_energy = np.maximum(
                0.0,
                base_energy + (
                    beta_SOC * 100.0 + data.mass_energy_kwh_km[vehicle_type, model]
                ) * data.distance_km,
            ) * energy_factor
            long_routes = np.unique(data.trip_route_idx[
                (data.vehicle_type_idx == vehicle_type) & (trip_energy > battery + 1e-12)
            ])
            long_vehicles = np.unique(data.vehicle_idx[
                np.isin(data.trip_route_idx, long_routes)
            ])
            eligible = vehicles[~np.isin(vehicles, long_vehicles)]
            if len(eligible) < target:
                feasible[vehicle_type, model] = 0
                continue
            if target == 0:
                continue
            route_ids, inverse, counts = np.unique(
                data.vehicle_route_idx[eligible], return_inverse=True, return_counts=True
            )
            quotas = counts * target / len(eligible)
            allocations = np.floor(quotas).astype(int)
            remainder = target - allocations.sum()
            order = np.lexsort((route_ids, -(quotas - allocations)))
            allocations[order[:remainder]] += 1
            for route_pos, count in enumerate(allocations):
                route_vehicles = eligible[inverse == route_pos]
                selected = route_vehicles[np.argsort(priority[route_vehicles], kind="stable")[:count]]
                masks[vehicle_type, model, selected] = 1
    return masks, feasible


def _normalise_population(population: np.ndarray, data: CNFastData) -> np.ndarray:
    population = np.asarray(population)
    if population.ndim == 1:
        population = population.reshape(1, -1)
    if population.shape[1] != data.num_stations + 3:
        raise ValueError(
            f"Expected {data.num_stations + 3} variables, got {population.shape[1]}"
        )
    rounded = np.rint(population).astype(np.int32)
    if np.any((rounded[:, :data.num_stations] < 0) | (rounded[:, :data.num_stations] > 1)):
        raise ValueError("Station variables must be binary")
    for vehicle_type in range(3):
        values = rounded[:, data.num_stations + vehicle_type]
        if np.any((values < 0) | (values >= data.model_counts[vehicle_type])):
            raise ValueError(f"Invalid model index for vehicle type {vehicle_type}")
    return rounded


@njit(cache=True, parallel=True)
def _choose_station_matrices_numba(
    population: np.ndarray,
    num_stations: int,
    station_candidate_idx: np.ndarray,
    station_candidate_distance_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    population_size = population.shape[0]
    destination_count = station_candidate_idx.shape[0]
    candidate_count = station_candidate_idx.shape[1]
    chosen_idx = np.full((population_size, destination_count), -1, dtype=np.int32)
    chosen_distance = np.full(
        (population_size, destination_count), 99_000.0, dtype=np.float64
    )
    for solution in prange(population_size):
        for destination in range(destination_count):
            for candidate in range(candidate_count):
                station = station_candidate_idx[destination, candidate]
                if station < 0:
                    break
                if station < num_stations and population[solution, station] == 1:
                    chosen_idx[solution, destination] = station
                    chosen_distance[solution, destination] = (
                        station_candidate_distance_m[destination, candidate]
                    )
                    break
    return chosen_idx, chosen_distance


def choose_station_matrices(
    population: np.ndarray, data: CNFastData
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the scalar model's first-built-candidate rule."""
    return _choose_station_matrices_numba(
        population,
        data.num_stations,
        data.station_candidate_idx,
        data.station_candidate_distance_m,
    )


@njit(cache=True)
def _hours_to_us(hours: float) -> int:
    return int(np.rint(hours * _HOUR_US))


@njit(cache=True)
def _select_charger(
    station: int,
    arrival_us: int,
    time_gap_h: float,
    slow_time_h: float,
    fast_end_us: np.ndarray,
    slow_end_us: np.ndarray,
    fast_queue: np.ndarray,
    slow_queue: np.ndarray,
    fast_count: int,
    slow_count: int,
) -> tuple[int, int, int]:
    first_fast = -1
    first_slow = -1
    for channel in range(fast_count):
        if fast_queue[station, channel] == 0 or fast_end_us[station, channel] <= arrival_us:
            first_fast = channel
            break
    for channel in range(slow_count):
        if slow_queue[station, channel] == 0 or slow_end_us[station, channel] <= arrival_us:
            first_slow = channel
            break

    if first_fast >= 0 and first_slow >= 0:
        if time_gap_h > slow_time_h:
            return 1, first_slow, 0
        return 0, first_fast, 0
    if first_fast >= 0:
        return 0, first_fast, 0
    if first_slow >= 0:
        return 1, first_slow, 0

    # Use the first fast charger's queue length as the reference.
    reference_queue = fast_queue[station, 0]
    selected_fast = -1
    selected_slow = -1
    for channel in range(fast_count):
        if fast_queue[station, channel] == reference_queue:
            selected_fast = channel
            break
    for channel in range(slow_count):
        if slow_queue[station, channel] == reference_queue:
            selected_slow = channel
            break
    if selected_fast >= 0 and selected_slow >= 0:
        if time_gap_h > slow_time_h:
            return 1, selected_slow, 1
        return 0, selected_fast, 1
    if selected_fast >= 0:
        return 0, selected_fast, 1
    return 1, selected_slow, 1


@njit(cache=True)
def _piecewise_charge_time(start_soc, end_soc, soc_per_hour):
    duration = 0.0
    soc = start_soc
    if soc < 0.8:
        upper = min(end_soc, 0.8)
        duration += max(0.0, upper - soc) / soc_per_hour
        soc = upper
    if soc < 0.85:
        upper = min(end_soc, 0.85)
        duration += max(0.0, upper - soc) / (0.625 * soc_per_hour)
        soc = upper
    return duration + max(0.0, end_soc - soc) / (0.46875 * soc_per_hour)


@njit(cache=True)
def _piecewise_charged_soc(start_soc, duration_h, soc_per_hour):
    soc = start_soc
    if soc < 0.8:
        segment_h = (0.8 - soc) / soc_per_hour
        if duration_h <= segment_h:
            return soc + duration_h * soc_per_hour
        duration_h -= segment_h
        soc = 0.8
    if soc < 0.85:
        segment_h = (0.85 - soc) / (0.625 * soc_per_hour)
        if duration_h <= segment_h:
            return soc + duration_h * 0.625 * soc_per_hour
        duration_h -= segment_h
        soc = 0.85
    return soc + duration_h * 0.46875 * soc_per_hour


@njit(cache=True)
def _apply_charging(
    arrival_us: int,
    arrival_soc: float,
    min_charging_h: float,
    time_gap_h: float,
    battery_kwh: float,
    power_kw: float,
    end_us_old: int,
    queue_old: int,
    is_waiting: int,
    max_energy_ratio: float,
    nonlinear: bool,
) -> tuple[float, float, float, int, int]:
    target_us = arrival_us + _hours_to_us(time_gap_h)
    full_charge_h = (
        _piecewise_charge_time(arrival_soc, 1.0, power_kw / battery_kwh)
        if nonlinear else max_energy_ratio * battery_kwh / power_kw
    )
    if is_waiting == 0:
        start_us = arrival_us
        queue_new = 1
        if time_gap_h >= min_charging_h:
            if (time_gap_h >= full_charge_h if nonlinear
                    else time_gap_h * power_kw / battery_kwh >= max_energy_ratio):
                duration_h = full_charge_h if nonlinear else (1.0 - arrival_soc) * battery_kwh / power_kw
                end_us = start_us + _hours_to_us(duration_h)
            else:
                end_us = target_us
            timeout_h = 0.0
        else:
            end_us = start_us + _hours_to_us(min_charging_h)
            timeout_h = min_charging_h - time_gap_h
        wait_h = 0.0
    else:
        start_us = end_us_old
        queue_new = queue_old + 1
        real_gap_h = (target_us - start_us) / _HOUR_US
        if real_gap_h >= 0.0:
            if real_gap_h >= min_charging_h:
                if (real_gap_h >= full_charge_h if nonlinear
                        else real_gap_h * power_kw / battery_kwh >= max_energy_ratio):
                    duration_h = full_charge_h if nonlinear else (1.0 - arrival_soc) * battery_kwh / power_kw
                    end_us = start_us + _hours_to_us(duration_h)
                else:
                    end_us = target_us
                timeout_h = 0.0
            else:
                end_us = start_us + _hours_to_us(min_charging_h)
                timeout_h = min_charging_h - real_gap_h
        else:
            end_us = start_us + _hours_to_us(min_charging_h)
            timeout_h = (end_us - target_us) / _HOUR_US
        wait_h = (start_us - arrival_us) / _HOUR_US

    charged_h = (end_us - start_us) / _HOUR_US
    charged_soc = (
        _piecewise_charged_soc(arrival_soc, charged_h, power_kw / battery_kwh)
        if nonlinear else charged_h * power_kw / battery_kwh + arrival_soc
    )
    return charged_soc, timeout_h, wait_h, end_us, queue_new


@njit(cache=True)
def _stable_timeout_order_desc(values: np.ndarray, count: int) -> np.ndarray:
    order = np.arange(count, dtype=np.int32)
    temp = np.empty(count, dtype=np.int32)
    width = 1
    while width < count:
        left = 0
        while left < count:
            middle = min(left + width, count)
            right = min(left + 2 * width, count)
            i = left
            j = middle
            write = left
            while i < middle and j < right:
                left_idx = order[i]
                right_idx = order[j]
                if values[left_idx] > values[right_idx] or (
                    values[left_idx] == values[right_idx] and left_idx < right_idx
                ):
                    temp[write] = left_idx
                    i += 1
                else:
                    temp[write] = right_idx
                    j += 1
                write += 1
            while i < middle:
                temp[write] = order[i]
                i += 1
                write += 1
            while j < right:
                temp[write] = order[j]
                j += 1
                write += 1
            left += 2 * width
        for i in range(count):
            order[i] = temp[i]
        width *= 2
    return order


@njit(cache=True)
def _reschedule_one(
    timeout_row: np.ndarray,
    vehicle_idx: np.ndarray,
    vehicle_type_idx: np.ndarray,
    distance_m: np.ndarray,
    base_energy_kwh: np.ndarray,
    batteries: np.ndarray,
    masses: np.ndarray,
    collect_substitutions: int,
    energy_factor: float = 1.0,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    trip_count = timeout_row.shape[0]
    survivor_orig = np.empty(trip_count, dtype=np.int32)
    survivor_vehicle = np.empty(trip_count, dtype=np.int32)
    survivor_type = np.empty(trip_count, dtype=np.int8)
    survivor_distance = np.empty(trip_count, dtype=np.float64)
    survivor_base_energy = np.empty(trip_count, dtype=np.float64)
    timeout_work = np.empty(trip_count, dtype=np.float64)
    survivor_count = 0
    remaining_above_threshold = 0
    for trip in range(trip_count):
        timeout = timeout_row[trip]
        if timeout > 0.0:
            survivor_orig[survivor_count] = trip
            survivor_vehicle[survivor_count] = vehicle_idx[trip]
            survivor_type[survivor_count] = vehicle_type_idx[trip]
            survivor_distance[survivor_count] = distance_m[trip]
            survivor_base_energy[survivor_count] = base_energy_kwh[trip]
            timeout_work[survivor_count] = timeout
            if timeout > 0.05:
                remaining_above_threshold += 1
            survivor_count += 1

    extras = np.zeros(3, dtype=np.int32)
    substituted = np.empty(trip_count, dtype=np.int32)
    substitution_count = 0
    if survivor_count == 0:
        return extras, 0.0, substituted, 0

    order = _stable_timeout_order_desc(timeout_work, survivor_count)
    saved_time = 0.0
    for rank in range(survivor_count):
        idx = order[rank]
        if survivor_vehicle[idx] == -1:
            continue
        if timeout_work[idx] != 0.0:
            vehicle_type = survivor_type[idx]
            extras[vehicle_type] += 1
            remaining_energy = batteries[vehicle_type]
            mass_delta = (masses[vehicle_type] - baseline_mass) / people_mass * beta_P_l
            selected_vehicle = survivor_vehicle[idx]
            for idx2 in range(idx, survivor_count):
                if survivor_vehicle[idx2] != selected_vehicle:
                    continue
                energy = max(
                    0.0,
                    survivor_base_energy[idx2]
                    + (beta_SOC * 100.0 + mass_delta)
                    * survivor_distance[idx2]
                    / 1000.0,
                )
                energy *= energy_factor
                if energy <= remaining_energy:
                    remaining_energy -= energy
                    survivor_vehicle[idx2] = -1
                    saved_time += timeout_work[idx2]
                    if timeout_work[idx2] > 0.05:
                        remaining_above_threshold -= 1
                    timeout_work[idx2] = 0.0
                    if collect_substitutions == 1:
                        substituted[substitution_count] = survivor_orig[idx2]
                        substitution_count += 1
                else:
                    break
        if remaining_above_threshold == 0:
            break
    return extras, saved_time, substituted, substitution_count


@njit(cache=True, parallel=True)
def _evaluate_core(
    population: np.ndarray,
    chosen_station_idx: np.ndarray,
    chosen_distance_m: np.ndarray,
    num_stations: int,
    num_vehicles: int,
    vehicle_idx: np.ndarray,
    vehicle_type_idx: np.ndarray,
    vehicle_type_by_vehicle: np.ndarray,
    trip_idx: np.ndarray,
    s_us: np.ndarray,
    e_us: np.ndarray,
    first_s_us: np.ndarray,
    next_pos: np.ndarray,
    interval_min: np.ndarray,
    distance_m: np.ndarray,
    distance_km: np.ndarray,
    transition_distance_m: np.ndarray,
    transition_distance_km: np.ndarray,
    base_energy_rate_kwh_km: np.ndarray,
    base_energy_kwh: np.ndarray,
    destination_idx: np.ndarray,
    battery_kwh: np.ndarray,
    mass_kg: np.ndarray,
    mass_energy_kwh_km: np.ndarray,
    per_emission_g_km: np.ndarray,
    reserve_energy: float,
    min_interval: float,
    degradation_factor: float,
    degradation_masks: np.ndarray,
    degradation_feasible: np.ndarray,
    degraded_capacity_factor: float,
    power_factor: float,
    slow_power_factor: float,
    fast_count: int,
    slow_count: int,
    energy_factor: float,
    hvac_power: float,
    charging_profile: int,
    store_timeouts: int,
    timeout_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    population_size = population.shape[0]
    trip_count = vehicle_idx.shape[0]
    metrics = np.zeros((population_size, 5), dtype=np.float64)
    fast_used_count = np.zeros((population_size, num_stations), dtype=np.int16)
    slow_used_count = np.zeros((population_size, num_stations), dtype=np.int16)
    invalid = np.zeros(population_size, dtype=np.int8)
    rescheduling = np.zeros((population_size, 5), dtype=np.float64)
    tree_initializations = np.zeros(population_size, dtype=np.int32)

    for solution in prange(population_size):
        selected_battery = np.empty(3, dtype=np.float64)
        selected_mass = np.empty(3, dtype=np.float64)
        selected_mass_energy = np.empty(3, dtype=np.float64)
        selected_emission = np.empty(3, dtype=np.float64)
        for vehicle_type in range(3):
            model = int(population[solution, num_stations + vehicle_type])
            if degradation_feasible[vehicle_type, model] == 0:
                invalid[solution] = 1
            selected_battery[vehicle_type] = (
                battery_kwh[vehicle_type, model] * degradation_factor
            )
            selected_mass[vehicle_type] = mass_kg[vehicle_type, model]
            selected_mass_energy[vehicle_type] = mass_energy_kwh_km[
                vehicle_type, model
            ]
            selected_emission[vehicle_type] = per_emission_g_km[
                vehicle_type, model
            ]
        if invalid[solution] == 1:
            continue

        scenario_base_energy = base_energy_kwh
        reference_base_energy = base_energy_kwh
        if hvac_power >= 0.0:
            scenario_base_energy = np.empty(trip_count, dtype=np.float64)
            reference_base_energy = np.empty(trip_count, dtype=np.float64)
            for trip in range(trip_count):
                vehicle_type = vehicle_type_idx[trip]
                distance = distance_km[trip]
                non_hvac_energy = (
                    base_energy_kwh[trip]
                    + (beta_SOC * 100.0 + selected_mass_energy[vehicle_type]
                       - beta_HVAC * 20.0) * distance
                )
                max_power = np.floor(
                    (selected_battery[vehicle_type] - non_hvac_energy)
                    / (beta_HVAC * distance) + 1e-10
                )
                if max_power < 0.0:
                    invalid[solution] = 1
                    break
                applied_power = min(hvac_power, max_power)
                scenario_base_energy[trip] = (
                    base_energy_kwh[trip]
                    + beta_HVAC * (applied_power - 20.0) * distance
                )
                reference_base_energy[trip] = (
                    base_energy_kwh[trip]
                    + beta_HVAC * (min(4.0, max_power) - 20.0) * distance
                )
            if invalid[solution] == 1:
                continue

        soc_by_vehicle = np.ones(num_vehicles, dtype=np.float64)
        active_battery = np.empty(num_vehicles, dtype=np.float64)
        for vehicle in range(num_vehicles):
            vehicle_type = vehicle_type_by_vehicle[vehicle]
            model = int(population[solution, num_stations + vehicle_type])
            active_battery[vehicle] = selected_battery[vehicle_type] * (
                degraded_capacity_factor if degradation_masks[vehicle_type, model, vehicle] else 1.0
            )
        replacement_start = np.zeros(trip_count, dtype=np.uint8)
        segment_first_us = first_s_us.copy()
        replacement_count = np.zeros(3, dtype=np.int32)
        timeout_row = np.zeros(trip_count, dtype=np.float64)
        fast_end_us = np.zeros((num_stations, fast_count), dtype=np.int64)
        slow_end_us = np.zeros((num_stations, slow_count), dtype=np.int64)
        fast_queue = np.zeros((num_stations, fast_count), dtype=np.int32)
        slow_queue = np.zeros((num_stations, slow_count), dtype=np.int32)
        fast_used = np.zeros((num_stations, fast_count), dtype=np.uint8)
        slow_used = np.zeros((num_stations, slow_count), dtype=np.uint8)
        trip_energy_sum = 0.0
        d2s_energy_sum = 0.0
        wait_time_sum = 0.0
        emission_sum = 0.0
        energy_delta_sum = 0.0
        timeout_sum = 0.0
        fast_power = 100.0 * power_factor
        slow_power = 50.0 * slow_power_factor

        for trip in range(trip_count):
            if trip_idx[trip] == 1:
                tree_initializations[solution] += 1
            vehicle = vehicle_idx[trip]
            vehicle_type = vehicle_type_idx[trip]
            if replacement_start[trip] == 1:
                active_battery[vehicle] = selected_battery[vehicle_type]
            battery = active_battery[vehicle]
            mass_delta = selected_mass_energy[vehicle_type]
            if replacement_start[trip] == 1:
                segment_first_us[vehicle] = s_us[trip]
            previous_soc = 1.0 if trip_idx[trip] == 1 or replacement_start[trip] == 1 else soc_by_vehicle[vehicle]
            trip_energy = max(
                0.0,
                scenario_base_energy[trip]
                + (beta_SOC * previous_soc * 100.0 + mass_delta)
                * distance_km[trip],
            ) * energy_factor
            soc_trip = previous_soc - trip_energy / battery
            if soc_trip < -1e-12:
                invalid[solution] = 1
                break
            transition_energy = transition_distance_km[trip] * max(
                0.0,
                (base_energy_rate_kwh_km[trip] if hvac_power < 0.0
                 else scenario_base_energy[trip] / distance_km[trip])
                + beta_SOC * soc_trip * 100.0
                + mass_delta,
            ) * energy_factor
            # Compare energy on the executed path, without a second simulation.
            trip_energy_delta = trip_energy - max(
                0.0, reference_base_energy[trip]
                + (beta_SOC * previous_soc * 100.0 + mass_delta) * distance_km[trip],
            )
            reference_rate = max(
                0.0, reference_base_energy[trip] / distance_km[trip]
                + beta_SOC * soc_trip * 100.0 + mass_delta,
            )
            transition_energy_delta = (
                transition_energy - transition_distance_km[trip] * reference_rate
            )

            next_trip = next_pos[trip]
            if next_trip >= 0:
                next_energy = max(
                    0.0,
                    scenario_base_energy[next_trip]
                    + (beta_SOC * soc_trip * 100.0 + mass_delta)
                    * distance_km[next_trip],
                ) * energy_factor
                next_energy += transition_energy
            else:
                next_energy = 99_999.0
            charging_demand = (
                soc_trip * battery <= next_energy + reserve_energy
                or interval_min[trip] >= min_interval
            )
            trip_energy_sum += trip_energy
            energy_delta_sum += trip_energy_delta

            if not charging_demand:
                next_soc = soc_trip - transition_energy / battery
                if next_trip >= 0:
                    next_trip_energy = max(
                        0.0,
                        scenario_base_energy[next_trip]
                        + (beta_SOC * next_soc * 100.0 + mass_delta)
                        * distance_km[next_trip],
                    ) * energy_factor
                    if next_soc * battery < next_trip_energy - 1e-12:
                        replacement_start[next_trip] = 1
                        replacement_count[vehicle_type] += 1
                        soc_by_vehicle[vehicle] = soc_trip
                        emission_sum += distance_km[trip] * selected_emission[vehicle_type]
                        continue
                soc_by_vehicle[vehicle] = next_soc
                if next_soc < -1e-12:
                    invalid[solution] = 1
                    break
                d2s_energy_sum += transition_energy
                energy_delta_sum += transition_energy_delta
                emission_sum += (distance_km[trip] + transition_distance_km[trip]) * selected_emission[vehicle_type]
                continue

            destination = destination_idx[trip]
            station = chosen_station_idx[solution, destination]
            if station < 0:
                invalid[solution] = 1
                break
            d2s_m = chosen_distance_m[solution, destination]
            d2s_km = d2s_m / 1000.0
            rate = max(
                0.0,
                (base_energy_rate_kwh_km[trip] if hvac_power < 0.0
                 else scenario_base_energy[trip] / distance_km[trip])
                + beta_SOC * soc_trip * 100.0
                + mass_delta,
            )
            d2s_energy_one_way = d2s_km * rate * energy_factor
            d2s_energy_delta = d2s_energy_one_way - d2s_km * reference_rate
            arrival_soc = soc_trip - d2s_energy_one_way / battery
            if arrival_soc < -1e-12:
                if next_trip < 0:
                    invalid[solution] = 1
                    break
                replacement_start[next_trip] = 1
                replacement_count[vehicle_type] += 1
                soc_by_vehicle[vehicle] = soc_trip
                emission_sum += distance_km[trip] * selected_emission[vehicle_type]
                continue
            travel_h = d2s_km / 30.0
            arrival_us = e_us[trip] + _hours_to_us(travel_h)

            if next_trip >= 0:
                transition_h = transition_distance_km[trip] / 30.0
                time_gap_h = (
                    (s_us[next_trip] - arrival_us) / _HOUR_US
                    - travel_h - transition_h
                )
                remaining_capacity = (
                    battery - energy_factor * beta_SOC * 100.0 * distance_km[next_trip]
                )
                if remaining_capacity <= 0.0:
                    invalid[solution] = 1
                    break
                required_origin_soc = max(
                    0.0,
                    energy_factor * (scenario_base_energy[next_trip]
                     + mass_delta * distance_km[next_trip]) / remaining_capacity,
                )
                required_station_soc = required_origin_soc + (
                    transition_energy + d2s_energy_one_way
                ) / battery
                if required_station_soc > 1.0 + 1e-12:
                    replacement_start[next_trip] = 1
                    replacement_count[vehicle_type] += 1
                    soc_by_vehicle[vehicle] = soc_trip
                    emission_sum += distance_km[trip] * selected_emission[vehicle_type]
                    continue
                min_energy_ratio = min(
                    1.0 - arrival_soc,
                    max(0.05, required_station_soc - arrival_soc + 0.05),
                )
            else:
                time_gap_h = (
                    segment_first_us[vehicle] - arrival_us + _DAY_US
                ) / _HOUR_US - travel_h
                min_energy_ratio = 1.0 - arrival_soc

            max_energy_ratio = 1.0 - arrival_soc
            nonlinear = charging_profile == 1
            fast_time_h = (
                _piecewise_charge_time(arrival_soc, arrival_soc + min_energy_ratio, fast_power / battery)
                if nonlinear else battery * min_energy_ratio / fast_power
            )
            slow_time_h = (
                _piecewise_charge_time(arrival_soc, arrival_soc + min_energy_ratio, slow_power / battery)
                if nonlinear else battery * min_energy_ratio / slow_power
            )
            channel_type, channel, is_waiting = _select_charger(
                station,
                arrival_us,
                time_gap_h,
                slow_time_h,
                fast_end_us,
                slow_end_us,
                fast_queue,
                slow_queue,
                fast_count,
                slow_count,
            )
            if channel_type == 0:
                soc_charged, timeout_h, wait_h, end_us, queue_new = _apply_charging(
                    arrival_us,
                    arrival_soc,
                    fast_time_h,
                    time_gap_h,
                    battery,
                    fast_power,
                    fast_end_us[station, channel],
                    fast_queue[station, channel],
                    is_waiting,
                    max_energy_ratio,
                    nonlinear,
                )
                fast_end_us[station, channel] = end_us
                fast_queue[station, channel] = queue_new
                fast_used[station, channel] = 1
            else:
                soc_charged, timeout_h, wait_h, end_us, queue_new = _apply_charging(
                    arrival_us,
                    arrival_soc,
                    slow_time_h,
                    time_gap_h,
                    battery,
                    slow_power,
                    slow_end_us[station, channel],
                    slow_queue[station, channel],
                    is_waiting,
                    max_energy_ratio,
                    nonlinear,
                )
                slow_end_us[station, channel] = end_us
                slow_queue[station, channel] = queue_new
                slow_used[station, channel] = 1

            next_soc = (
                soc_charged - d2s_energy_one_way / battery
                - transition_energy / battery
            )
            if next_trip >= 0:
                next_trip_energy = max(
                    0.0,
                    scenario_base_energy[next_trip]
                    + (beta_SOC * next_soc * 100.0 + mass_delta)
                    * distance_km[next_trip],
                ) * energy_factor
                if next_soc * battery < next_trip_energy - 1e-12:
                    replacement_start[next_trip] = 1
                    replacement_count[vehicle_type] += 1
                    soc_by_vehicle[vehicle] = soc_charged
                    d2s_energy_sum += d2s_energy_one_way
                    energy_delta_sum += d2s_energy_delta
                    wait_time_sum += wait_h
                    emission_sum += (distance_m[trip] + d2s_m) / 1000.0 * selected_emission[vehicle_type]
                    continue
            soc_by_vehicle[vehicle] = next_soc
            if next_soc < -1e-12:
                invalid[solution] = 1
                break
            d2s_energy_sum += d2s_energy_one_way * 2.0 + transition_energy
            energy_delta_sum += 2.0 * d2s_energy_delta + transition_energy_delta
            wait_time_sum += wait_h
            timeout_row[trip] = timeout_h
            timeout_sum += timeout_h
            emission_sum += (
                distance_m[trip] + 2.0 * d2s_m + transition_distance_m[trip]
            ) / 1000.0 * selected_emission[vehicle_type]

        metrics[solution, 0] = trip_energy_sum
        metrics[solution, 1] = d2s_energy_sum
        metrics[solution, 2] = wait_time_sum
        metrics[solution, 3] = emission_sum
        metrics[solution, 4] = energy_delta_sum
        for station in range(num_stations):
            fast_total = 0
            slow_total = 0
            for channel in range(fast_count):
                fast_total += fast_used[station, channel]
            for channel in range(slow_count):
                slow_total += slow_used[station, channel]
            fast_used_count[solution, station] = fast_total
            slow_used_count[solution, station] = slow_total

        if store_timeouts == 1:
            for trip in range(trip_count):
                timeout_output[solution, trip] = timeout_row[trip]
        if invalid[solution] == 0:
            extras, saved_time, _, _ = _reschedule_one(
                timeout_row,
                vehicle_idx,
                vehicle_type_idx,
                distance_m,
                scenario_base_energy,
                selected_battery,
                selected_mass,
                0,
                energy_factor,
            )
            rescheduling[solution, 0] = extras[0]
            rescheduling[solution, 1] = extras[1]
            rescheduling[solution, 2] = extras[2]
            for vehicle_type in range(3):
                rescheduling[solution, vehicle_type] += replacement_count[vehicle_type]
            rescheduling[solution, 3] = saved_time
            rescheduling[solution, 4] = timeout_sum - saved_time
    return (
        metrics,
        fast_used_count,
        slow_used_count,
        invalid,
        rescheduling,
        tree_initializations,
    )


@njit(cache=True, parallel=True)
def _assemble_results(
    population: np.ndarray,
    num_stations: int,
    vehicle_counts: np.ndarray,
    fix_cost_yuan_year: np.ndarray,
    purchase_price_yuan: np.ndarray,
    battery_kwh: np.ndarray,
    battery_purchase_factor: float,
    vehicle_price_factor: float,
    fast_charger_cost_factor: float,
    core: np.ndarray,
    fast_counts: np.ndarray,
    slow_counts: np.ndarray,
    invalid: np.ndarray,
    rescheduling: np.ndarray,
    e_price: float,
    e_price_factor: float,
    baseline_energy_kwh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    population_size = population.shape[0]
    objectives = np.full((population_size, 2), -1e30, dtype=np.float64)
    diagnostics = np.zeros(
        (population_size, 2 * num_stations + 6), dtype=np.float64
    )
    metrics = np.zeros((population_size, len(METRIC_COLUMNS)), dtype=np.float64)
    for solution in prange(population_size):
        diagnostics[solution, 0] = invalid[solution]
        if invalid[solution] != 0:
            continue
        station_count = 0
        fast_total = 0
        slow_total = 0
        for station in range(num_stations):
            station_count += population[solution, station]
            fast_total += fast_counts[solution, station]
            slow_total += slow_counts[solution, station]
            diagnostics[solution, 1 + station] = -fast_counts[solution, station]
            diagnostics[solution, 1 + num_stations + station] = -slow_counts[
                solution, station
            ]

        vehicle_cost = 0.0
        for vehicle_type in range(3):
            model = int(population[solution, num_stations + vehicle_type])
            unit_cost = fix_cost_yuan_year[vehicle_type, model]
            if battery_purchase_factor != 1.0 or vehicle_price_factor != 1.0:
                base_price = purchase_price_yuan[vehicle_type, model]
                adjusted_price = (
                    base_price
                    + battery_kwh[vehicle_type, model]
                    * (battery_purchase_factor - 1.0) * 1150.0
                ) * vehicle_price_factor
                unit_cost += (adjusted_price - base_price) / 10.0
            vehicle_cost += (
                vehicle_counts[vehicle_type] + rescheduling[solution, vehicle_type]
            ) * unit_cost
        station_cost = station_count * 600_000.0
        station_emission = station_count * 80.0
        charger_cost = fast_total * 4_000.0 * fast_charger_cost_factor + slow_total * 2_000.0
        energy_cost = (
            core[solution, 0] + core[solution, 1]
        ) * 365.0 * e_price * e_price_factor
        objectives[solution, 0] = -(
            vehicle_cost + station_cost + charger_cost + energy_cost
        ) / 1_000_000.0
        energy_delta_g = (
            core[solution, 0] + core[solution, 1] - baseline_energy_kwh[solution]
        ) * 699.0196646565107
        objectives[solution, 1] = -(
            (core[solution, 3] + energy_delta_g) * 365.0 / 1000.0
            + station_emission
        ) / 1000.0

        diagnostics[solution, 1 + 2 * num_stations] = energy_cost / 1_000_000.0
        diagnostics[solution, 2 + 2 * num_stations] = rescheduling[solution, 4]
        diagnostics[solution, 3 + 2 * num_stations] = -rescheduling[solution, 0]
        diagnostics[solution, 4 + 2 * num_stations] = -rescheduling[solution, 1]
        diagnostics[solution, 5 + 2 * num_stations] = -rescheduling[solution, 2]

        metrics[solution, 0] = station_count
        metrics[solution, 1] = fast_total
        metrics[solution, 2] = slow_total
        metrics[solution, 3] = rescheduling[solution, 0]
        metrics[solution, 4] = rescheduling[solution, 1]
        metrics[solution, 5] = rescheduling[solution, 2]
        metrics[solution, 6] = core[solution, 0]
        metrics[solution, 7] = core[solution, 1]
        metrics[solution, 8] = core[solution, 2]
        metrics[solution, 9] = rescheduling[solution, 4]
        metrics[solution, 10] = core[solution, 3]
        metrics[solution, 11] = vehicle_cost
        metrics[solution, 12] = station_cost
        metrics[solution, 13] = charger_cost
        metrics[solution, 14] = energy_cost
    return objectives, diagnostics, metrics


def evaluate_population_fast(
    population: np.ndarray,
    problem,
    data: CNFastData | None = None,
    *,
    reserve_energy: float | None = None,
    min_interval: float = 60.0,
    degradation_factor: float = 1.0,
    degradation_share: float = 0.0,
    degraded_capacity_factor: float = 0.9,
    degradation_seed: int = 42,
    degradation_masks: np.ndarray | None = None,
    degradation_feasible: np.ndarray | None = None,
    battery_purchase_factor: float = 1.0,
    vehicle_price_factor: float = 1.0,
    power_factor: float = 1.0,
    slow_power_factor: float | None = None,
    fast_charger_cost_factor: float = 1.0,
    e_price_factor: float = 1.0,
    fast_chargers: int = 20,
    slow_chargers: int = 20,
    energy_factor: float = 1.0,
    hvac_power: float = -1.0,
    adjust_energy_emissions: bool = False,
    charging_profile: str = "linear",
    return_timeouts: bool = False,
) -> CNFastResult:
    """Evaluate an integer population with the array-based simulator."""
    if data is None:
        data = build_fast_data(problem)
    population = _normalise_population(population, data)
    if fast_chargers <= 0:
        raise ValueError("At least one fast charger is required by the current queue rule")
    if slow_chargers < 0:
        raise ValueError("slow_chargers cannot be negative")
    if energy_factor <= 0 or not np.isfinite(energy_factor):
        raise ValueError("energy_factor must be positive and finite")
    if hvac_power != -1.0 and (hvac_power < 0 or hvac_power > 25
                               or hvac_power != int(hvac_power)):
        raise ValueError("hvac_power must be an integer between 0 and 25 kW")
    if battery_purchase_factor <= 0 or not np.isfinite(battery_purchase_factor):
        raise ValueError("battery_purchase_factor must be positive and finite")
    if vehicle_price_factor <= 0 or not np.isfinite(vehicle_price_factor):
        raise ValueError("vehicle_price_factor must be positive and finite")
    if power_factor <= 0 or not np.isfinite(power_factor):
        raise ValueError("power_factor must be positive and finite")
    if slow_power_factor is None:
        slow_power_factor = power_factor
    if slow_power_factor <= 0 or not np.isfinite(slow_power_factor):
        raise ValueError("slow_power_factor must be positive and finite")
    if fast_charger_cost_factor <= 0 or not np.isfinite(fast_charger_cost_factor):
        raise ValueError("fast_charger_cost_factor must be positive and finite")
    if charging_profile not in ("linear", "part_e_both"):
        raise ValueError("Unknown charging profile")
    if degradation_masks is None or degradation_feasible is None:
        degradation_masks, degradation_feasible = build_degradation_masks(
            data, degradation_share, degradation_seed, degraded_capacity_factor,
            hvac_power=hvac_power, energy_factor=energy_factor,
        )
    set_num_threads(min(_MAX_NUMBA_THREADS, len(population)))
    chosen_idx, chosen_distance = choose_station_matrices(population, data)
    timeout_matrix = (
        np.empty((len(population), data.num_trips), dtype=np.float64)
        if return_timeouts
        else np.empty((1, 1), dtype=np.float64)
    )
    core, fast_counts, slow_counts, invalid, rescheduling, tree_initializations = _evaluate_core(
        population,
        chosen_idx,
        chosen_distance,
        data.num_stations,
        data.num_vehicles,
        data.vehicle_idx,
        data.vehicle_type_idx,
        data.vehicle_type_by_vehicle,
        data.trip_idx,
        data.s_us,
        data.e_us,
        data.first_s_us,
        data.next_pos,
        data.interval_min,
        data.distance_m,
        data.distance_km,
        data.transition_distance_m,
        data.transition_distance_km,
        data.base_energy_rate_kwh_km,
        data.base_energy_kwh,
        data.destination_idx,
        data.battery_kwh,
        data.mass_kg,
        data.mass_energy_kwh_km,
        data.per_emission_g_km,
        float(problem.reserve_energy if reserve_energy is None else reserve_energy),
        float(min_interval),
        float(degradation_factor),
        degradation_masks,
        degradation_feasible,
        float(degraded_capacity_factor),
        float(power_factor),
        float(slow_power_factor),
        int(fast_chargers),
        int(slow_chargers),
        float(energy_factor),
        float(hvac_power),
        int(charging_profile == "part_e_both"),
        int(return_timeouts),
        timeout_matrix,
    )
    baseline_energy_kwh = core[:, 0] + core[:, 1]
    objectives, diagnostics, metrics = _assemble_results(
        population,
        data.num_stations,
        data.vehicle_counts,
        data.fix_cost_yuan_year,
        data.purchase_price_yuan,
        data.battery_kwh,
        float(battery_purchase_factor),
        float(vehicle_price_factor),
        float(fast_charger_cost_factor),
        core,
        fast_counts,
        slow_counts,
        invalid,
        rescheduling,
        float(problem.e_price),
        float(e_price_factor),
        baseline_energy_kwh,
    )
    selected_models = population[:, data.num_stations:data.num_stations + 3]
    unit_vehicle = VEHICLE_CYCLE_T_PER_YEAR[np.arange(3)[None, :], selected_models]
    vehicle_emission = ((data.vehicle_counts[None, :] + metrics[:, 3:6])
                        * unit_vehicle).sum(axis=1)
    fuel_emission_g_day = (metrics[:, 6] + metrics[:, 7]) * GRID_G_CO2E_PER_KWH / CHARGING_EFFICIENCY
    fuel_emission = fuel_emission_g_day * 365.0 / 1e6
    station_emission = metrics[:, 0] * STATION_T_CO2E_PER_YEAR
    feasible = invalid == 0
    objectives[feasible, 1] = -(vehicle_emission + fuel_emission + station_emission)[feasible]
    metrics[feasible, 10] = fuel_emission_g_day[feasible]
    metrics[feasible, 15] = vehicle_emission[feasible]
    metrics[feasible, 16] = fuel_emission[feasible]
    metrics[feasible, 17] = station_emission[feasible]
    return CNFastResult(
        objectives=objectives,
        diagnostics=diagnostics,
        metrics=metrics,
        timeouts=timeout_matrix if return_timeouts else None,
        tree_initializations=tree_initializations,
    )


def reschedule_one_with_substitutions(
    timeout_row: np.ndarray,
    decision: np.ndarray,
    data: CNFastData,
    degradation_factor: float = 1.0,
) -> tuple[float, list[int], int, int, int]:
    """Expose rescheduling details for strict scalar-versus-array validation."""
    decision = _normalise_population(decision, data)[0]
    batteries = np.empty(3, dtype=np.float64)
    masses = np.empty(3, dtype=np.float64)
    for vehicle_type in range(3):
        model = int(decision[data.num_stations + vehicle_type])
        batteries[vehicle_type] = (
            data.battery_kwh[vehicle_type, model] * degradation_factor
        )
        masses[vehicle_type] = data.mass_kg[vehicle_type, model]
    extras, saved, substitutions, count = _reschedule_one(
        np.asarray(timeout_row, dtype=np.float64),
        data.vehicle_idx,
        data.vehicle_type_idx,
        data.distance_m,
        data.base_energy_kwh,
        batteries,
        masses,
        1,
    )
    return (
        float(saved),
        substitutions[:count].astype(int).tolist(),
        int(extras[0]),
        int(extras[1]),
        int(extras[2]),
    )



@dataclass(frozen=True)
class EvaluationSettings:
    reserve_energy: float | None = None
    min_interval: float = 60.0
    degradation_factor: float = 1.0
    degradation_share: float = 0.0
    degraded_capacity_factor: float = 0.9
    degradation_seed: int = 42
    battery_purchase_factor: float = 1.0
    vehicle_price_factor: float = 1.0
    power_factor: float = 1.0
    slow_power_factor: float | None = None
    fast_charger_cost_factor: float = 1.0
    e_price_factor: float = 1.0
    fast_chargers: int = 20
    slow_chargers: int = 20
    energy_factor: float = 1.0
    hvac_power: float = 4.0
    adjust_energy_emissions: bool = False
    charging_profile: str = "linear"


class FastAMOSAEvaluator:
    """Adapt the population evaluator to the scalar interface used by AMOSA."""

    def __init__(self, problem, settings: EvaluationSettings | None = None):
        self.problem = problem
        self.data: CNFastData = build_fast_data(problem)
        self.settings = settings or EvaluationSettings()
        self.degradation_masks, self.degradation_feasible = build_degradation_masks(
            self.data, self.settings.degradation_share,
            self.settings.degradation_seed, self.settings.degraded_capacity_factor,
            hvac_power=self.settings.hvac_power, energy_factor=self.settings.energy_factor,
        )

    def evaluate_population(self, population):
        population = np.asarray(population, dtype=np.int32)
        if population.ndim == 1:
            population = population.reshape(1, -1)
        result = evaluate_population_fast(
            population,
            self.problem,
            data=self.data,
            reserve_energy=self.settings.reserve_energy,
            min_interval=self.settings.min_interval,
            degradation_factor=self.settings.degradation_factor,
            degradation_share=self.settings.degradation_share,
            degraded_capacity_factor=self.settings.degraded_capacity_factor,
            degradation_masks=self.degradation_masks,
            degradation_feasible=self.degradation_feasible,
            battery_purchase_factor=self.settings.battery_purchase_factor,
            vehicle_price_factor=self.settings.vehicle_price_factor,
            power_factor=self.settings.power_factor,
            slow_power_factor=self.settings.slow_power_factor,
            fast_charger_cost_factor=self.settings.fast_charger_cost_factor,
            e_price_factor=self.settings.e_price_factor,
            fast_chargers=self.settings.fast_chargers,
            slow_chargers=self.settings.slow_chargers,
            energy_factor=self.settings.energy_factor,
            hvac_power=self.settings.hvac_power,
            adjust_energy_emissions=self.settings.adjust_energy_emissions,
            charging_profile=self.settings.charging_profile,
        )
        self.last_result = result
        # Match the simulator's random draws for tree identifiers.
        for count in result.tree_initializations:
            for _ in range(int(count)):
                random.getrandbits(14)
        return result.objectives, result.diagnostics

    def evaluate(self, vars_):
        objectives, diagnostics = self.evaluate_population(vars_)
        return objectives[0], diagnostics[0]
