from datetime import datetime

import pandas as pd
import sim_utils
from simulation import InfeasibleSolutionError, SimVehicle, SimVehicleTrip
from charger import Charger, FastCharger, SlowCharger
from typing import List, Dict
from datetime import timedelta
from vehicle_type import (VehicleTypes, baseline_mass, beta_const, beta_g, beta_SOC, beta_R_c, beta_HVAC, beta_P_l,
                          beta_D_Agg, beta_S_d, beta_V_a, beta_C_D, people_mass)


class Station:

    def __init__(self, s_name: int, s_idx: int,
                 fast_charger: int = 2, slow_charger: int = 2, power_factor: float = 1):
        # Static Attributes
        self.s_name = s_name  # The osmid of station
        self.s_index = s_idx  # The index of station in the cs_gdf

        self.fast_charger_count = int(fast_charger)
        self.slow_charger_count = int(slow_charger)
        self.charger_count = int(fast_charger + slow_charger)  # the amount of chargers in this station
        # The power of fast charging and slow charging, respectively
        self.power_factor = power_factor
        self.fast_power = 100 * self.power_factor
        self.slow_power = 50 * self.power_factor
        self.max_used_fast_charger = 0
        self.max_used_slow_charger = 0

        # Instantiate chargers; all the chargers instance in this station
        self.chargers = {}
        for i in range(self.charger_count):
            if i < self.fast_charger_count:
                self.chargers.update({'f' + str(i + 1): FastCharger(self.s_name, self.s_index, self.charger_count,
                                                                    'f' + str(i + 1), self.fast_power)})
            else:
                self.chargers.update({'s' + str(i + 1): SlowCharger(self.s_name, self.s_index, self.charger_count,
                                                                    's' + str(i + 1), self.slow_power)})

        # Dynamic attributes
        self.avl_chargers = list(self.chargers.keys())  # the available chargers

    def check_avl(self, arr_time: datetime):

        """
        Check all chargers' status(i.e. whether available) when a new car drive in the station and ask for charging

        :param arr_time: the time when vehicle push charging request
        """

        # Get all chargers' status
        status_list: List[bool] = list(map(lambda x: x.is_available(arr_time), self.chargers.values()))
        status_dict = dict(zip(self.chargers.values(), status_list))
        # update the avl_chargers
        self.avl_chargers = list(filter(lambda x: status_dict.get(x) is True, status_dict))
        self.avl_chargers = [ch.charger_idx for ch in self.avl_chargers]

        available_fast = sum(1 for ch in self.avl_chargers if 'f' in ch)
        available_slow = sum(1 for ch in self.avl_chargers if 's' in ch)
        used_fast = self.fast_charger_count - available_fast
        used_slow = self.slow_charger_count - available_slow
        
        if used_fast > self.max_used_fast_charger:
            self.max_used_fast_charger = used_fast
        if used_slow > self.max_used_slow_charger:
            self.max_used_slow_charger = used_slow


    def charging_request(self,
                         sim_v: SimVehicle,  # the simulation vehicle
                         sim_v_trip: SimVehicleTrip,  # the simulation vehicle trip
                         arr_time: datetime,  # the time when sim_v arrives at this station
                         arr_soc: float,  # The soc when sim_v arrives at this station
                         e_trip_duration: float, # duration from terminal to the charging station (hour)
                         post_charge_energy: float = 0.0,
                         ):
        # Once new sim_v drives in, check all the chargers status and update the avl_chargers
        self.check_avl(arr_time)

        trip_idx = sim_v_trip.trip_idx  # Get this vehicle's current trip index

        last_trip = 0
        try:
            delta_mass_energy = (sim_v.mass - baseline_mass) / people_mass * beta_P_l
            next_distance = sim_v.distances[trip_idx] / 1000
            remaining_capacity = sim_v.battery - beta_SOC * 100 * next_distance
            if remaining_capacity <= 0:
                raise InfeasibleSolutionError('Next trip exceeds battery capacity')
            required_origin_soc = max(0, (sim_v.base_energy[trip_idx] +
                                          delta_mass_energy * next_distance) / remaining_capacity)
            required_station_soc = required_origin_soc + post_charge_energy / sim_v.battery
            if required_station_soc > 1 + 1e-12:
                raise InfeasibleSolutionError('Next trip is unreachable after charging')
        except IndexError:  # When this is the last trip, trip_idx will cause IndexError
            last_trip = 1  # This is the last trip

        if last_trip:
            e_ch_min = 1 - arr_soc
        else:
            e_ch_min = min(1 - arr_soc, max(0.05, required_station_soc - arr_soc + 0.05))

        e_ch_max = 1 - arr_soc  # The maximum charging e ratio

        # Evaluate hou much time needed using fast and slow charging
        f_time = (sim_v.battery * e_ch_min) / self.fast_power  # unit: hour
        s_time = (sim_v.battery * e_ch_min) / self.slow_power  # unit: hour

        # Check the time gap between arrival time and the beginning of next trip (minus e_trip_duration); unit: hour
        try:
            time_gap = (sim_v.trip_s_time[trip_idx] - arr_time).total_seconds() / 3600 - e_trip_duration
        except IndexError:
            time_gap = (sim_v.trip_s_time[0] - arr_time + timedelta(days=1)).total_seconds() / 3600 - e_trip_duration  # hour


        # Check if there are available chargers
        if len(self.avl_chargers) != 0:
            # If both fast and slow charger are free
            if any(map(lambda x: 'f' in x, self.avl_chargers)) &\
                    any(map(lambda x: 's' in x, self.avl_chargers)):

                if time_gap > s_time:  # choose slow charger
                    # Choose one slow charger for charging
                    sel_charger_name = list(filter(lambda x: x if 's' in x else None, self.avl_chargers))[0]

                else:  # not enough gap time period, choose fast charging
                    sel_charger_name = list(filter(lambda x: x if 'f' in x else None, self.avl_chargers))[0]

            else:  # Only one kind of type charger available
                sel_charger_name = self.avl_chargers[0]

            is_waiting = False

        else:  # No available chargers; avl_chargers = []
            # Get all chargers waiting queue length; {charger_name: queuing_length}
            queuing_length: Dict[str, int] = {ch.charger_idx: len(ch.queue) for ch in self.chargers.values()}
            candidate_chargers = sim_utils.find_keys(queuing_length, value='first_key')

            if len(candidate_chargers) == 1:
                sel_charger_name = candidate_chargers[0]

            else:
                if any(map(lambda x: 'f' in x, candidate_chargers)) & any(map(lambda x: 's' in x, candidate_chargers)):

                    if time_gap > s_time:  # choose slow charger
                        # Choose one slow charger for charging
                        sel_charger_name = list(filter(lambda x: x if 's' in x else None, candidate_chargers))[0]

                    else:  # not enough gap time period, choose fast charging
                        sel_charger_name = list(filter(lambda x: x if 'f' in x else None, candidate_chargers))[0]

                else:  # Only one kind of type charger available
                    sel_charger_name = candidate_chargers[0]

            is_waiting = True

        # After choosing suitable charger, execute the charging behavior

        sel_charger = self.chargers.get(sel_charger_name)
        ch_time = f_time if 'f' in sel_charger.charger_idx else s_time

        if not is_waiting:
            result = sel_charger.charging(sim_v, arr_time, arr_soc, ch_time, time_gap, max_e=e_ch_max)
        else:
            result = sel_charger.charging4queue(arr_time, time_gap, sim_v, arr_soc, ch_time, max_e=e_ch_max)

        return list(result) + [sel_charger_name]
