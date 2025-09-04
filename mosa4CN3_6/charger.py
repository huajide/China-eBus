"""
@Project ：DissertationPY 
@File    ：charger.py
@IDE     ：PyCharm 
@Author  ：Xander PENG
@Date    ：12/8/2022 01:49 
"""
from datetime import datetime, timedelta
from typing import Dict, List
from simulation import SimVehicle


class Charger:

    def __init__(self, s_name: int, s_idx, s_charger_count: int,
                 charger_idx: str, power: float):
        # Static attributes
        self.s_name = s_name  # The osmid of station
        self.s_index = s_idx  # The index of station in the cs_gdf
        self.s_charger_count = s_charger_count  # the amount of chargers in this station
        self.charger_idx = charger_idx  # The index of this charger in this station(charger name)
        self.power = power  # charging power; unit: kwh

        # Dynamic attribute
        self.is_avl = True  # The status of this charger
        self.queue: List[datetime] = []

        # Charging start time and end time
        self.start_time = None
        self.end_time = None
        self.v_name = None

    def is_available(self, request_time: datetime):
        """
        Check if this charger is available

        :param request_time: The datetime that sim_v deliver charging request(equals to the sim_v arrival time)
        :return: True or False
        """
        if len(self.queue) == 0:  # No waiting vs
            self.is_avl = True

        elif (len(self.queue) >= 1) & (self.end_time <= request_time):

            self.is_avl = True

        else:
            self.is_avl = False

        return self.is_avl

    def charging(self, sim_v: SimVehicle, arr_time: datetime, arr_soc: float,
                 ch_time: float, time_gap: float, max_e: float):
        """
        Simulate charging event happening for NO-Queuing scenario

        :param sim_v: The vehicle that needs charging; instance of SimVehicle
        :param arr_time: The time when this sim_v arrives at the station
        :param arr_soc: The soc of sim_v when it arrives at the station; within [0, 1]
        :param min_e_demand: The minimum electricity demand for the sake of next trip; (n% of battery_capacity) [0 ~ 1]
        :param ch_time: The estimated charging time. unit: hours
        :param time_gap: The time gap between the next trip start time and arrival time; unit: hours
        :param max_e: The maximum electricity ratio this sim_v would charge
        :return: A list of [the soc after charging, the charging timeout, wait_time=0]
        """
        # Clean the queue list and append current charging event
        self.queue.clear()
        self.queue.append(arr_time)

        # Update the start charging time
        self.start_time = arr_time

        # Update the charging finished time and cal charging timeout
        if time_gap >= ch_time:
            # When the min_charging demand can be fulfilled within time_gap
            if (time_gap * self.power / sim_v.battery) >= max_e:
                # When the interval is large enough to fulfill this vehicle
                duration2full = (1 - arr_soc) * sim_v.battery / self.power  # unit: hour
                self.end_time = arr_time + timedelta(hours=duration2full)
                ch_timeout = 0
            else:  # The time_gap is long enough for fulfilling the min_e but not the max_e
                self.end_time = arr_time + timedelta(hours=time_gap)  # Charging until the time_gap
                ch_timeout = 0

        else:  # When the min_charging demand cannot be fulfilled within time_gap
            self.end_time = arr_time + timedelta(hours=ch_time)
            ch_timeout = ch_time - time_gap  # unit: hour

        # Cal the soc after charging
        soc_ch = (self.end_time - arr_time).seconds / 3600 * self.power / sim_v.battery + arr_soc

        result = [soc_ch, ch_timeout, 0]  # Return the soc after charging and the timeout of charging, Plus 0 wait time

        return result

    def charging4queue(self, arr_time: datetime, time_gap: float, sim_v: SimVehicle, arr_soc: float,
                       min_ch_time: float, max_e: float):

        # Add this charging event into the queue
        self.queue.append(arr_time)

        # Update the start_time
        self.start_time: datetime = self.end_time  # The charging start time is the end of previous one's

        # Update the charging end_time
        real_time_gap: timedelta = arr_time + timedelta(hours=time_gap) - self.start_time
        # Since it is possible that charging start time is later than next trip start time, there should be discussions
        if real_time_gap.days == 0:  # When charging start time earlier than next trip time
            # if (real_time_gap.seconds / 3600 * self.power / sim_v.battery) >= min_e_demand:
            if (real_time_gap.seconds/3600) >= min_ch_time:
                # When the min_charging demand can be fulfilled within real_time_gap
                if (real_time_gap.seconds/3600 * self.power / sim_v.battery) >= max_e:
                    # When the interval is large enough to fulfill this vehicle
                    duration2full = (1 - arr_soc) * sim_v.battery / self.power  # unit: hour
                    self.end_time = self.start_time + timedelta(hours=duration2full)
                    ch_timeout = 0
                    ch_wait_time = (self.start_time - arr_time).seconds / 3600  # unit: hour
                else:  # The time_gap is long enough for fulfilling the min_e but not the max_e
                    self.end_time = self.start_time + real_time_gap
                    ch_timeout = 0
                    ch_wait_time: float = (self.start_time - arr_time).seconds / 3600  # unit: hour
            else:  # When the min_charging demand cannot be fulfilled within real_time_gap
                self.end_time: datetime = self.start_time + timedelta(hours=min_ch_time)
                ch_timeout = min_ch_time - real_time_gap.seconds / 3600  # unit: hour
                ch_wait_time: float = (self.start_time - arr_time).seconds / 3600  # unit: hour

        else:  # When charging start time is later than next trip start time
            self.end_time = self.start_time + timedelta(hours=min_ch_time)
            ch_timeout = (self.end_time - (arr_time + timedelta(hours=time_gap))).seconds / 3600  # unit: hour
            ch_wait_time: float = (self.start_time - arr_time).seconds / 3600  # unit: hour

        # Cal the soc after charging
        soc_ch = (self.end_time - self.start_time).seconds / 3600 * self.power / sim_v.battery + arr_soc

        # Return the soc after charging and the timeout of charging as well as waiting time
        result = [soc_ch, ch_timeout, ch_wait_time]

        return result


class FastCharger(Charger):

    def __init__(self, s_name: int, s_idx, s_charger_count: int,
                 charger_idx: str, power: float):
        # Inherent from Charger
        super(FastCharger, self).__init__(s_name, s_idx, s_charger_count,
                                          charger_idx, power)
        self.type = 'fast'


class SlowCharger(Charger):
    def __init__(self, s_name: int, s_idx, s_charger_count: int,
                 charger_idx: str, power: float):
        # Inherent from Charger
        super(SlowCharger, self).__init__(s_name, s_idx, s_charger_count,
                                          charger_idx, power)
        self.type = 'slow'
