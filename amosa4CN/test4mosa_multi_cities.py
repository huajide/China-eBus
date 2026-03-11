import time
import numpy as np
import pandas as pd
import geopandas as gpd
import math
import os
import multiprocessing as mp
from functools import partial

from simulation import SimVehicle, SimVehicleTrip
from station import Station
import networkx as nx
import data_utils
import sim_utils
from vehicle_type import VehicleTypes
from mosa import MOSA
from mosa import Problem
import pickle
from pyproj import CRS
from vehicle_scheduling import extra_timesave
from dateutil import parser

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
        self.avg_velocity = self.vs_parking_df['avg_velocity'].to_list()
        self.v_type = self.vs_parking_df['vehicle_type'].to_list()
        self.base_energy = self.vs_parking_df['base_energy'].to_list()

        self.simv_v_name = self.sim_v_info.index.to_list()
        self.simv_v_type = self.sim_v_info['vehicle_type'].to_list()

        self.large_num = self.simv_v_type.count('large')
        self.medium_num = self.simv_v_type.count('medium')
        self.small_num = self.simv_v_type.count('small')

        self.e_price = kwargs.get('e_price')
        self.default_chargers = 20

    def eval_vars(self, vars_, is_test=False, **kwargs):
        use_custom_chargers = ('fast_chargers' in kwargs and
                                'slow_chargers' in kwargs)
        # Use custom charger counts if provided; otherwise use defaults.
        fast_chargers = kwargs.get('fast_chargers', [self.default_chargers] * self.cs_num)
        slow_chargers = kwargs.get('slow_chargers', [self.default_chargers] * self.cs_num)

        # Check whether extra vehicle counts are provided.
        use_custom_extra = ('extra_large' in kwargs and
                            'extra_medium' in kwargs and
                            'extra_small' in kwargs)

        if use_custom_extra:
            extra_large = kwargs.get('extra_large', 0)
            extra_medium = kwargs.get('extra_medium', 0)
            extra_small = kwargs.get('extra_small', 0)

        power_factor = kwargs.get('power_factor', 1)
        e_price_factor = kwargs.get('e_price_factor', 1)
        degradation_factor = kwargs.get('degradation_factor', 1)  # not larger than 1
        min_interval = kwargs.get('min_interval', 60)

        cal_s_time = time.perf_counter()
        # Instantiate all sim_v and stations as well as simVTrip
        for i in range(len(self.simv_v_name)):
            if self.simv_v_type[i] == 'large':
                vehi_type = vars_[self.cs_num]
            elif self.simv_v_type[i] == 'medium':
                vehi_type = vars_[self.cs_num + 1]
            else:
                vehi_type = vars_[self.cs_num + 2]

            # Create a VehicleTypes object with battery degradation.
            degraded_vehi = VehicleTypes(vehi_type, v_type=self.simv_v_type[i])
            degraded_vehi.battery = degraded_vehi.battery * degradation_factor

            self.sim_v_dict[self.simv_v_name[i]].model = degraded_vehi
            # self.sim_v_dict[self.simv_v_name[i]].driving_range = degraded_vehi.driving_range
            self.sim_v_dict[self.simv_v_name[i]].battery = degraded_vehi.battery
            self.sim_v_dict[self.simv_v_name[i]].mass = degraded_vehi.mass

        # Create Station objects using the provided charger counts.
        station_dict = {row.node_id: Station(row.node_id, idx,
                                             fast_charger=fast_chargers[idx] if idx < len(fast_chargers) else self.default_chargers,
                                             slow_charger=slow_chargers[idx] if idx < len(slow_chargers) else self.default_chargers,
                                             power_factor=power_factor)
                        for idx, row in self.cs_gdf.iterrows()}

        timeout_list = []
        # Start simulation
        e_trip_sum, e_d2s_sum, wait_time_sum, emission_sum = 0, 0, 0, 0  # Initialize 4 variables for storing simulation values
        for i in range(len(self.v_name)):
            sim_v_trip = SimVehicleTrip(self.v_name[i], self.trip[i], self.s_time[i], self.e_time[i],
                                        self.destination[i],
                                        self.distance[i], self.base_energy[i], self.avg_velocity[i])
            e_trip, e_d2s, timeout, wait_time, trip_dist = sim_v_trip.simulation(
                self.sim_v_dict.get(self.v_name[i]),
                station_dict, vars_,
                sim_cs_method='get',
                all_d2s_dict=self.all_d2s_dict, cs_dict=self.cs_dict, min_interval=min_interval)
            e_trip_sum += e_trip
            e_d2s_sum += e_d2s
            wait_time_sum += wait_time
            timeout_list.append(timeout)

            emission_sum += trip_dist * self.sim_v_dict.get(self.v_name[i]).model.per_emission

        # Count actually used fast and slow chargers from the tree structure.
        charger_usage = {}  # {station_id: {'fast': set(), 'slow': set()}}

        # Traverse all vehicles and collect station usage.
        for sim_v in self.sim_v_dict.values():
            # Traverse each vehicle data_tree to collect charging events.
            for node in sim_v.data_tree.all_nodes():
                if node.data and 'charger_used' in node.data:
                    charger_info = node.data['charger_used']
                    # Extract the station ID from tags such as "t1s12345".
                    if 's' in node.tag and node.tag != sim_v.v_name:
                        parts = node.tag.split('s')
                        if len(parts) >= 2:
                            try:
                                station_id = int(parts[-1])

                                if station_id not in charger_usage:
                                    charger_usage[station_id] = {'fast': set(), 'slow': set()}

                                # Count fast and slow chargers by charger name.
                                if charger_info.startswith('f'):
                                    charger_usage[station_id]['fast'].add(charger_info)
                                elif charger_info.startswith('s'):
                                    charger_usage[station_id]['slow'].add(charger_info)
                            except ValueError:
                                continue

        if use_custom_chargers:
            actual_fast_charger_counts = fast_chargers
            actual_slow_charger_counts = slow_chargers
        else:
            # Count actual fast and slow chargers following station_dict order.
            actual_fast_charger_counts = []
            actual_slow_charger_counts = []

            for node_id, station in station_dict.items():
                fast_count = len(charger_usage.get(node_id, {}).get('fast', set()))
                slow_count = len(charger_usage.get(node_id, {}).get('slow', set()))
                actual_fast_charger_counts.append(fast_count)
                actual_slow_charger_counts.append(slow_count)

        if use_custom_extra:
            # Use the provided extra-vehicle counts and skip extra_timesave.
            # These values are used directly in vehicle_cost later.
            saved_time, subs_idx, extra_large_a, extra_medium_a, extra_small_a = extra_timesave(
                self.v_name, self.distance, self.base_energy, timeout_list, self.v_type,
                VehicleTypes(vars_[self.cs_num], v_type='large'),
                VehicleTypes(vars_[self.cs_num + 1], v_type='medium'),
                VehicleTypes(vars_[self.cs_num + 2], v_type='small'),
                extra_large, extra_medium, extra_small, degradation=degradation_factor
                )
        else:
            # Estimate extra vehicles with extra_timesave.
            saved_time, subs_idx, extra_large, extra_medium, extra_small = extra_timesave(
                self.v_name, self.distance, self.base_energy, timeout_list, self.v_type,
                VehicleTypes(vars_[self.cs_num], v_type='large'),
                VehicleTypes(vars_[self.cs_num + 1], v_type='medium'),
                VehicleTypes(vars_[self.cs_num + 2], v_type='small'),
                degradation=degradation_factor
                )
        timeout_sum = (sum(timeout_list) - saved_time)
        if is_test:
            return timeout_list, subs_idx

        # System costs
        # Vehicle cost
        vehicle_cost = ((self.large_num + extra_large) * VehicleTypes(vars_[self.cs_num], v_type='large').fix_cost +
                        (self.medium_num + extra_medium) * VehicleTypes(vars_[self.cs_num + 1],
                                                                        v_type='medium').fix_cost +
                        (self.small_num + extra_small) * VehicleTypes(vars_[self.cs_num + 2], v_type='small').fix_cost)

        # Station construction and maintenance costs per year
        station_cost = sum(vars_[0: self.cs_num]) * 600000 * 1
        station_emission = sum(vars_[0: self.cs_num]) * 80
        # 9w and 3w for fast and slow charger, respectively
        # Get chargers' num of selected stations and do calculation
        charger_cost = sum(actual_fast_charger_counts) * 4000 + sum(actual_slow_charger_counts) * 2000

        # 1.2 yuan/kwh
        # The life cycle cost of energy consumption in this trip， including both operation and go-charging distances
        trip_cost = (e_trip_sum + e_d2s_sum) * 365 * self.e_price * e_price_factor
        print(f'{e_trip_sum:.2f} {e_d2s_sum:.2f}')

        # print(f'Station Count: {sum(vars_[:self.cs_num])}; Extra Vehicles: {extra_large} {extra_medium} {extra_small}')

        '''one-objective'''
        # emission_cost = (emission_sum*365/1000+station_emission)*1.05  # kg * 1.05 yuan/kg social cost of emission
        # f0 = -(vehicle_cost + station_cost + charger_cost + trip_cost + emission_cost) / 1000000  # 1M yuan/year
        '''multi-objective'''
        f1 = -(vehicle_cost + station_cost + charger_cost + trip_cost) / 1000000  # 1M yuan/year
        f2 = -(emission_sum * 365 / 1000 + station_emission) / 1000  # T/year

        cv1 = sim_utils.set_cv1(self.cs_num, vars_)  # Constraint term defined by the model formulation.

        cv_and_params = ([cv1] + [-x for x in actual_fast_charger_counts] + [-x for x in actual_slow_charger_counts] +
                         [trip_cost/1000000, timeout_sum] + [-extra_large, -extra_medium, -extra_small])
        cal_e_time = time.perf_counter()
        # print(f'Single calculation time: {(cal_e_time - cal_s_time):.2f}s')

        # print(f"f1: {-f1:.1f} f2: {-f2:.1f}")

        return np.array([f1, f2]), np.hstack(cv_and_params)

def process_single_city(city_i, cities_df, is_simplified=False, is_referred=False,
                        is_dict_loaded=True, is_tested=True, SIM_N=50, what_if=False, min_delta_hv=0.01):
    """Process a single city using the same logic as the main workflow."""
    city_name = cities_df['city'][city_i]
    e_price = cities_df['eprice'][city_i]

    try:
        print(f"Processing city: {city_name}")

        if is_simplified:
            vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid_simplified/{city_name}.csv')
        else:
            vs_parking_df = pd.read_csv(rf'../data/input/vs_parking_nodeid/{city_name}.csv')

        vs_parking_df['s_time'] = vs_parking_df['s_time'].apply(parser.parse)
        vs_parking_df['e_time'] = vs_parking_df['e_time'].apply(parser.parse)
        vs_parking_df.sort_values(['e_time'], inplace=True, ignore_index=True)

        cs_gdf = gpd.read_file(rf'../data/input/cs_gdf/{city_name}.shp', crs=CRS.from_epsg(4547))
        cs_gdf['lon'] = cs_gdf['geometry'].x
        cs_gdf['lat'] = cs_gdf['geometry'].y

        sim_v_info = pd.DataFrame.from_dict(vs_parking_df.groupby('v_name').
                                            apply(lambda x: data_utils.derive_simV_info(x)).to_dict(), orient='index')
        sim_v_dict = {}
        for idx, row in sim_v_info.iterrows():
            sim_v_dict[idx] = SimVehicle(idx, row.trip, row.s_time, row.e_time, row.destination, row.distance,
                                         row.base_energy, row.avg_velocity)

        if is_dict_loaded:
            with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'rb') as f:
                all_d2s_dict = pickle.load(f)
        else:
            nodes_sim = gpd.read_file(
                rf'../data/road/{city_name}/nodes_sim.shp', crs=CRS.from_epsg(4547))
            edges_sim = gpd.read_file(
                rf'../data/road/{city_name}/edges_sim.shp', crs=CRS.from_epsg(4547))
            G = nx.from_pandas_edgelist(df=edges_sim, source='u', target='v', edge_attr=['edge_id', 'length'],
                                        create_using=nx.Graph())
            all_d2s_dict = data_utils.get_d2s_realdict(vs_parking_df, cs_gdf, nodes_sim, G,
                                                       near_n=100, sim_n=SIM_N, distance_limit=100000,
                                                       is_projected=True)
            with open(rf"../data/input/all_d2s_dict/{city_name}.pkl", 'wb') as f:
                pickle.dump(all_d2s_dict, f)

        num_vars = data_utils.set_var_num(cs_gdf)
        print("num_vars: ", num_vars)

        cs_dict = {v: k for k, v in zip(cs_gdf.index, cs_gdf['node_id'])}
        print('read files successfully!')

        problem = Location(num_vars=num_vars, sim_v_info=sim_v_info, sim_v_dict=sim_v_dict, cs_gdf=cs_gdf,
                           vs_parking_df=vs_parking_df,
                           all_d2s_dict=all_d2s_dict, cs_dict=cs_dict, e_price=e_price)

        annealing_iters = 100  # 100
        algorithm = MOSA(problem, annealing_iters=annealing_iters)
        # algorithm.end_temperature = 1  # 900
        algorithm.annealing_strength = 0.6
        algorithm.cooling_alpha = 0.9953
        algorithm.min_delta_hv = min_delta_hv
        algorithm.multiprocess = False
        algorithm.what_if = what_if
        algorithm.is_tested = is_tested

        # # estimate how long would be processed
        print(f"Estimated: {math.log(algorithm.end_temperature / algorithm.initial_temperature) /
                            math.log(algorithm.cooling_alpha) * annealing_iters * 2 / 3600:.1f} h")  # 10s is estimated circle time

        algorithm.early_termination = {'max_iters': 500000, 'max_duration': 240,
                                       'max_no_eliminated': 20000}
        if is_referred:
            refer_vars = pd.read_csv(
                rf"../data/output/mosa/250630/{city_name}/archive_vars.csv").values[:, 1:]
        else:
            refer_vars = sim_utils.set_refer_vars(len(cs_gdf), refer_num=20)

        algorithm.load_refer_solutions(refer_vars)
        save_path = rf"../data/output/mosa/251026/{city_name}"
        # Skip the city if save_path already exists and is not empty.
        if os.path.exists(save_path) and os.listdir(save_path):
            print(f"Results for city {city_name} already exist and are not empty; skipping")
            return f"City {city_name} already has results; skipped"

        # Create the output directory if needed.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        algorithm.run(inf='infeasible', is_cv=True, path=save_path, store='store')  # ,multi_tasks=4

        if is_tested:
            # convergence = algorithm.output_fitness()

            A = algorithm.archive_history
            # Stack all objective values into one array.
            all_objs = np.vstack([objs for objs in A if objs.size != 0])
            # Get the minimum and maximum value of each objective.
            min_vals = np.min(all_objs, axis=0)
            max_vals = np.max(all_objs, axis=0)
            print("Minimum value of each objective:", min_vals)
            print("Maximum value of each objective:", max_vals)
            standard_convergence = algorithm.output_fitness(min_vals, max_vals)

        print(f"Finished processing city: {city_name}")
        return f"City {city_name} processed successfully"

    except Exception as e:
        print(f"Error processing city {city_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"City {city_name} failed: {str(e)}"

def main(cities_file='../data/224cities.csv', processes=1, city_indices=None,
         is_simplified=False, is_referred=False, is_dict_loaded=True,
         is_tested=True, SIM_N=50, what_if=False, min_delta_hv=0.01):
    """Main entry point for multi-city optimization with configurable parameters."""

    # Read the city list.
    cities = pd.read_csv(cities_file)

    if city_indices is None:
        city_indices = list(range(len(cities)))

    print(f"Preparing to process {len(city_indices)} cities")
    print(f"Number of processes: {processes}")
    print(f"Parameters: simplified={is_simplified}, referred={is_referred}, dict_loaded={is_dict_loaded}")
    print(f"          tested={is_tested}, SIM_N={SIM_N}, what_if={what_if}, min_delta_hv={min_delta_hv}")

    if processes == 1:
        # Single-process execution.
        results = []
        for city_i in city_indices:
            result = process_single_city(
                city_i, cities,
                is_simplified=is_simplified,
                is_referred=is_referred,
                is_dict_loaded=is_dict_loaded,
                is_tested=is_tested,
                SIM_N=SIM_N,
                what_if=what_if,
                min_delta_hv=min_delta_hv
            )
            results.append(result)
    else:
        # Multiprocessing execution.
        # Create a partial function with shared parameters.
        process_func = partial(
            process_single_city,
            cities_df=cities,
            is_simplified=is_simplified,
            is_referred=is_referred,
            is_dict_loaded=is_dict_loaded,
            is_tested=is_tested,
            SIM_N=SIM_N,
            what_if=what_if,
            min_delta_hv=min_delta_hv
        )

        # Run cities in a multiprocessing pool.
        num_processes = min(processes, len(city_indices))
        print(f"Processing {len(city_indices)} cities with {num_processes} workers")

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_func, city_indices)

    # Summarize the processing results.
    success_count = sum(1 for result in results if "processed successfully" in result)
    fail_count = len(results) - success_count

    print(f"\nProcessing summary:")
    print(f"Success: {success_count} cities")
    print(f"Failed: {fail_count} cities")

    # Print failed cities.
    if fail_count > 0:
        print("Failed cities:")
        for result in results:
            if "failed:" in result:
                print(result)

    return results

# Keep the original single-city workflow for direct execution.
if __name__ == '__main__':
    main(
        cities_file='../data/224cities.csv',
        processes=10,
        is_simplified=True,
        is_referred=False,
        is_tested=False
    )
