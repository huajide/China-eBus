import pandas as pd
from vehicle_type import (VehicleTypes, baseline_mass, beta_const, beta_g, beta_SOC, beta_R_c, beta_HVAC, beta_P_l,
                          beta_D_Agg, beta_S_d, beta_V_a, beta_C_D, people_mass)

def extra_timesave(v_name_list, dist_list, base_energy_list, timeout_list, v_type_list,
                   vehicle_large: VehicleTypes, vehicle_medium: VehicleTypes, vehicle_small: VehicleTypes,
                   max_extra_large=99999, max_extra_medium=99999, max_extra_small=99999, degradation=1.0):
    """
    Vehicle scheduling based on parking duration and timeout conditions

    Args:
    vs_parking_df (pd.DataFrame): DataFrame containing vehicle parking information
    timeout_list (List[datetime]): List of datetime objects representing timeout conditions
    extra_bus (int): Number of extra buses to be scheduled
    extra_minibus (int): Number of extra minibuses to be scheduled
    distance:m ; range: km
    Returns:
    saved_time
    """
    all_not_none = (max_extra_large is not None) and (max_extra_medium is not None) and (max_extra_small is not None)

    '''Select trips in descending timeout order. Each selected trip allows later trips within the range limit to be replaced by the new vehicle.'''
    survived_indices = [i for i, t in enumerate(timeout_list) if t > 0]
    v_name_list = [v_name_list[i] for i in survived_indices]
    dist_list = [dist_list[i] for i in survived_indices]
    base_energy_list = [base_energy_list[i] for i in survived_indices]
    timeout_list = [timeout_list[i] for i in survived_indices]
    v_type_list = [v_type_list[i] for i in survived_indices]

    battery_large, battery_medium, battery_small = vehicle_large.battery, vehicle_medium.battery, vehicle_small.battery
    battery_large *= degradation
    battery_medium *= degradation
    battery_small *= degradation
    mass_large, mass_medium, mass_small = vehicle_large.mass, vehicle_medium.mass, vehicle_small.mass

    sorted_indices = sorted(range(len(timeout_list)), key=lambda i: timeout_list[i], reverse=True)
    saved_time_all = 0
    sub_idx = []
    extra_large, extra_medium, extra_small = 0,0,0
    for rank in range(len(sorted_indices)):
        idx = sorted_indices.index(rank)
        is_replaced = True
        if v_name_list[idx]==-1:
            continue
        if timeout_list[idx]:
            if v_type_list[idx] == 'large' and max_extra_large:
                extra_large += 1
                max_extra_large -= 1
                battery_state = battery_large
                mass_state = mass_large
            elif v_type_list[idx] == 'medium' and max_extra_medium:
                extra_medium += 1
                max_extra_medium -= 1
                battery_state = battery_medium
                mass_state = mass_medium
            elif v_type_list[idx] == 'small' and max_extra_small:
                extra_small += 1
                max_extra_small -= 1
                battery_state = battery_small
                mass_state = mass_small
            else:
                is_replaced = False
            if not is_replaced:
                continue
            # Find all later indices with the same vehicle name.
            samevehi_indices = [i for i in range(idx, len(v_name_list)) if v_name_list[i] == v_name_list[idx]]

            delta_mass_energy = (mass_state - baseline_mass) / people_mass * beta_P_l

            for idx2 in samevehi_indices:
                energy_consumed = max(0, base_energy_list[idx2] + (beta_SOC*100 + delta_mass_energy)*dist_list[idx2]/1000)
                if energy_consumed<=battery_state:  # unit: m
                    battery_state-=energy_consumed
                    v_name_list[idx2] = -1
                    saved_time_all += timeout_list[idx2]
                    timeout_list[idx2]=0
                    sub_idx.append(survived_indices[idx2])
                else:
                    break
        if all(t <= 0.05 for t in timeout_list):  # all timeout < 3mins
            break

    return saved_time_all, sub_idx, extra_large, extra_medium, extra_small

if __name__ == '__main__':
    test = pd.read_csv(r"test4extra.csv")
    test_timeout_list = test['timeout'].to_list()
    v_name_list = test['v_name'].to_list()
    dist_list = test['distance'].to_list()

    saved_time, subs_idx, extra_bus_num = extra_timesave(v_name_list,dist_list,test_timeout_list,500)
    print(sum(test_timeout_list)-saved_time)
    print(extra_bus_num)
