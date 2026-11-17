import numpy as np


models = [['Large 1','Large 2','Large 3','Large 4','Large 5'],
          ['Medium 1','Medium 2','Medium 3','Medium 4','Medium 5'],
          ['Small 1','Small 2','Small 3']]
mass = [[11500,12000,11650,11300,10900],[6800,5800,7300,5120,6000],[5300,5350,3960]]
battery_capacity = [[314,350,359,350,255],[169,141,134,141,141],[155,128,116]]  # kwh
price = [[1088000,1008000,992000,956000,1058000],[799500,715000,586000,699000,789900],[468000,396000,619000]]  # yuan
per_emission = [[800,804,804,802,795],[780,776,779,775,777],[776,774,771]]  # GHGs emission per km (g/km)

baseline_mass = 14935  # kg
people_mass = 75  # kg
beta_const, beta_g, beta_SOC, beta_R_c, beta_HVAC, beta_P_l, beta_D_Agg, beta_S_d, beta_V_a, beta_C_D = (
    -0.885, 0.380, 0.012, 0.260, 0.036, 0.005, 0.065, 0.128, 0.007, 0.173)


de_factor = 1
price = [[p + b * (de_factor-1) * 1150 for p, b in zip(p_list, b_list)] for p_list, b_list in zip(price, battery_capacity)]
battery_capacity = [[capacity * de_factor for capacity in sublist] for sublist in battery_capacity]

price_factor = 1
price = [[P * price_factor for P in sublist] for sublist in price]



maintain = 10000  # per year
discount = 0.05
lifespan = 10 # (year)

fix_cost = []
for i in range(len(price)):
    fix_cost.append([])
    for j in range(len(price[i])):
        fc = (price[i][j] + maintain * (1 - (1 + discount) ** (-lifespan)) / discount) / lifespan
        fix_cost[i].append(fc)

type_map = {'large': 0, 'medium': 1, 'small': 2}


def vehi_type_num(v_type: str):
    """Return the number of candidate models for the requested size class."""
    return len(models[type_map[v_type]])


class VehicleTypes:

    def __init__(self, model: int, v_type: str):
        if v_type not in type_map:
            raise ValueError(f"Invalid vehicle type: {v_type}. Expected 'large', 'medium', or 'small'.")
        k = type_map[v_type]
        self.model = models[k][model]
        self.battery = battery_capacity[k][model]
        self.mass = mass[k][model]
        self.price = price[k][model]
        self.per_emission = per_emission[k][model]

        self.fix_cost = fix_cost[k][model]  # yuan per year

EMISSION_METHOD = "vehicle_fuel_cycle"
GRID_G_CO2E_PER_KWH = 673.8731220322626
CHARGING_EFFICIENCY = 0.85
STATION_T_CO2E_PER_YEAR = 0.08

# Tonnes CO2e per vehicle per year, in the same order as models.
VEHICLE_CYCLE_T_PER_YEAR = np.array([
    [8.729419558686137, 9.330285384027551, 9.292668539649078,
     9.053478414554545, 7.831444021870125],
    [5.247073716614986, 4.538076752126979, 5.052844934687491,
     4.269178553210344, 4.617164457690694],
    [4.497136706552401, 4.21454871186924, 3.5305069710573727, 0.0, 0.0],
], dtype=float)

