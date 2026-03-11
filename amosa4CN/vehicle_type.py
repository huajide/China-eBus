models = [['SLK6101UBEVN1','NJL6126EVD1','ZK6126BEVG11E','TEG6105BEV30','JS6108GHBEV38'],
          ['XML6705JEVY0C1','WD6707BEVLG03','KLQ6816GAEVN2','ZK6707BEVG6E','JS6709GHBEV'],
          ['HFF6609G7EV21','KLQ6650GEVN6','ZK6606BEVG6C']]

mass = [[11500,12000,11650,11300,10900],[6800,5800,7300,5120,6000],[5300,5350,3960]]
battery_capacity = [[314,350,359,350,255],[169,141,134,141,141],[155,128,116]]  # kwh
price = [[1088000,1008000,992000,956000,1058000],[799500,715000,586000,699000,789900],[468000,396000,619000]]  # yuan
per_emission = [[197,230,241,245,172],[162,164,152,170,155],[178,153,199]]  # GHGs emission per km (g/km)

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
    """
    vt: 0-large; 1-medium; 2-small
    """
    return len(models[type_map[v_type]])

class VehicleTypes:

    def __init__(self, model: int, v_type: str):
        # type_map = {'large': 0, 'medium': 1, 'small': 2}
        if v_type not in type_map:
            raise ValueError(f"Invalid vehicle type: {v_type}. Expected 'large', 'medium', or 'small'.")
        k = type_map[v_type]
        self.model = models[k][model]
        # self.driving_range = driving_range[k][model]
        self.battery = battery_capacity[k][model]
        self.mass = mass[k][model]
        # self.e2s_ratio = e2s_ratio[k][model]
        self.price = price[k][model]
        self.per_emission = per_emission[k][model]

        self.fix_cost = fix_cost[k][model]
