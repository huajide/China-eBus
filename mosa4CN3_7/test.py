

def energy_consumption(g=0, SOC=100, R_c=2, HVAC=1.25, P_l=0, D_Agg=2, S_d=2, V_a=15, C_D=0.6):
    E_c = (-0.885 + 0.380 * g + 0.012 * SOC + 0.260 * R_c +0.036 * HVAC + 0.005 * P_l + 0.065*D_Agg +
           0.128 * S_d + 0.007*V_a + 0.173*C_D)
    return E_c

if __name__ == '__main__':
    print(f"{energy_consumption(g=4, SOC=40, R_c=2, HVAC=20, P_l=10, D_Agg=2, S_d=2, V_a=20, C_D=0.6):.2f} kWh/km")

    print(f'{314/520:.2f} kWh/km')