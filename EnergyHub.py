""" Energy Innovation and Management - Spring Semester 2024
 ========================================================================
 Author of original code: M.Hohmann                                         |
 Source: https://hues.empa.ch/index.php/Model:Generic_Energy_Hub_YALMIP     |
 Edited and extended by: G. Mavromatidis (gmavroma@ethz.ch) 28.01.2020      |
 Converted from Matlab to Python by: S. Powell (spowell@ethz.ch) 05.01.2024 |
 ========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxpy as cp

# Energy demands
# ===============
demand_data = pd.read_excel('demands.xlsx',
                            names=['Heating demand [kWh]', 'Electricity demand [kWh]'],
                            header=None)
elec_demand = demand_data['Electricity demand [kWh]'].values
heat_demand = demand_data['Heating demand [kWh]'].values

# Renewable energy potentials
# ============================
solar = pd.read_excel('solar.xlsx', header=None, names=['Solar radiation [kWh/m2]'])
wind = pd.read_excel('wind.xlsx', header=None, names=['Wind speed [m/s]'])

# Optimization horizon
# =====================
Horizon = 8760  # Hours in a calendar year

# Discounted cash flow calculations
# ==================================
d = XXX  # Interest rate used to discount future operational cashflows

# Connections with energy grids at the input of the energy system
# ================================================================

# Variable definitions
# --------------------
Imp_elec = cp.Variable(Horizon)  # Electricity import from the grid for every time step [kWh]
Imp_gas = cp.Variable(Horizon)  # Natural gas import from the grid for every time step [kWh]
Exp_elec = cp.Variable(Horizon)  # Electricity export from the grid for every time step [kWh]

# Parameter definitions
# ---------------------
price_gas = XXX  # Natural gas price [CHF, EUR, USD/kWh]
esc_gas = XXX  # Escalation rate per year for natural gas price
price_elec = XXX  # Grid electricity price [CHF, EUR, USD/kWh]
esc_elec = XXX  # Escalation rate per year for electricity price
exp_price_elec = XXX  # Feed-in tariff for exported electricity [CHF, EUR, USD/kWh]
esc_elec_exp = XXX  # Escalation rate per year for feed-in tariff for exported electricity [%]
co2_gas = XXX  # Natural gas emission factor [kgCO2/kWh]
co2_elec = XXX  # Electricity emission factor [kgCO2/kWh]

# Constraint definitions
# ----------------------
grid_con = [Imp_elec >= 0, Imp_gas >= 0, Exp_elec >= 0]

## Natural gas boiler (gb)
# ========================

# Parameter definitions
# ---------------------
eff_gb = XXX  # Conversion efficiency of gas boiler
cost_gb = XXX  # Investment cost for gas boiler [CHF, EUR, USD/kW]

# Capacity variable
# ------------------
Cap_gb = cp.Variable(1)  # Capacity of natural gas boiler [kW]

# Input and output variables
# --------------------------
P_in_gb = cp.Variable(Horizon)  # Input energy to natural gas boiler [kWh]
P_out_gb = cp.Variable(Horizon)  # Heat generation by natural gas boiler [kWh]

# Gas boiler constraints
# ----------------------
gb_con = [Cap_gb >= 0, P_out_gb == P_in_gb * eff_gb, P_in_gb >= 0, P_out_gb >= 0, P_out_gb <= Cap_gb]

# Ground-source heat pump (gshp)
# ===============================

# Parameter definitions
# ---------------------
eff_gshp = XXX  # Conversion efficiency (Coefficient of Performance) of ground-source heat pump
cost_gshp = XXX  # Investment cost for ground-source heat pump [CHF, EUR, USD/kW]

# Capacity variables
# ------------------
Cap_gshp = cp.Variable(1)  # Capacity of ground-source heat pump [kW]

# Input and output variables
# --------------------------
P_in_gshp = cp.Variable(Horizon)  # Input energy to ground-source heat pump [kWh]
P_out_gshp = cp.Variable(Horizon)  # Heat generation by ground-source heat pump [kWh]

# GSHP constraints
# ----------------
gshp_con = [Cap_gshp >= 0, P_out_gshp == P_in_gshp * eff_gshp, P_in_gshp >= 0, P_out_gshp >= 0, P_out_gshp <= Cap_gshp]

# Combined heat and power engine (chp)
# =====================================

# Parameter definitions
# ---------------------
eff_elec_chp = XXX  # Electrical efficiency of combined heat and power engine
eff_heat_chp = XXX  # Thermal efficiency of combined heat and power engine
cost_chp = XXX  # Investment cost for combined heat and power engine [CHF, EUR, USD/kWe]

# Capacity variable
# -----------------
Cap_chp = cp.Variable(1)  # Electrical capacity of combined heat and power engine [kWe]

# Input and output variables
# --------------------------
P_in_chp = cp.Variable(Horizon)  # Input energy to combined heat and power engine (natural gas) [kWh]
P_out_heat_chp = cp.Variable(Horizon)  # Heat generation by combined heat and power engine [kWh]
P_out_elec_chp = cp.Variable(Horizon)  # Electricity generation by combined heat and power engine [kWh]

# CHP constraints
# ---------------
chp_con = [Cap_chp >= 0, P_out_heat_chp == P_in_chp * eff_heat_chp, P_out_elec_chp == P_in_chp * eff_elec_chp,
           P_in_chp >= 0, P_out_heat_chp >= 0, P_out_elec_chp >= 0, P_out_elec_chp <= Cap_chp]

# Photovoltaic panels
# ====================

# Definitions
# -----------
eff_pv = XXX  # Conversion efficiency (Coefficient of Performance) of photovoltaic panels
cost_pv = XXX  # Investment cost for photovoltaic panels [CHF, EUR, USD/m2]
max_solar_area = XXX  # Maximum available area to accommodate photovoltaic panels [m2]

# Capacity variable
# -----------------
Cap_pv = cp.Variable(1)  # Capacity of photovoltaic panels [m2]

# Input and output variables
# --------------------------
P_out_pv = cp.Variable(Horizon)  # Electricity generation by photovoltaic panels [kWh]

# PV constraints
# --------------
pv_con = [Cap_pv >= 0, Cap_pv <= max_solar_area, P_out_pv >= 0,
          P_out_pv == solar['Solar radiation [kWh/m2]'].values * Cap_pv * eff_pv]

# Wind turbine
# =============

# Definitions
# -----------
cut_out_wind_speed = XXX  # Cut-off wind speed [m/s]
cut_in_wind_speed = XXX  # Cut-in wind speed [m/s]
rated_wind_speed = XXX  # Rated wind speed [m/s]
cost_wind = XXX  # Investment cost for wind turbines [CHF, EUR, USD/kW]
max_wind_cap = XXX  # Maximum possible capacity of wind turbines that can be accommodated [kW]

# Capacity variable
# -----------------
Cap_wind = cp.Variable(1)  # Capacity of wind turbines [kW]

# Input and output variables
# --------------------------
P_out_wind = cp.Variable(Horizon)  # Electricity generation by wind turbines [kWh]

# Wind constraints
# ----------------
wind_con = [Cap_wind >= 0, Cap_wind <= max_wind_cap, P_out_wind >= 0]

for t in np.arange(0, 8760):
    if wind.loc[t, 'Wind speed [m/s]'] <= cut_in_wind_speed or wind.loc[t, 'Wind speed [m/s]'] >= cut_out_wind_speed:
        wind_con = wind_con + [P_out_wind[t] == 0]
    elif wind.loc[t, 'Wind speed [m/s]'] < cut_out_wind_speed and wind.loc[t, 'Wind speed [m/s]'] >= rated_wind_speed:
        wind_con = wind_con + [P_out_wind[t] == Cap_wind]
    else:
        wind_con = wind_con + [P_out_wind[t] == Cap_wind * (wind.loc[t, 'Wind speed [m/s]'] - cut_in_wind_speed) / (
                    rated_wind_speed - cut_in_wind_speed)]

# Thermal storage tank
# =====================

# Definitions
# -----------
self_dis_ts = XXX  # Self-discharging losses of thermal storage tank
ch_eff_ts = XXX  # Charging efficiency of thermal storage tank
dis_eff_ts = XXX  # Discharging efficiency of thermal storage tank
max_ch_ts = XXX  # Maximum charging rate of thermal storage tank (given as percentage of tank capacity)
max_dis_ts = XXX  # Maximum discharging rate of thermal storage tank
cost_ts = XXX  # Investment cost for thermal storage tank [CHF, EUR, USD/kWh]

# Capacity variables
# ------------------
Cap_ts = cp.Variable(1)  # Capacity of thermal storage tank [kWh]

# Storage variables
# -----------------
Q_in_ts = cp.Variable(Horizon)  # Input energy flow to thermal storage tank [kWh]
Q_out_ts = cp.Variable(Horizon)  # Output energy flow from thermal storage tank [kWh]
E_ts = cp.Variable(Horizon + 1)  # Stored energy in thermal storage tank [kWh]

# Storage tank constraints
# ------------------------
ts_con_1 = [Cap_ts >= 0, Q_in_ts >= 0, Q_out_ts >= 0, E_ts >= 0, E_ts <= Cap_ts, Q_in_ts <= max_ch_ts * Cap_ts,
            Q_out_ts <= max_dis_ts * Cap_ts]

# Storage constraints
# -------------------
ts_con_2 = [E_ts[1:] == (1 - self_dis_ts) * E_ts[:-1] + ch_eff_ts * Q_in_ts - (1 / dis_eff_ts) * Q_out_ts, E_ts[0] == 0]

# Combine constraints
# -------------------
ts_con = ts_con_1 + ts_con_2

# Battery
# ========

# Definitions
# -----------
self_dis_bat = XXX  # Self-discharging losses of battery
ch_eff_bat = XXX  # Charging efficiency of battery
dis_eff_bat = XXX  # Discharging efficiency of battery
max_ch_bat = XXX  # Maximum charging rate of battery (as percentage of capacity)
max_dis_bat = XXX  # Maximum discharging rate of battery
cost_bat = XXX  # Investment cost for battery [CHF, EUR, USD/kWh]

# Capacity variables
# ------------------
Cap_bat = cp.Variable(1)  # Capacity of battery [kWh]

# Storage variables
# -----------------
Q_in_bat = cp.Variable(Horizon)  # Input energy flow to battery [kWh]
Q_out_bat = cp.Variable(Horizon)  # Output energy flow from battery [kWh]
E_bat = cp.Variable(Horizon + 1)  # Stored energy in battery [kWh]

# Battery constraints
# -------------------
bat_con_1 = [Cap_bat >= 0, Q_in_bat >= 0, Q_out_bat >= 0, E_bat >= 0, E_bat <= Cap_bat,
             Q_in_bat <= max_ch_bat * Cap_bat, Q_out_bat <= max_dis_bat * Cap_bat]

# Battery constraints
# -------------------
bat_con_2 = [E_bat[1:] == (1 - self_dis_bat) * E_bat[:-1] + ch_eff_bat * Q_in_bat - (1 / dis_eff_bat) * Q_out_bat,
             E_bat[1] == 0]

# Combine constraints
# -------------------
bat_con = bat_con_1 + bat_con_2

# Balance equations
# ==================
heat_con = [P_out_heat_chp + P_out_gshp + P_out_gb + Q_out_ts - Q_in_ts == heat_demand[:Horizon]]  # Heat balance
power_con = [Imp_elec + P_out_pv + P_out_wind + P_out_elec_chp - P_in_gshp + Q_out_bat - Q_in_bat == elec_demand[
                                                                                                     :Horizon] + Exp_elec]  # Electricity balance
gas_con = [Imp_gas - P_in_gb - P_in_chp == 0]  # Natural gas balance

# Objective function
# ===================

# Total costs: Investment costs + 25 years of energy costs
# --------------------------------------------------------
Inv = Cap_gb * cost_gb + Cap_gshp * cost_gshp + Cap_chp * cost_chp + Cap_pv * cost_pv + Cap_wind * cost_wind + Cap_ts * cost_ts + Cap_bat * cost_bat

Op = cp.Variable(25)
op_con = []
for y in np.arange(0, 25):
    op_con = op_con + [Op[y] == cp.sum(Imp_gas * price_gas * (np.power(1 + esc_gas, y - 1))) + cp.sum(
        Imp_elec * price_elec * (np.power(1 + esc_elec, y - 1))) - cp.sum(
        Exp_elec * exp_price_elec * (np.power(1 + esc_elec_exp, y - 1)))]

cost = Inv + cp.sum(Op / np.power((1 + d), np.arange(1, 26)))
co2 = 25 * cp.sum(Imp_gas * co2_gas + Imp_elec * co2_elec)

# Collect all constraints
# ========================
constraints = grid_con + gb_con + gshp_con + chp_con + pv_con + wind_con + ts_con + bat_con + heat_con + power_con + gas_con + op_con

# Start the optimization
# =======================

# Select the desired objective
# ----------------------------
objective = cost
# objective = co2

prob = cp.Problem(cp.Minimize(objective), constraints)
# Optimize the design of the energy system
# ----------------------------------------

print('Installed solvers:', cp.installed_solvers())
prob.solve(solver='MOSEK')

# Output objective function value
# ================================
print('The value of the total system cost is equal to: ', str(cost.value), ' CHF, EUR, USD')
print('The value of the total system emissions is equal to: ', str(co2.value), ' kg CO_2')

# Output optimal energy system design
# ====================================
print('The capacity of the gas boiler is: ', str(np.round(Cap_gb.value, 1)), ' kW')
print('The capacity of the combined heat and power engine is: ', str(np.round(Cap_chp.value)), ' kW')
print('The capacity of the ground-source heat pump is: ', str(np.round(Cap_gshp.value)), ' kW')
print('The capacity of the photovoltaic panels is: ', str(np.round(Cap_pv.value)), ' m2')
print('The capacity of the wind turbines is: ', str(np.round(Cap_wind.value)), ' kW')
print('The capacity of the thermal storage is: ', str(np.round(Cap_ts.value)), ' kWh')
print('The capacity of the battery is: ', str(np.round(Cap_bat.value)), ' kWh')

# Plot the optimal energy system operation results
# =================================================

# Power
t = np.arange(0, Horizon)
plt.figure()
plt.plot(t, elec_demand[:Horizon], label='Load')
plt.plot(t, P_out_elec_chp.value, label='CHP')
plt.plot(t, -P_in_gshp.value, label='GSHP')
plt.plot(t, P_out_pv.value, label='PV')
plt.plot(t, P_out_wind.value, label='Wind')
plt.plot(t, Imp_elec.value, label='Electricity grid')
plt.plot(t, Q_out_bat.value, label='Battery out')
plt.plot(t, -Q_in_bat.value, label='Battery in')
plt.legend()
plt.xlabel('Time [h]')
plt.ylabel('Output [kW]')
plt.title('Power node')
plt.tight_layout()
plt.savefig('fig1.png', bbox_inches='tight')
plt.show()

# Heat
plt.figure()
plt.plot(t, heat_demand[:Horizon], label='Load')
plt.plot(t, P_out_heat_chp.value, label='CHP')
plt.plot(t, P_out_gshp.value, label='GSHP')
plt.plot(t, P_out_gb.value, label='Gas boiler')
plt.plot(t, Q_out_ts.value, label='Storage tank out')
plt.plot(t, -Q_in_ts.value, label='Storage tank in')
plt.legend()
plt.xlabel('Time [h]')
plt.ylabel('Output [kW]')
plt.title('Heat node')
plt.tight_layout()
plt.savefig('fig2.png', bbox_inches='tight')
plt.show()

# Gas
plt.figure()
plt.plot(t, P_in_chp.value, label='CHP')
plt.plot(t, P_in_gb.value, label='Gas boiler')
plt.plot(t, Imp_gas.value, label='Natural gas grid')
plt.legend()
plt.xlabel('Time [h]')
plt.ylabel('Output [kW]')
plt.title('Gas node')
plt.tight_layout()
plt.savefig('fig3.png', bbox_inches='tight')
plt.show()

# End
