""" Energy Innovation and Management - Spring Semester 2024
 ========================================================================
 Author of original code: M.Hohmann                                         |
 Source: https://hues.empa.ch/index.php/Model:Generic_Energy_Hub_YALMIP     |
 Edited and extended by: G. Mavromatidis (gmavroma@ethz.ch) 28.01.2020      |
 Converted from Matlab to Python by: S. Powell (spowell@ethz.ch) 05.01.2024 |
 ========================================================================

 Technologies: Gas boiler, PV, battery, heat storage, HeatPump
 Parameters: See first list, Favela Area for PV, Income Average Household favela, Job created per technology, renewable ninja

Andrej: Gas Price, Electricity Price, Feed-in Tariff, Elecitricity Emission Factor, renewable ninja
Lukas: Income Average Household favela, Job created per technology, Favela Area for PV


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxpy as cp
import mosek

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
d = 0.03  # Interest rate used to discount future operational cashflows

# Connections with energy grids at the input of the energy system
# ================================================================

# Variable definitions
# --------------------
Imp_elec = cp.Variable(Horizon)  # Electricity import from the grid for every time step [kWh]
Imp_gas = cp.Variable(Horizon)  # Natural gas import from the grid for every time step [kWh]
Exp_elec = cp.Variable(Horizon)  # Electricity export from the grid for every time step [kWh]

# Parameter definitions
# ---------------------
price_gas = 100  # Natural gas price [CHF, EUR, USD/kWh] #ToDO find this for Brazil
esc_gas = 0.02  # Escalation rate per year for natural gas price
price_elec = 0.15  # Grid electricity price [CHF, EUR, USD/kWh] #ToDO find this for Brazil
esc_elec = 0.02  # Escalation rate per year for electricity price
exp_price_elec = 0.12  # Feed-in tariff for exported electricity [CHF, EUR, USD/kWh] #ToDO find this for Brazil
esc_elec_exp = 0.02  # Escalation rate per year for feed-in tariff for exported electricity [%]
co2_gas = 0.198  # Natural gas emission factor [kgCO2/kWh]
co2_elec = 0.0  # Electricity emission factor [kgCO2/kWh] #ToDO find this for Brazil

# Constraint definitions
# ----------------------
grid_con = [Imp_elec >= 0, Imp_gas >= 0, Exp_elec >= 0]

## Natural gas boiler (gb)
# ========================

# Parameter definitions
# ---------------------
eff_gb = 0.9  # Conversion efficiency of gas boiler
cost_gb = 110  # Investment cost for gas boiler [CHF, EUR, USD/kW]
jobs_created_gb = 0.00237 #[job years/ kW] excluding fuel related jobs as we have a pipeline already built with fuel + 15.1 jobs/PJ of gas

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
eff_gshp = 4  # Conversion efficiency (Coefficient of Performance) of ground-source heat pump
cost_gshp = 850  # Investment cost for ground-source heat pump [CHF, EUR, USD/kW]
jobs_created_gshp = 0.0073 #[job years/ kW]

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

# Combined heat and power engine (chp) -> not considered
# =====================================

# Parameter definitions
# ---------------------
eff_elec_chp = 0.3  # Electrical efficiency of combined heat and power engine
eff_heat_chp = 0.6  # Thermal efficiency of combined heat and power engine
cost_chp = 700  # Investment cost for combined heat and power engine [CHF, EUR, USD/kWe]

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
chp_con = [Cap_chp == 0, P_out_heat_chp == P_in_chp * eff_heat_chp, P_out_elec_chp == P_in_chp * eff_elec_chp,
           P_in_chp >= 0, P_out_heat_chp >= 0, P_out_elec_chp >= 0, P_out_elec_chp <= Cap_chp]

# Photovoltaic panels
# ====================

# Parameter definitions
size_favela = 1.2e6  # Area of the favela [m2]
percentage_area_roof = 0.56  # Percentage of the favela area that can be used for photovoltaic panels
jobs_created_pv = 0.0341 #[job years/ kW]

# Definitions
# -----------
eff_pv = 0.15  # Conversion efficiency (Coefficient of Performance) of photovoltaic panels
cost_pv = 250  # Investment cost for photovoltaic panels [CHF, EUR, USD/m2]
max_solar_area = size_favela*percentage_area_roof  # Maximum available area to accommodate photovoltaic panels [m2]

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

# Wind turbine -> not considered
# =============

# Definitions
# -----------
cut_out_wind_speed = 25  # Cut-off wind speed [m/s]
cut_in_wind_speed = 3  # Cut-in wind speed [m/s]
rated_wind_speed = 12.5  # Rated wind speed [m/s]
cost_wind = 1600  # Investment cost for wind turbines [CHF, EUR, USD/kW]
max_wind_cap = 1e2  # Maximum possible capacity of wind turbines that can be accommodated [kW]

# Capacity variable
# -----------------
Cap_wind = cp.Variable(1)  # Capacity of wind turbines [kW]

# Input and output variables
# --------------------------
P_out_wind = cp.Variable(Horizon)  # Electricity generation by wind turbines [kWh]

# Wind constraints
# ----------------
wind_con = [Cap_wind == 0, Cap_wind <= max_wind_cap, P_out_wind >= 0]

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
self_dis_ts = 0.01  # Self-discharging losses of thermal storage tank
ch_eff_ts = 0.9  # Charging efficiency of thermal storage tank
dis_eff_ts = 0.9  # Discharging efficiency of thermal storage tank
max_ch_ts = 0.25  # Maximum charging rate of thermal storage tank (given as percentage of tank capacity)
max_dis_ts = 0.25 # Maximum discharging rate of thermal storage tank
cost_ts = 30  # Investment cost for thermal storage tank [CHF, EUR, USD/kWh]
jobs_creaated_ts = 0.00023 #[job years/ kW]

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
self_dis_bat = 0.001  # Self-discharging losses of battery
ch_eff_bat = 0.95  # Charging efficiency of battery
dis_eff_bat = 0.95  # Discharging efficiency of battery
max_ch_bat = 0.30   # Maximum charging rate of battery (as percentage of capacity)
max_dis_bat = 0.30  # Maximum discharging rate of battery
cost_bat = 350  # Investment cost for battery [CHF, EUR, USD/kWh]
jobs_created_bat = 0.0281 #[job years/ kW]

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
jobs = Cap_gb * jobs_created_gb + Cap_gshp * jobs_created_gshp + Cap_pv * jobs_created_pv + Cap_ts * jobs_creaated_ts + Cap_bat * jobs_created_bat

# Collect all constraints
# ========================
constraints = grid_con + gb_con + gshp_con + chp_con + pv_con + wind_con + ts_con + bat_con + heat_con + power_con + gas_con + op_con

# Start the optimization
# =======================

# Select the desired objective
# ----------------------------
#objective = cost
# objective = co2

#prob = cp.Problem(cp.Minimize(objective), constraints)
# Optimize the design of the energy system
# ----------------------------------------
print('Installed solvers:', cp.installed_solvers())
prob.solve(solver='SCIPY')

#Multi Objective Optimization -> cost and co2
eta = [i/10 for i in range(1,10,1)]
sol_cost = []
sol_co2 = []
#initial optimization
#minimze cost optimal
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver='SCIPY')
sol_cost.append(cost.value)
sol_co2.append(co2.value)
#minimize co2 optimal
prob = cp.Problem(cp.Minimize(co2), constraints)
prob.solve(solver='SCIPY')
sol_cost.append(cost.value)
sol_co2.append(co2.value)

for i in eta:
    print('Multi Objective Optimization with eta = ', i)
    co2_con = [co2 <= sol_co2[1]+i*(sol_co2[0]-sol_co2[1])]
    constraints_mo = constraints + co2_con
    #minimize cost and co2
    prob = cp.Problem(cp.Minimize(cost), constraints_mo)
    prob.solve(solver='SCIPY')
    sol_cost.append(cost.value)
    sol_co2.append(co2.value)



print(sol_co2)
print(sol_cost)

#Multiobjective Optimization -> cost and jobs
eta = [i/10 for i in range(1,10,1)]
sol_cost = []
sol_jobs = []
#initial optimization
#minimze cost optimal
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver='SCIPY')
sol_cost.append(cost.value)
sol_jobs.append(jobs.value)
#minimize jobs optimal
prob = cp.Problem(cp.Minimize(jobs), constraints)
prob.solve(solver='SCIPY')
sol_cost.append(cost.value)
sol_jobs.append(jobs.value)

for i in eta:
    print('Multi Objective Optimization with eta = ', i)
    jobs_con = [jobs >= sol_jobs[0]+i*(sol_jobs[1]-sol_jobs[0])]
    constraints_mo = constraints + jobs_con
    #minimize cost and co2
    prob = cp.Problem(cp.Minimize(cost), constraints_mo)
    prob.solve(solver='SCIPY')
    sol_cost.append(cost.value)
    sol_jobs.append(jobs.value)

#Investment Analysis
#ToDo add base case with investment to just meet the heat demand
#ToDO decide whether to do the costs in CHF or USD
#Parameters
annual_income_per_persom = 170*12 #Income per person per year [US Dollar] # in CHF: 155*12
percentage_invest = 0.01 #Percentage of income invested
populationsize = 55361 #Population size
average_government_exp = 198 #Average government expenses per person per year [US Dollar/Person] # in CHF: 180
Inv_bound = [annual_income_per_persom*percentage_invest*populationsize, average_government_exp*populationsize]

inv_sol_cost = []
inv_sol_co2 = []

for bound in Inv_bound:
    print('Investment Analysis with bound = ', bound)
    inv_con = [Inv <= bound]
    constraints_inv = constraints + inv_con
    #minimize cost and co2
    prob = cp.Problem(cp.Minimize(cost), constraints_inv)
    prob.solve(solver='SCIPY')
    inv_sol_cost.append(cost.value)
    inv_sol_co2.append(co2.value)

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
