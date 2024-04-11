""" Energy Innovation and Management - Spring Semester 2024
 ========================================================================
 Author of original code: M.Hohmann                                         |
 Source: https://hues.empa.ch/index.php/Model:Generic_Energy_Hub_YALMIP     |
 Edited and extended by: G. Mavromatidis (gmavroma@ethz.ch) 28.01.2020      |
 Converted from Matlab to Python by: S. Powell (spowell@ethz.ch) 05.01.2024 |
 ========================================================================

    ========================================================================
    Parameter sources and assumptions:
    -> Demands:
    - Heat Demand (onyl for cooking and showering): Heat demand was taken from renewables.ninja with choosing the coordinates
    - Electricity Demand (per month average household consumption at 170 kWh): Electricity demand for brazil was very hard to find, so we took a electricity profile of the US and 
                                                                                and scaled it accordingly so the energy consumption matches 170kWh per household.

    -> Energy Prices and Emission Factors:
    - Electricity Mix Emission in Brazil (kgCO2/kWh): https://www.climatiq.io/data/emission-factor/2ac52a91-5922-4f9f-8def-f4302f4ecf55


    -> Technology Parameters:
    - Jobs created per Capacity installed of technology: Supplementary data of https://www.sciencedirect.com/science/article/pii/S0360544221019381?via%3Dihub#appsec1
    - Income Average Household favela: https://rioonwatch.org/?p=57787
    - Area of favela: https://www.citypopulation.de/en/brazil/rio/_/33045570538__cidade_de_deus/
    - Area of favela that is covered by roofs (%): https://isprs-annals.copernicus.org/articles/IV-2-W5/437/2019/isprs-annals-IV-2-W5-437-2019.pdf (page 443) -> gives information of how densily houses are built in favelas -> thus how much of the total area is covered by roofs -> roughly 60%
    - Percenatage of roof area that can be covered by (flat) PV panels: https://www.nrel.gov/docs/fy14osti/60593.pdf -> roughly 50% (see page 6)

    -> Investment:
    - Average Government Expenses in energy per person in Brazil: https://www.gov.br/en/government-of-brazil/latest-news/2021/brazil-is-targeting-extensive-energy-investments#:~:text=According%20to%20official%20data%20from,and%20new%20sources%20of%20energy. -> Value given is total money spent -> divided by total population and then multiplied by population of favela

    -> Price of gas

    natrual gas: 	https://de.globalpetrolprices.com/USA/natural_gas_prices/
		            https://insightcrime.org/news/militias-price-gouging-locals-essential-services-rio-favelas/#:~:text=The%20average%20price%20of%20a,to%20Brazilian%20news%20site%20Globo.

    Electricity price:		https://rioonwatch.org/?p=66501

    feed in price:		https://www.roedl.com/renewable-energy-consulting/markets/countries/marketing-models-brazil#:~:text=Differently%20from%20some%20developed%20countries,used%20is%20%E2%80%9CNet%20Metering%E2%80%9D.

    elec emissions:		https://www.carbonfootprint.com/docs/2023_02_emissions_factors_sources_for_2022_electricity_v10.pdf

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import cvxpy as cp
import mosek
import data_import

#Help functions
def pie_plot_production(name):
    # Energy sources and their respective production values
    energy_sources = ['Solar', 'Natural Gas','CHP','HeatPump', 'Grid']
    production_values = [P_out_pv.value.sum(), P_out_gb.value.sum(), P_out_heat_chp.value.sum()+P_out_elec_chp.value.sum() , P_out_gshp.value.sum(), Imp_elec.value.sum()]

    # Creating the pie plot
    plt.figure(figsize=(8, 8))
    plt.pie(production_values, labels=energy_sources, autopct='%1.1f%%', startangle=140)
    plt.title('Energy Production by Source')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend()
    plt.savefig(name,dpi=300)
    plt.show()


def pie_plot_capacity(name):
    # Energy sources and their respective production values
    energy_sources = ['Solar', 'Natural Gas','CHP' ,'HeatPump', 'Battery', 'Heat Storage']
    production_values = [Cap_pv.value.sum(), Cap_gb.value.sum(),Cap_chp.value.sum(),Cap_gshp.value.sum(), Cap_bat.value.sum(), Cap_ts.value.sum()]

    # Filter out energy sources with production values of zero
    filtered_sources = []
    filtered_values = []
    for source, value in zip(energy_sources, production_values):
        if value != 0:
            filtered_sources.append(source)
            filtered_values.append(value)

    # Creating the pie plot
    plt.figure(figsize=(8, 8))
    plt.pie(filtered_values,labels=[f'{source}: {value}' for source, value in zip(filtered_sources, filtered_values)], startangle=140)
    plt.title('Capacity Installed by Source')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend()
    plt.savefig(name, dpi=300)
    plt.show()

#Energy Demands
"""

BRAZIL DATA IMPORT

"""

elec_demand, heat_demand = data_import.get_data()

"""

IMPORT FINISH

"""

# Renewable energy potentials
# ============================
#solar = pd.read_excel('solar.xlsx', header=None, names=['Solar radiation [kWh/m2]'])
solar = pd.read_csv('maruas_solar.csv', delimiter=',', comment='#')['swgdn']*0.001 # in kWh/m2
solar_header = solar.head(4)
solar.drop(solar.index[:4], inplace=True)
both_solar = [solar, solar_header]
solar = pd.concat(both_solar, ignore_index=True)
assert len(solar) == 8760


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
price_gas = 0.21*1.4  # Natural gas price [CHF, EUR, USD/kWh]  # USD: 0.231*1.4; CHF 0.21*1.4
esc_gas = 0.02  # Escalation rate per year for natural gas price # assumption: 2% per year -> average inflation rate
price_elec = 0.16  # Grid electricity price [CHF/kWh]
esc_elec = 0.02  # Escalation rate per year for electricity price
exp_price_elec = 0.0  # Feed-in tariff for exported electricity [CHF/kWh] #assumption no export possible -> a feed-in-tariff does not seem to be avaialble to such an extent in Brazil as in Europe
esc_elec_exp = 0.02  # Escalation rate per year for feed-in tariff for exported electricity [%]
co2_gas = 0.198  # Natural gas emission factor [kgCO2/kWh]
co2_elec = 0.1295  # Electricity emission factor [kgCO2/kWh]

# Constraint definitions
# ----------------------
grid_con = [Imp_elec >= 0, Imp_gas >= 0, Exp_elec >= 0]

## Natural gas boiler (gb)
# ========================

# Parameter definitions
# ---------------------
eff_gb = 0.9  # Conversion efficiency of gas boiler
cost_gb = 110  # Investment cost for gas boiler [CHF, EUR, USD/kW]
jobs_created_gb = 0.00237 #[job years/ kW] excluding fuel related jobs as we have a pipeline already built with fuel available

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

# Combined heat and power engine (chp)
# =====================================

# Parameter definitions
# ---------------------
eff_elec_chp = 0.3  # Electrical efficiency of combined heat and power engine
eff_heat_chp = 0.6  # Thermal efficiency of combined heat and power engine
cost_chp = 700  # Investment cost for combined heat and power engine [CHF, EUR, USD/kWe]
jobs_created_chp = 0.00076 # [job-years/kW]

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

# Parameter definitions
size_favela = 1.2e6  # Area of the favela [m2]
percentage_area_roof = 0.3  # Percentage of the favela area that can be used for photovoltaic panels
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
          P_out_pv == solar.values * Cap_pv * eff_pv]

# Wind turbine -> not considered
# =============

# Definitions
# -----------
cut_out_wind_speed = 25  # Cut-off wind speed [m/s]
cut_in_wind_speed = 3  # Cut-in wind speed [m/s]
rated_wind_speed = 12.5  # Rated wind speed [m/s]
cost_wind = 1600  # Investment cost for wind turbines [CHF, EUR, USD/kW]
max_wind_cap = 0  # Maximum possible capacity of wind turbines that can be accommodated [kW]

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
jobs_created_ts = 0.00023 #[job years/ kW]

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
bat_con_1 = [Cap_bat >= 0, Cap_bat <= 70000, Q_in_bat >= 0, Q_out_bat >= 0, E_bat >= 0, E_bat <= Cap_bat,
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
jobs = Cap_gb * jobs_created_gb + Cap_chp*jobs_created_chp + Cap_gshp * jobs_created_gshp + Cap_pv * jobs_created_pv + Cap_ts * jobs_created_ts + Cap_bat * jobs_created_bat

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
#prob.solve(solver='SCIPY')


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
pie_plot_production("pie_plot_production_min_cost.png")
pie_plot_capacity("pie_plot_capacity_min_cost.png")
total_area_pv = Cap_pv.value
print("Total area of PV panels installed in cost optimal case: ", total_area_pv)
print("Percentage of roof area covered by PV panels in cost optimal case: ", total_area_pv / max_solar_area * 100, "%")

#minimize co2 optimal
prob = cp.Problem(cp.Minimize(co2), constraints)
prob.solve(solver='SCIPY')
sol_cost.append(cost.value)
sol_co2.append(co2.value)
pie_plot_production("pie_plot_production_min_emission.png")
pie_plot_capacity("pie_plot_capacity_min_emission.png")
total_area_pv = Cap_pv.value
print("Total area of PV panels installed in emission optimal case: ", total_area_pv)
print("Percentage of roof area covered by PV panels in emission optimal case: ", total_area_pv / max_solar_area * 100, "%")



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

# Specify the file name
file_name = "emissions_costs_data.csv"

# Writing data to CSV file
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Emissions (kg)", "Costs"])  # Writing header
    writer.writerows(zip(list(sol_co2), sol_cost))  # Writing data rows

print("Data has been successfully saved to", file_name)


# Plotting
sol_co2 = pd.read_csv('emissions_costs_data.csv')['Emissions (kg)']
sol_cost = pd.read_csv('emissions_costs_data.csv')['Costs']

emission = list(sol_co2)
costs = [float(x[1:-1]) for x in sol_cost]
emission = emission[1:] + [emission[0]]
costs = costs[1:] + [costs[0]]
plt.plot(emission, costs, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Emissions (kg)')
plt.ylabel('Costs (CHF)')
plt.title('Emissions vs Costs')

# Displaying the plot
plt.grid(True)
plt.savefig('emission_vs_cost.png', dpi=300)
plt.show()




#Multiobjective Optimization -> cost and jobs
"""
Note that we tried to implement a multiobjective optimization with cost and jobs. 
However maximizing jobs did not converge even after adding constraints such as maximal allowed investment costs or 
an upper bound of capacity for each technology or an upper bound of jobs created.
Consequently, we decided to show jobs created as an output of the multiobjective optimization with cost and co2.
"""
eta = [i/10 for i in range(1,10,1)]
sol_cost = []
sol_jobs = []
sol_co2 = []
#initial optimization
#minimze cost optimal
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver='SCIPY')
sol_cost.append(cost.value)
sol_jobs.append(jobs.value)
sol_co2.append(co2.value)

#minimize jobs optimal
prob = cp.Problem(cp.Minimize(co2), constraints)
prob.solve(solver='SCIPY')
sol_cost.append(cost.value)
sol_jobs.append(jobs.value)
sol_co2.append(co2.value)
for i in eta:
    print('Multi Objective Optimization with eta = ', i)
    #jobs_con = [jobs >= sol_jobs[0]+i*(sol_jobs[1]-sol_jobs[0])]
    #jobs_max = [jobs <= 55361*25]
    #inv_max = [Inv <= sol_cost[0]*2.5]
    co2_con = [co2 <= sol_co2[1] + i * (sol_co2[0] - sol_co2[1])]
    constraints_mo = constraints + co2_con #+ jobs_con + jobs_max #+ inv_max
    #minimize cost and co2
    prob = cp.Problem(cp.Minimize(cost), constraints_mo)
    prob.solve(solver='SCIPY')
    sol_cost.append(cost.value)
    sol_jobs.append(jobs.value)

print(sol_jobs)
print(sol_cost)

# Specify the file name
file_name = "jobs_costs_data.csv"

# Writing data to CSV file
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Jobs", "Costs"])  # Writing header
    writer.writerows(zip(list(sol_jobs), sol_cost))  # Writing data rows

print("Data has been successfully saved to", file_name)

# Plotting
sol_jobs = pd.read_csv('jobs_costs_data.csv')['Jobs']
sol_cost = pd.read_csv('jobs_costs_data.csv')['Costs']


jobs = [float(x[1:-1]) for x in sol_jobs]
costs = [float(x[1:-1]) for x in sol_cost]
jobs = jobs[1:] + [jobs[0]]
costs = costs[1:] + [costs[0]]
plt.plot(jobs, costs, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('Jobs created')
plt.ylabel('Costs (CHF)')
plt.title('Jobs vs Costs')

# Displaying the plot
plt.grid(True)
plt.savefig('jobs_vs_cost.png', dpi=300)
plt.show()


#Investment Analysis
#Parameters
annual_income_per_persom = 155*12 #Income per person per year [US Dollar] # in CHF: 155*12; in USD: 170*12
percentage_invest = 0.01 #Percentage of income invested
populationsize = 55361 #Population size
average_government_exp = 180 #Average government expenses per person per year [US Dollar/Person] # in CHF: 180; in USD: 198
inv_base_case = heat_demand.max()*cost_chp*(eff_elec_chp/eff_heat_chp) #Investment for base case -> as there is no heat grid to import heat directly, at least the given heat demand must be met to ensure feasibility, thus there must be enough money to invest in the cheapest heat source to meet the maximum demand
Inv_bound = [inv_base_case,annual_income_per_persom*percentage_invest*populationsize, average_government_exp*populationsize]

inv_sol_cost = []
inv_sol_co2 = []
inv_sol_jobs = []

for bound in Inv_bound:
    print('Investment Analysis with bound = ', bound)
    inv_con = [Inv <= bound]
    constraints_inv = constraints + inv_con
    #minimize cost and co2
    prob = cp.Problem(cp.Minimize(cost), constraints_inv)
    prob.solve(solver='SCIPY')
    inv_sol_cost.append(cost.value)
    inv_sol_co2.append(co2.value)
    inv_sol_jobs.append(jobs.value[0])

#unlimited investment
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver='SCIPY')
inv_sol_cost.append(cost.value)
inv_sol_co2.append(co2.value)
inv_sol_jobs.append(jobs.value)


print(inv_sol_co2)
print(inv_sol_cost)

# Specify the file name
file_name = "investment_costs_data.csv"

# Writing data to CSV file
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Emissions (kg)", "Costs"])  # Writing header
    writer.writerows(zip(list(inv_sol_co2), inv_sol_cost))  # Writing data rows

print("Data has been successfully saved to", file_name)

file_name = "investment_jobs_data.csv"

# Writing data to CSV file
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Jobs", "Costs"])  # Writing header
    writer.writerows(zip(list(inv_sol_jobs), inv_sol_cost))  # Writing data rows

print("Data has been successfully saved to", file_name)

# Plotting
inv_sol_co2 = pd.read_csv('investment_costs_data.csv')['Emissions (kg)']
inv_sol_cost = pd.read_csv('investment_costs_data.csv')['Costs']
inv_sol_jobs = pd.read_csv('investment_jobs_data.csv')['Jobs']

#PLot
# Sample data (replace with your actual data)
scenarios = ['Base Case', 'Income', 'Government', 'Unlimited']
emissions = list(inv_sol_co2)  # Emissions in kg
costs = list([float(x[1:-1]) for x in inv_sol_cost])
jobs_created = list([float(x) for x in inv_sol_jobs[0:-1]])
jobs_created+=[float(inv_sol_jobs[3][1:-1])]

# Setting up positions for the bars
x = np.arange(len(scenarios))  # the scenario locations
width = 0.35  # the width of the bars

# Plotting emissions
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, emissions, width, label='Emissions in kg CO2')

# Plotting costs
bars2 = ax.bar(x + width/2, costs, width, label='Costs in CHF')

# Adding labels and title
ax.set_xlabel('Scenarios')
ax.set_ylabel('Values')
ax.set_title('Emissions and Costs for Different Scenarios')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.legend()

# Show plot
plt.savefig("investment.png",dpi=300)
plt.show()

#PLot with Jobs created

# Setting up positions for the bars
x = np.arange(len(scenarios))  # the scenario locations
width = 0.2  # the width of the bars

# Plotting emissions
fig, ax = plt.subplots()
bars1 = ax.bar(x - width, emissions, width, label='Emissions in kg CO2')

# Plotting costs
bars2 = ax.bar(x, costs, width, label='Costs in CHF')

# Plotting jobs created (using a secondary y-axis)
ax2 = ax.twinx()
bars3 = ax2.bar(x + width, jobs_created, width, color='orange', label='Jobs Created')

# Adding labels and title
ax.set_xlabel('Scenarios')
ax.set_ylabel('Emissions and Costs')
ax2.set_ylabel('Jobs Created')
ax.set_title('Emissions, Costs, and Jobs Created for Different Scenarios')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
# Positioning legends
ax.legend(loc='upper center', bbox_to_anchor=(0.67, 1))
ax2.legend(loc='upper center', bbox_to_anchor=(0.32, 1))


# Show plot
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.savefig("investment_with_jobs.png", dpi=300)
plt.show()

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




