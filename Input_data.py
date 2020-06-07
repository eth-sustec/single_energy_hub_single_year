# -*- coding: utf-8 -*-
"""
Input data for the simple energy hub model (only energy supply)

@author: Georgios Mavromatidis (ETH Zurich, gmavroma@ethz.ch)
"""

import pandas as pd

# import numpy as np

# Defining input values for model sets
# ====================================
Number_of_days = 14
Days = list(range(1, Number_of_days + 1))

Number_of_time_steps = 24
Time_steps = list(range(1, Number_of_time_steps + 1))

Solar_tech = ["PV", "ST"]
Dispatchable_tech = ["ASHP", "GSHP", "Gas_Boiler", "Oil_Boiler", "Bio_Boiler", "CHP"]
Conversion_tech = Dispatchable_tech + Solar_tech
Storage_tech = ["Thermal_storage_tank", "Battery"]
Energy_carriers = ["Heat", "Elec", "NatGas", "Oil", "Biomass"]
Energy_carriers_imp = ["Elec", "NatGas", "Oil", "Biomass"]
Energy_carriers_exp = ["Elec"]
Energy_carriers_dem = ["Heat", "Elec"]

# Defining input values for model parameters
# ==========================================
Demands = pd.read_excel(
    "Time_series_inputs.xlsx", usecols="A:D", index_col=[0,1], header=[0], sheet_name="Loads_solar"
)  # Read from some Excel/.csv file
Demands = Demands.stack().reorder_levels([2,0,1]).to_dict()

Number_of_days = pd.read_excel(
    "Time_series_inputs.xlsx", usecols="A:B", index_col=0, header=0, sheet_name="Number_of_days"
)  # Read from some Excel/.csv file
Number_of_days = Number_of_days.to_dict()
Number_of_days = Number_of_days["Number_of_days"]

C_to_T = pd.read_excel(
    "Time_series_inputs.xlsx", usecols="A:B", index_col=0, header=None, sheet_name="C_to_T_matching"
)  # Read from some Excel/.csv file
C_to_T = C_to_T.to_dict()
C_to_T = C_to_T[1]


P_solar = pd.read_excel(
    "Time_series_inputs.xlsx", usecols="A:B,E", index_col=[0,1], header=[0], sheet_name="Loads_solar"
)  # Read from some Excel/.csv file
P_solar = P_solar.to_dict()
P_solar = P_solar["P_solar"]

Discount_rate = 0.080

Network_efficiency = {"Heat": 0.90, "Elec": 1.00}
Network_length = 200
Network_lifetime = 40
Network_inv_cost_per_m = 800

Roof_area = 1260


# Generation technologies
# -----------------------
# Linear_conv_costs, Fixed_conv_costs, Lifetime_tech
gen_tech = {    
    "PV": [300, 5750, 20],
    "ST": [1590, 7630, 20],
    "ASHP": [1530, 6830, 20],
    "GSHP": [2170, 9070, 20],
    "Gas_Boiler": [640, 11920, 20],
    "Bio_Boiler": [1150, 24940, 20],
    "Oil_Boiler": [540, 15890, 20],
    "CHP": [3100, 43450, 20],
}

Linear_conv_costs = {key: gen_tech[key][0] for key in gen_tech.keys()}
Fixed_conv_costs = {key: gen_tech[key][1] for key in gen_tech.keys()}
Lifetime_tech = {key: gen_tech[key][2] for key in gen_tech.keys()}

Export_prices = {'Elec': 0.120}
Import_prices = {'Elec': 0.256, 'NatGas': 0.113, 'Oil': 0.106, "Biomass": 0.100}
Carbon_factors_import = {'Elec': 0.0095, 'NatGas': 0.198, 'Oil': 0.265, "Biomass": 0.0}

Conv_factor = {
    ("PV", "Elec"): 0.15,
    ("ST", "Heat"): 0.35,
    ("ASHP", "Heat"): 3.0,
    ("ASHP", "Elec"): -1.0,
    ("GSHP", "Heat"): 4.0,
    ("GSHP", "Elec"): -1.0,
    ("Gas_Boiler", "Heat"): 0.9,
    ("Gas_Boiler", "NatGas"): -1.0,
    ("Bio_Boiler", "Heat"): 0.9,
    ("Bio_Boiler", "Biomass"): -1.0,
    ("Oil_Boiler", "Heat"): 0.9,
    ("Oil_Boiler", "Oil"): -1.0,
    ("CHP", "Heat"): 0.6,
    ("CHP", "Elec"): 0.3,
    ("CHP", "NatGas"): -1.0,
}

Minimum_part_load = {
    "ASHP": 0.0,
    "GSHP": 0.0,
    "Gas_Boiler": 0.0,
    "Bio_Boiler": 0.0,
    "Oil_Boiler": 0.0,
    "CHP": 0.0,
}

# Energy storage technologies
# ---------------------------
# Linear_stor_costs, Fixed_stor_costs, Storage_max_charge, Storage_max_discharge, Storage_standing_losses, Storage_charging_eff,
# Storage_discharging_eff, Storage_max_cap, Lifetime_stor
stor_tech = {
    "Thermal_storage_tank": [12.5, 1685, 0.25, 0.25, 0.01, 0.90, 0.90, 600, 20],
    "Battery": [2000, 0, 0.25, 0.25, 0.001, 0.90, 0.90, 600, 20],
}

Linear_stor_costs = {key: stor_tech[key][0] for key in stor_tech.keys()}
Fixed_stor_costs = {key: stor_tech[key][1] for key in stor_tech.keys()}
Storage_max_charge = {key: stor_tech[key][2] for key in stor_tech.keys()}
Storage_max_discharge = {key: stor_tech[key][3] for key in stor_tech.keys()}
Storage_standing_losses = {key: stor_tech[key][4] for key in stor_tech.keys()}
Storage_charging_eff = {key: stor_tech[key][5] for key in stor_tech.keys()}
Storage_discharging_eff = {key: stor_tech[key][6] for key in stor_tech.keys()}
Storage_max_cap = {key: stor_tech[key][7] for key in stor_tech.keys()}
Lifetime_stor = {key: stor_tech[key][8] for key in stor_tech.keys()}

Storage_tech_coupling = {
        ("Thermal_storage_tank","Heat"): 1.0,
        ("Battery", "Elec"): 1.0
}