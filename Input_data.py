# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:17:22 2019

@author: Georgios Mavromatidis (ETH Zurich, gmavroma@ethz.ch)
"""

import pandas as pd

# import numpy as np

# Defining input values for model sets
# ====================================
Number_of_time_steps = 336
Time = list(range(1, Number_of_time_steps + 1))
First_hour = list(range(1, Number_of_time_steps + 1, 24))
Number_of_buildings = 10
Solar_pv_inputs = ["PV" + str(b) for b in range(1, Number_of_buildings + 1)]
Solar_th_inputs = ["ST" + str(b) for b in range(1, Number_of_buildings + 1)]
Dispatchable_Tech = ["ASHP", "GSHP", "Gas_Boiler", "Bio_Boiler", "Oil_Boiler", "CHP"]
Inputs = ["Grid"] + Dispatchable_Tech + Solar_pv_inputs + Solar_th_inputs
Inputs_wo_grid = Inputs.copy()
Inputs_wo_grid.remove("Grid")
CHP_Tech = ["CHP"]
Outputs = ["Heat", "Elec"]

# Defining input values for model parameters
# ==========================================
Loads = pd.read_excel(
    "Time_series_inputs.xlsx", usecols="A:B", index_col=None, header=None
)  # Read from some Excel/.csv file
Loads.index = range(1, Number_of_time_steps + 1)
Loads.columns = ["Heat", "Elec"]
Loads = Loads.stack().to_dict()

Number_of_days = pd.read_excel(
    "Time_series_inputs.xlsx", usecols="C", index_col=None, header=None
)  # Read from some Excel/.csv file
Number_of_days.index = range(1, Number_of_time_steps + 1)
Number_of_days.columns = ["Number_of_days"]
Number_of_days = Number_of_days.to_dict()
Number_of_days = Number_of_days["Number_of_days"]

P_solar = pd.read_excel(
    "Time_series_inputs.xlsx", usecols="D:M", index_col=None, header=None
)  # Read from some Excel/.csv file
P_solar.index = range(1, Number_of_time_steps + 1)
P_solar.columns = [i for i in range(1, Number_of_buildings + 1)]
P_solar = P_solar.stack().to_dict()

Interest_rate = 0.080

Network_efficiency = {"Heat": 0.90, "Elec": 1.00}
Network_length = 200
Network_lifetime = 40
Network_inv_cost_per_m = 800
CRF_network = (Interest_rate * (1 + Interest_rate) ** Network_lifetime) / (
    (1 + Interest_rate) ** Network_lifetime - 1
)

Roof_area_per_bldg = [265, 21, 227, 162, 162, 215, 50, 50, 94, 14]
Roof_area = {key + 1: Roof_area_per_bldg[key] for key in range(0, Number_of_buildings)}


# Generation technologies
# -----------------------
# Operating_costs, Linear_inv_costs, Fixed_inv_costs, Carbon_factors, Lifetime_tech
gen_tech = {
    "Grid": [0.256, 0, 0, 0.0095, 20],
    "PV1": [0.0, 300, 5750, 0, 20],
    "ST1": [0.0, 1590, 7630, 0, 20],
    "ASHP": [0.0, 1530, 6830, 0, 20],
    "GSHP": [0.0, 2170, 9070, 0, 20],
    "Gas_Boiler": [0.113, 640, 11920, 0.198, 20],
    "Bio_Boiler": [0.100, 1150, 24940, 0, 20],
    "Oil_Boiler": [0.106, 540, 15890, 0.265, 20],
    "CHP": [0.113, 3100, 43450, 0.198, 20],
}
gen_tech.update(
    {
        key: gen_tech["PV1"]
        for key in ["PV" + str(i) for i in range(2, Number_of_buildings + 1)]
    }
)
gen_tech.update(
    {
        key: gen_tech["ST1"]
        for key in ["ST" + str(i) for i in range(2, Number_of_buildings + 1)]
    }
)

Operating_costs = {key: gen_tech[key][0] for key in gen_tech.keys()}
Linear_inv_costs = {key: gen_tech[key][1] for key in gen_tech.keys()}
Fixed_inv_costs = {key: gen_tech[key][2] for key in gen_tech.keys()}
Carbon_factors = {key: gen_tech[key][3] for key in gen_tech.keys()}
Lifetime_tech = {key: gen_tech[key][4] for key in gen_tech.keys()}
CRF_tech = {
    key: (Interest_rate * (1 + Interest_rate) ** gen_tech[key][4])
    / ((1 + Interest_rate) ** gen_tech[key][4] - 1)
    for key in gen_tech.keys()
}

FiT = 0.120

Cmatrix = {
    ("Elec", "Grid"): 1.0,
    ("Elec", "PV1"): 0.15,
    ("Heat", "ST1"): 0.35,
    ("Heat", "ASHP"): 3.0,
    ("Elec", "ASHP"): -1.0,
    ("Heat", "GSHP"): 4.0,
    ("Elec", "GSHP"): -1.0,
    ("Heat", "Gas_Boiler"): 0.9,
    ("Heat", "Bio_Boiler"): 0.9,
    ("Heat", "Oil_Boiler"): 0.9,
    ("Heat", "CHP"): 0.6,
    ("Elec", "CHP"): 0.3,
}
Cmatrix.update(
    {
        key: Cmatrix[("Elec", "PV1")]
        for key in [("Elec", "PV" + str(i)) for i in range(2, Number_of_buildings + 1)]
    }
)
Cmatrix.update(
    {
        key: Cmatrix[("Heat", "ST1")]
        for key in [("Heat", "ST" + str(i)) for i in range(2, Number_of_buildings + 1)]
    }
)

# Energy storage technologies
# ---------------------------
# Linear_stor_costs, Fixed_stor_costs, Storage_max_charge, Storage_max_discharge, Storage_standing_losses, Storage_charging_eff,
# Storage_discharging_eff, Storage_max_cap, Lifetime_stor
stor_tech = {
    "Heat": [12.5, 1685, 0.25, 0.25, 0.01, 0.90, 0.90, 600, 20],
    "Elec": [2000, 0, 0.25, 0.25, 0.001, 0.90, 0.90, 600, 20],
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
CRF_stor = {
    key: (Interest_rate * (1 + Interest_rate) ** stor_tech[key][8])
    / ((1 + Interest_rate) ** stor_tech[key][8] - 1)
    for key in stor_tech.keys()
}
