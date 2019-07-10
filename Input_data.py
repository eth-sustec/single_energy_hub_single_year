# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:17:22 2019

@author: gmavroma
"""

import pandas as pd
import numpy as np

# Defining input values for model sets
# ====================================
Number_of_time_steps = 288
Time = list(range(1, Number_of_time_steps+1))
First_hour = list(range(1, Number_of_time_steps+1, 24))
Inputs = ['Grid', 'PV', 'ASHP', 'GSHP', 'Gas_Boiler', 'Bio_Boiler', 'Oil_Boiler', 'CHP']
Solar_inputs = ['PV']
Inputs_wo_grid = Inputs.copy()
Inputs_wo_grid.remove('Grid')
Dispatchable_Tech = ['ASHP', 'GSHP', 'Gas_Boiler', 'Bio_Boiler', 'Oil_Boiler', 'CHP']
CHP_Tech = ['CHP']
Outputs = ['Heat', 'Elec']

# Defining input values for model parameters
# ==========================================
Loads = pd.read_excel('Loads.xlsx', index_col=None, header=None)    # Read from some Excel/.csv file
Loads.index = range(1, 289)
Loads.columns = ['Heat','Elec']
Loads = Loads.stack().to_dict()

Number_of_days = list(np.repeat(2, 288))
P_solar = 1

Network_efficiency = 0.80
Network_length = 200
Network_lifetime = 50
Net_inv_cost_per_m = 800
Roof_area = 200

# Generation technologies
# -----------------------
# Operating_costs, Linear_inv_costs, Fixed_inv_costs, Carbon_factors, Lifetime_tech
gen_tech = {'Grid'       : [0.0256, 0, 0, 0.0095, 20], 
            'PV'         : [0.0, 300, 5750, 0, 20], 
            'ASHP'       : [0.0, 1530, 6830, 0, 20], 
            'GSHP'       : [0.0, 2170, 9070, 0, 20], 
            'Gas_Boiler' : [0.113, 640, 11920, 0.198, 20], 
            'Bio_Boiler' : [0.100, 1150, 24940, 0, 20], 
            'Oil_Boiler' : [0.106, 540, 15890, 0.265, 20], 
            'CHP'        : [0.113, 3100, 43450, 0.198, 20]
        }

Operating_costs = {key: gen_tech[key][0] for key in gen_tech.keys()}
Linear_inv_costs = {key: gen_tech[key][1] for key in gen_tech.keys()}
Fixed_inv_costs = {key: gen_tech[key][2] for key in gen_tech.keys()}
Carbon_factors = {key: gen_tech[key][3] for key in gen_tech.keys()}
Lifetime_tech = {key: gen_tech[key][4] for key in gen_tech.keys()}

FiT = 0.120
Interest_rate = 0.04

Cmatrix = {
        ('Elec', 'Grid')      : 1.0,
        ('Elec', 'PV')        : 0.15,
        ('Heat','ASHP')       : 3.0,
        ('Elec', 'ASHP')      : -1.0,
        ('Heat','GSHP')       : 4.0,
        ('Elec', 'GSHP')      : -1.0,
        ('Heat','Gas_Boiler') : 0.8,
        ('Heat','Bio_Boiler') : 0.8,
        ('Heat','Oil_Boiler') : 0.8,
        ('Heat','CHP')        : 0.8,
        ('Elec','CHP')        : 0.8
        }


# Linear_stor_costs, Fixed_stor_costs, Storage_max_charge, Storage_max_discharge, Storage_standing_losses, Storage_charging_eff, 
# Storage_discharging_eff, Storage_max_cap, Lifetime_stor
stor_tech = {'Heat' : [12.5, 1685, 0.25, 0.25, 0.01, 0.90, 0.90, 100, 20],
             'Elec' : [2000, 0, 0.25, 0.25, 0.001, 0.90, 0.90, 100, 20]
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
