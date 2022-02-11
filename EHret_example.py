# -*- coding: utf-8 -*-
"""
@author: Dr. Georgios Mavromatidis (ETH Zurich, gmavroma@ethz.ch)
"""

import pandas as pd

ehr_inp = dict()

# Defining input values for model sets
# ====================================
Number_of_scenarios = 10

Number_of_days = 14
ehr_inp["Days"] = list(range(1, Number_of_days + 1))

Number_of_time_steps = 24
ehr_inp["Time_steps"] = list(range(1, Number_of_time_steps + 1))


ehr_inp["Solar_tech"] = ["PV", "ST"]
ehr_inp["PV_tech"] = ["PV"]
ehr_inp["ST_tech"] = ["ST"]
ehr_inp["Dispatchable_tech"] = [
    "ASHP",
    "GSHP",
    "Gas_Boiler",
    "Oil_Boiler",
    "Bio_Boiler",
    "CHP",
]
ehr_inp["Conversion_tech"] = ehr_inp["Dispatchable_tech"] + ehr_inp["Solar_tech"]
ehr_inp["Storage_tech"] = ["Thermal_storage_tank", "Battery"]
ehr_inp["Energy_carriers"] = ["Heat", "Elec", "NatGas", "Oil", "Biomass"]
ehr_inp["Energy_carriers_imp"] = ["Elec", "NatGas", "Oil", "Biomass"]
ehr_inp["Energy_carriers_exp"] = ["Elec"]
ehr_inp["Energy_carriers_dem"] = ["Heat", "Elec"]

ehr_inp["Retrofit_scenarios"] = [
    "Hempcrete01",
    "Hempcrete025",
    "EPS01",
    "EPS025",
    "Rockwool01",
    "Rockwool025",
    "Straw01",
    "Straw025",
    "Woodfibre01",
    "Woodfibre025",
]

# Defining input values for model parameters
# ==========================================
Demands = pd.read_excel(
    "Time_series_inputs_retrofit.xlsx",
    index_col=[0, 1],
    header=[0, 1],
    sheet_name="Loads",
)  # Read from some Excel/.csv file
ehr_inp["Demands"] = Demands.stack().stack().reorder_levels([3, 2, 0, 1]).to_dict()

Number_of_days = pd.read_excel(
    "Time_series_inputs_retrofit.xlsx",
    index_col=0,
    header=0,
    sheet_name="Number_of_days",
)  # Read from some Excel/.csv file
ehr_inp["Number_of_days"] = Number_of_days.stack().reorder_levels([1, 0]).to_dict()

C_to_T = pd.read_excel(
    "Time_series_inputs_retrofit.xlsx",
    index_col=0,
    header=0,
    sheet_name="C_to_T_matching",
)  # Read from some Excel/.csv file
ehr_inp["C_to_T"] = C_to_T.stack().reorder_levels([1, 0]).to_dict()


P_solar = pd.read_excel(
    "Time_series_inputs_retrofit.xlsx",
    index_col=[0, 1],
    header=[0],
    sheet_name="Solar",
)  # Read from some Excel/.csv file
ehr_inp["P_solar"] = P_solar.stack().reorder_levels([2, 0, 1]).to_dict()

Grid_intensity = pd.read_excel(
    "Time_series_inputs_retrofit.xlsx", index_col=[0, 1], header=[0], sheet_name="Grid",
)  # Read from some Excel/.csv file
ehr_inp["Grid_intensity"] = Grid_intensity.stack().reorder_levels([2, 0, 1]).to_dict()


Elec_import_prices = pd.read_excel(
    "Time_series_inputs_retrofit.xlsx",
    index_col=[0, 1],
    header=[0],
    sheet_name="Import_elec",
)  # Read from some Excel/.csv file
ehr_inp["Elec_import_prices"] = (
    Elec_import_prices.stack().reorder_levels([2, 0, 1]).to_dict()
)

Elec_export_prices = pd.read_excel(
    "Time_series_inputs_retrofit.xlsx",
    index_col=[0, 1],
    header=[0],
    sheet_name="Export_elec",
)  # Read from some Excel/.csv file
ehr_inp["Elec_export_prices"] = (
    Elec_export_prices.stack().reorder_levels([2, 0, 1]).to_dict()
)

ehr_inp["Discount_rate"] = 0.080

ehr_inp["Network_efficiency"] = {"Heat": 1.00, "Elec": 1.00}
ehr_inp["Network_length"] = 0  # set to 0 as this is not considered
ehr_inp["Network_lifetime"] = 10
ehr_inp["Network_inv_cost_per_m"] = 0  # set to 0 as this is not considered

ehr_inp["Roof_area"] = 120

ehr_inp["Carbon_benefit_CO2_certificates"] = 0
ehr_inp["Cost_CO2_certificates"] = 35.7

# Generation technologies
# -----------------------
# Linear_conv_costs, Fixed_conv_costs, Lifetime_tech, Embodied_emissions_linear, Embodied_emissions_fixed
gen_tech = {
    "PV": [115, 17160, 20, 254, 0],
    "ST": [1000, 4000, 20, 184, 0],
    "ASHP": [610, 49635, 20, 75, 2329],
    "GSHP": [2450, 49130, 20, 72, 1806],
    "Gas_Boiler": [620, 27600, 20, 51, 0],
    "Oil_Boiler": [570, 26600, 20, 51, 0],
    "Bio_Boiler": [320, 55885, 20, 51, 0],
    "CHP": [790, 63280, 20, 100, 3750],
}

ehr_inp["Linear_conv_costs"] = {key: gen_tech[key][0] for key in gen_tech.keys()}
ehr_inp["Fixed_conv_costs"] = {key: gen_tech[key][1] for key in gen_tech.keys()}
ehr_inp["Lifetime_tech"] = {key: gen_tech[key][2] for key in gen_tech.keys()}
ehr_inp["Embodied_emissions_conversion_tech_linear"] = {
    key: gen_tech[key][3] for key in gen_tech.keys()
}
ehr_inp["Embodied_emissions_conversion_tech_fixed"] = {
    key: gen_tech[key][4] for key in gen_tech.keys()
}


ehr_inp["Import_prices"] = {"NatGas": 0.120, "Oil": 0.101, "Biomass": 0.100}
ehr_inp["Carbon_factors_import"] = {"NatGas": 0.228, "Oil": 0.301, "Biomass": 0.018}

ehr_inp["Conv_factor"] = {
    ("PV", "Elec"): 0.15,
    ("ST", "Heat"): 0.35,
    ("ASHP", "Heat"): 3.0,
    ("ASHP", "Elec"): -1.0,
    ("GSHP", "Heat"): 4.0,
    ("GSHP", "Elec"): -1.0,
    ("Gas_Boiler", "Heat"): 0.95,
    ("Gas_Boiler", "NatGas"): -1.0,
    ("Bio_Boiler", "Heat"): 0.9,
    ("Bio_Boiler", "Biomass"): -1.0,
    ("Oil_Boiler", "Heat"): 0.9,
    ("Oil_Boiler", "Oil"): -1.0,
    ("CHP", "Heat"): 0.6,
    ("CHP", "Elec"): 0.3,
    ("CHP", "NatGas"): -1.0,
}

ehr_inp["Minimum_part_load"] = {
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
# Storage_discharging_eff, Storage_max_cap, Lifetime_stor, Embodied_emissions_linear, Embodied_emissions_fixed
stor_tech = {
    "Thermal_storage_tank": [13, 1685, 0.25, 0.25, 0.005, 0.90, 0.90, 100, 20, 4.7, 31],
    "Battery": [235, 0, 0.25, 0.25, 0.0006, 0.90, 0.90, 60, 20, 157, 0],
}

ehr_inp["Linear_stor_costs"] = {key: stor_tech[key][0] for key in stor_tech.keys()}
ehr_inp["Fixed_stor_costs"] = {key: stor_tech[key][1] for key in stor_tech.keys()}
ehr_inp["Storage_max_charge"] = {key: stor_tech[key][2] for key in stor_tech.keys()}
ehr_inp["Storage_max_discharge"] = {key: stor_tech[key][3] for key in stor_tech.keys()}
ehr_inp["Storage_standing_losses"] = {
    key: stor_tech[key][4] for key in stor_tech.keys()
}
ehr_inp["Storage_charging_eff"] = {key: stor_tech[key][5] for key in stor_tech.keys()}
ehr_inp["Storage_discharging_eff"] = {
    key: stor_tech[key][6] for key in stor_tech.keys()
}
ehr_inp["Storage_max_cap"] = {key: stor_tech[key][7] for key in stor_tech.keys()}
ehr_inp["Lifetime_stor"] = {key: stor_tech[key][8] for key in stor_tech.keys()}
ehr_inp["Embodied_emissions_storage_tech_linear"] = {
    key: stor_tech[key][9] for key in stor_tech.keys()
}
ehr_inp["Embodied_emissions_storage_tech_fixed"] = {
    key: stor_tech[key][10] for key in stor_tech.keys()
}

ehr_inp["Storage_tech_coupling"] = {
    ("Thermal_storage_tank", "Heat"): 1.0,
    ("Battery", "Elec"): 1.0,
}

# Retrofits
# ---------
ehr_inp["Retrofit_inv_costs"] = {
    "Hempcrete01": 308731.79124000005,
    "Hempcrete025": 129547.87473000001,
    "EPS01": 130964.73248199999,
    "EPS025": 69138.17995399999,
    "Rockwool01": 140620.89826400002,
    "Rockwool025": 63749.453432,
    "Straw01": 128594.24677900002,
    "Straw025": 128493.87710000001,
    "Woodfibre01": 292696.5145,
    "Woodfibre025": 138403.17383499997,
}  # Cost of insulation
ehr_inp["Embodied_emissions_insulation"] = {
    "Hempcrete01": 4149.62085,
    "Hempcrete025": 1583.71485,
    "EPS01": 81501.810168,
    "EPS025": 31092.754008,
    "Rockwool01": 17805.892962,
    "Rockwool025": 6798.1562220000005,
    "Straw01": -93999.811553,
    "Straw025": -35888.638696999995,
    "Woodfibre01": -78037.393789000013,
    "Woodfibre025": -29758.772725000003,
}  # Embodied emissions of insulation
ehr_inp["Lifetime_retrofit"] = {
    "Hempcrete01": 40,
    "Hempcrete025": 40,
    "EPS01": 40,
    "EPS025": 40,
    "Rockwool01": 40,
    "Rockwool025": 40,
    "Straw01": 40,
    "Straw025": 40,
    "Woodfibre01": 40,
    "Woodfibre025": 40,
}  # Lifetime of the retrofit


#%% Create and solve the model
# ============================
import EnergyHubRetrofit_final as ehr

mod = ehr.EnergyHubRetrofit(ehr_inp, 1, 1, 1)  # Initialize the model
mod.create_model()  # Create the model
for i in range(Number_of_scenarios):
    if i == 0:
        mod.m.y_retrofit[ehr_inp["Retrofit_scenarios"][i]].fix(1)

    else:
        mod.m.y_retrofit[ehr_inp["Retrofit_scenarios"][i]].fix(1)
        mod.m.y_retrofit[ehr_inp["Retrofit_scenarios"][i - 1]].fix(0)

    mod.solve()  # Solve the model

# %%
