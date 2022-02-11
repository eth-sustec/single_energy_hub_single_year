# -*- coding: utf-8 -*-
"""
Single energy hub - single stage model for the optimal design of a multi-energy system including building retrofit options
Author: Georgios Mavromatidis (ETH Zurich, gmavroma@ethz.ch)
"""

import pyomo
from pyomo.core.base.initializer import Initializer
import pyomo.opt
import pyomo.environ as pe

# import pandas as pd
import numpy as np


class EnergyHubRetrofit:
    """This class implements a standard energy hub model for the optimal design and operation of distributed multi-energy systems"""

    def __init__(self, eh_input_dict, temp_res=1, optim_mode=1, num_of_pareto_points=5):
        """
        __init__ function to read in the input data and begin the model creation process

        Inputs to the function:
        -----------------------
            * eh_input_dict: dictionary that holds all the values for the model parameters
            * temp_res (default = 1): 1: typical days optimization, 2: full horizon optimization (8760 hours), 3: typical days with continuous storage state-of-charge
            * optim_mode (default = 3): 1: for cost minimization, 2: for carbon minimization, 3: for multi-objective optimization
            * num_of_pareto_points (default = 5): In case optim_mode is set to 3, then this specifies the number of Pareto points
        """

        self.inp = eh_input_dict
        self.temp_res = temp_res
        self.optim_mode = optim_mode
        if self.optim_mode == 1 or self.optim_mode == 2:
            self.num_of_pfp = 0
            print(
                "Warning: Number of Pareto front points specified is ignored. Single-objective optimization will be performed."
            )
        else:
            self.num_of_pfp = num_of_pareto_points

    def create_model(self):
        """Create the Pyomo energy hub model given the input data specified in the self.InputFile"""

        self.m = pe.ConcreteModel()

        #%% Model sets
        # ==========

        # Temporal dimensions
        # -------------------
        self.m.Days = pe.Set(
            initialize=self.inp["Days"],
            ordered=True,
            doc="The number of days considered in each year of the model | Index: d",
        )
        self.m.Time_steps = pe.Set(
            initialize=self.inp["Time_steps"],
            ordered=True,
            doc="Time steps considered in the model | Index: t",
        )
        self.m.Calendar_days = pe.Set(
            initialize=list(range(1, 365 + 1)),
            ordered=True,
            doc="Set for each calendar day of a full year | Index: cd",
        )
        # Energy carriers
        # ---------------
        self.m.Energy_carriers = pe.Set(
            initialize=self.inp["Energy_carriers"],
            doc="The set of all energy carriers considered in the model | Index : ec",
        )
        self.m.Energy_carriers_imp = pe.Set(
            initialize=self.inp["Energy_carriers_imp"],
            within=self.m.Energy_carriers,
            doc="The set of energy carriers for which imports are possible | Index : ec_imp",
        )
        self.m.Energy_carriers_exp = pe.Set(
            initialize=self.inp["Energy_carriers_exp"],
            within=self.m.Energy_carriers,
            doc="The set of energy carriers for which exports are possible | Index : ec_exp",
        )
        self.m.Energy_carriers_dem = pe.Set(
            initialize=self.inp["Energy_carriers_dem"],
            within=self.m.Energy_carriers,
            doc="The set of energy carriers for which end-user demands are defined | Index : ec_dem",
        )

        # Technologies
        # ------------
        self.m.Conversion_tech = pe.Set(
            initialize=self.inp["Conversion_tech"],
            doc="The energy conversion technologies of each energy hub candidate site | Index : conv_tech",
        )

        self.m.Solar_tech = pe.Set(
            initialize=self.inp["Solar_tech"],
            within=self.m.Conversion_tech,
            doc="Subset for solar technologies | Index : sol",
        )

        self.m.PV_tech = pe.Set(
            initialize=self.inp["PV_tech"],
            within=self.m.Solar_tech,
            doc="Subset for solar technologies | Index : pv",
        )

        self.m.Dispatchable_tech = pe.Set(
            initialize=self.inp["Dispatchable_tech"],
            within=self.m.Conversion_tech,
            doc="Subset for dispatchable/controllable technologies | Index : disp",
        )
        self.m.Storage_tech = pe.Set(
            initialize=self.inp["Storage_tech"],
            doc="The set of energy storage technologies for each energy hub candidate site | Index : stor_tech",
        )

        # Retrofitting
        # ------------
        self.m.Retrofit_scenarios = pe.Set(
            initialize=self.inp["Retrofit_scenarios"],
            doc="Retrofit scenarios considered for the building(s) connected to the energy hub",
        )

        #%% Model parameters
        # ================

        # Load parameters
        # ---------------
        self.m.Demands = pe.Param(
            self.m.Energy_carriers_dem,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            default=0,
            initialize=self.inp["Demands"],
            doc="Time-varying energy demand patterns for the energy hub",
        )

        if self.temp_res == 1 or self.temp_res == 3:
            self.m.Number_of_days = pe.Param(
                self.m.Retrofit_scenarios,
                self.m.Days,
                default=1,
                initialize=self.inp["Number_of_days"],
                doc="The number of days that each time step of typical day corresponds to",
            )
        else:
            self.m.Number_of_days = pe.Param(
                self.m.Retrofit_scenarios,
                self.m.Days,
                default=1,
                initialize=1,
                doc="Parameter equal to 1 for each time step, because full horizon optimization is performed (temp_res == 2)",
            )
        if self.temp_res == 3:
            self.m.C_to_T = pe.Param(
                self.m.Retrofit_scenarios,
                self.m.Calendar_days,
                initialize=self.inp["C_to_T"],
                within=self.m.Days,
                doc="Parameter to match each calendar day of a full year to a typical day for optimization",
            )

        # Technical parameters
        # --------------------

        self.m.Conv_factor = pe.Param(
            self.m.Conversion_tech,
            self.m.Energy_carriers,
            default=0,
            initialize=self.inp["Conv_factor"],
            doc="The conversion factors of the technologies in the energy hub",
        )
        self.m.Minimum_part_load = pe.Param(
            self.m.Dispatchable_tech,
            default=0,
            initialize=self.inp["Minimum_part_load"],
            doc="Minimum allowable part-load during the operation of dispatchable technologies",
        )
        self.m.Lifetime_tech = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp["Lifetime_tech"],
            doc="Lifetime of energy generation technologies",
        )
        self.m.Storage_tech_coupling = pe.Param(
            self.m.Storage_tech,
            self.m.Energy_carriers,
            initialize=self.inp["Storage_tech_coupling"],
            default=0,
            doc="Par",
        )
        self.m.Storage_max_charge = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Storage_max_charge"],
            doc="Maximum charging rate in % of the total capacity for the storage technologies",
        )
        self.m.Storage_max_discharge = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Storage_max_discharge"],
            doc="Maximum discharging rate in % of the total capacity for the storage technologies",
        )
        self.m.Storage_standing_losses = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Storage_standing_losses"],
            doc="Standing losses for the storage technologies",
        )
        self.m.Storage_charging_eff = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Storage_charging_eff"],
            doc="Efficiency of charging process for the storage technologies",
        )
        self.m.Storage_discharging_eff = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Storage_discharging_eff"],
            doc="Efficiency of discharging process for the storage technologies",
        )
        self.m.Storage_max_cap = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Storage_max_cap"],
            doc="Maximum allowable energy storage capacity per technology type",
        )
        self.m.Lifetime_stor = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Lifetime_stor"],
            doc="Lifetime of energy storage technologies",
        )
        self.m.Lifetime_retrofit = pe.Param(
            self.m.Retrofit_scenarios,
            initialize=self.inp["Lifetime_retrofit"],
            doc="Lifetime considered for each retrofit scenario",
        )
        self.m.Network_efficiency = pe.Param(
            self.m.Energy_carriers_dem,
            default=1,
            initialize=self.inp["Network_efficiency"],
            doc="The efficiency of the energy networks used by the energy hub",
        )
        self.m.Network_length = pe.Param(
            initialize=self.inp["Network_length"],
            doc="The length of the thermal network for the energy hub",
        )
        self.m.Network_lifetime = pe.Param(
            initialize=self.inp["Network_lifetime"],
            doc="The lifetime of the thermal network used by the energy hub",
        )

        # Cost parameters
        # ---------------
        self.m.Cost_CO2_certificates = pe.Param(
            initialize=self.inp["Cost_CO2_certificates"],
            doc="Cost of CO2 certificates",
        )

        self.m.Import_prices = pe.Param(
            self.m.Energy_carriers_imp,
            initialize=self.inp["Import_prices"],
            default=0,
            doc="Prices for importing energy carriers from the grid",
        )

        self.m.Linear_conv_costs = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp["Linear_conv_costs"],
            doc="Linear part of the investment cost (per kW or m2) for the generation technologies in the energy hub",
        )
        self.m.Fixed_conv_costs = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp["Fixed_conv_costs"],
            doc="Fixed part of the investment cost (per kW or m2) for the generation technologies in the energy hub",
        )
        self.m.Linear_stor_costs = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Linear_stor_costs"],
            doc="Linear part of the investment cost (per kWh) for the storage technologies in the energy hub",
        )
        self.m.Fixed_stor_costs = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Fixed_stor_costs"],
            doc="Fixed part of the investment cost (per kWh) for the storage technologies in the energy hub",
        )
        self.m.Network_inv_cost_per_m = pe.Param(
            initialize=self.inp["Network_inv_cost_per_m"],
            doc="Investment cost per pipe m of the thermal network of the energy hub",
        )
        self.m.Retrofit_inv_costs = pe.Param(
            self.m.Retrofit_scenarios,
            initialize=self.inp["Retrofit_inv_costs"],
            doc="Investment cost for each of the considered retrofit scenarios",
        )
        self.m.Discount_rate = pe.Param(
            initialize=self.inp["Discount_rate"],
            doc="The interest rate used for the CRF calculation",
        )

        def CRF_tech_rule(m, conv_tech):
            return (
                m.Discount_rate * (1 + m.Discount_rate) ** m.Lifetime_tech[conv_tech]
            ) / ((1 + m.Discount_rate) ** m.Lifetime_tech[conv_tech] - 1)

        self.m.CRF_tech = pe.Param(
            self.m.Conversion_tech,
            initialize=CRF_tech_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of generation technologies",
        )

        def CRF_stor_rule(m, stor_tech):
            return (
                m.Discount_rate * (1 + m.Discount_rate) ** m.Lifetime_stor[stor_tech]
            ) / ((1 + m.Discount_rate) ** m.Lifetime_stor[stor_tech] - 1)

        self.m.CRF_stor = pe.Param(
            self.m.Storage_tech,
            initialize=CRF_stor_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of storage technologies",
        )

        def CRF_network_rule(m):
            return (m.Discount_rate * (1 + m.Discount_rate) ** m.Network_lifetime) / (
                (1 + m.Discount_rate) ** m.Network_lifetime - 1
            )

        self.m.CRF_network = pe.Param(
            initialize=CRF_network_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of the networks used by the energy hub",
        )

        def CRF_retrofit_rule(m, ret):
            return (
                m.Discount_rate * (1 + m.Discount_rate) ** m.Lifetime_retrofit[ret]
            ) / ((1 + m.Discount_rate) ** m.Lifetime_retrofit[ret] - 1)

        self.m.CRF_retrofit = pe.Param(
            self.m.Retrofit_scenarios,
            initialize=CRF_retrofit_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of the considered retrofit scenarios",
        )

        # Environmental parameters
        # ------------------------
        self.m.Embodied_emissions_conversion_tech_linear = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp["Embodied_emissions_conversion_tech_linear"],
            doc="Variable embodied emissions of the conversion technologies",
        )

        self.m.Embodied_emissions_conversion_tech_fixed = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp["Embodied_emissions_conversion_tech_fixed"],
            doc="Fixed embodied emissions of the conversion technologies",
        )

        self.m.Embodied_emissions_storage_tech_linear = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Embodied_emissions_storage_tech_linear"],
            doc="Variable embodied emissions of the conversion technologies",
        )

        self.m.Embodied_emissions_storage_tech_fixed = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp["Embodied_emissions_storage_tech_fixed"],
            doc="Fixed embodied emissions of the conversion technologies",
        )

        self.m.Embodied_emissions_insulation = pe.Param(
            self.m.Retrofit_scenarios,
            initialize=self.inp["Embodied_emissions_insulation"],
            doc="Embodied emissions of the insulation materials, positive or negative",
        )

        self.m.Carbon_benefit_CO2_certificates = pe.Param(
            initialize=self.inp["Carbon_benefit_CO2_certificates"],
            doc="CO2 benefits of the certificates",
        )

        self.m.Carbon_factors_import = pe.Param(
            self.m.Energy_carriers_imp,
            initialize=self.inp["Carbon_factors_import"],
            doc="Energy carrier CO2 emission factors",
        )

        self.m.Grid_intensity = pe.Param(
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            initialize=self.inp["Grid_intensity"],
            doc="Electricity CO2 emission factors",
        )

        self.m.Elec_import_prices = pe.Param(
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            initialize=self.inp["Elec_import_prices"],
            doc="Hourly prices of imported electricity",
        )

        self.m.Elec_export_prices = pe.Param(
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            initialize=self.inp["Elec_export_prices"],
            doc="Hourly prices of iexported electricity",
        )

        self.m.epsilon = pe.Param(
            initialize=0,
            mutable=True,
            doc="Epsilon value used for the multi-objective epsilon-constrained optimization",
        )

        # Misc parameters
        # ---------------
        self.m.Roof_area = pe.Param(
            initialize=self.inp["Roof_area"],
            doc="Available roof area for the installation of solar technologies",
        )
        self.m.P_solar = pe.Param(
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            initialize=self.inp["P_solar"],
            doc="Incoming solar radiation patterns (kWh/m2) for solar technologies",
        )
        self.m.BigM = pe.Param(default=10 ** 6, doc="Big M: Sufficiently large value")
        self.m.BigM_2 = pe.Param(default=10 ** 4, doc="Big M: Sufficiently large value")
        self.m.min_install = pe.Param(
            default=10, doc="Minimum installable value as 20*y_conv"
        )

        #%% Model variables
        # ===============

        # Generation technologies
        # -----------------------
        self.m.P_import = pe.Var(
            self.m.Energy_carriers_imp,
            self.m.Days,
            self.m.Time_steps,
            within=pe.NonNegativeReals,
            doc="Imported energy flows for the energy hub at each time step",
        )
        self.m.P_conv = pe.Var(
            self.m.Conversion_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.NonNegativeReals,
            doc="The input energy carrier stream at each energy conversion technology of the energy hub at each time step",
        )
        self.m.P_export = pe.Var(
            self.m.Energy_carriers_exp,
            self.m.Days,
            self.m.Time_steps,
            within=pe.NonNegativeReals,
            doc="Exported energy by the energy hub at each time step",
        )
        self.m.y_on = pe.Var(
            self.m.Dispatchable_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.Binary,
            doc="Binary variable indicating the on (=1) or off (=0) state of a dispatchable technology",
        )
        self.m.y_conv = pe.Var(
            self.m.Conversion_tech,
            within=pe.Binary,
            initialize=0,
            doc="Binary variable denoting the installation (=1) of energy conversion technology",
        )
        self.m.Conv_cap = pe.Var(
            self.m.Conversion_tech,
            within=pe.NonNegativeReals,
            doc="Installed capacity for energy conversion technology",
        )

        # Storage technologies
        # --------------------
        self.m.Qin = pe.Var(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.NonNegativeReals,
            doc="Storage charging rate",
        )
        self.m.Qout = pe.Var(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.NonNegativeReals,
            doc="Storage discharging rate",
        )
        if self.temp_res != 3:
            self.m.SoC = pe.Var(
                self.m.Storage_tech,
                self.m.Days,
                self.m.Time_steps,
                within=pe.NonNegativeReals,
                doc="Storage state of charge",
            )
        else:
            self.m.SoC = pe.Var(
                self.m.Storage_tech,
                self.m.Calendar_days,
                self.m.Time_steps,
                within=pe.NonNegativeReals,
                doc="Storage state of charge",
            )
        self.m.y_stor = pe.Var(
            self.m.Storage_tech,
            within=pe.Binary,
            doc="Binary variable denoting the installation (=1) of energy storage technology",
        )
        self.m.Storage_cap = pe.Var(
            self.m.Storage_tech,
            within=pe.NonNegativeReals,
            doc="Installed capacity for energy storage technology",
        )

        # Retrofit scenarios
        # ------------------
        self.m.y_retrofit = pe.Var(
            self.m.Retrofit_scenarios,
            within=pe.Binary,
            doc="Binary variable denoting the retrofit state to be selected",
        )

        # CO2 certificates
        # -----------------------------
        self.m.Number_CO2_certificates = pe.Var(
            # self.m.Retrofit_scenarios,
            within=pe.NonNegativeIntegers,
            doc="The number of CO2 certificates bought for each ret scenario",
        )
        # Solar self consumption
        # -----------------------------
        self.m.Solar_self_consumption = pe.Var(
            self.m.PV_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.NonNegativeReals,
            doc="The part of the demand that is met with PV generation",
        )

        # Objective function components
        # -----------------------------
        self.m.Import_cost = pe.Var(
            within=pe.NonNegativeReals,
            doc="The operating cost for the consumption of energy carriers",
        )
        self.m.Export_profit = pe.Var(
            within=pe.NonNegativeReals, doc="Total income due to exported electricity"
        )
        self.m.Investment_cost = pe.Var(
            within=pe.NonNegativeReals,
            doc="Investment cost of all energy technologies in the energy hub",
        )
        self.m.Total_cost = pe.Var(
            within=pe.NonNegativeReals,
            doc="Total cost for the investment and the operation of the energy hub",
        )
        self.m.Total_carbon = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="Total carbon emissions due to the operation of the energy hub",
        )

        # Added variables for sanity check

        self.m.Embodied_conv = pe.Var(
            self.m.Conversion_tech,
            initialize=0,
            within=pe.NonNegativeReals,
            doc="Embodied emissions of conversion technologies",
        )

        self.m.Embodied_stor = pe.Var(
            self.m.Storage_tech,
            initialize=0,
            within=pe.NonNegativeReals,
            doc="Embodied emissions of storage technologies",
        )

        self.m.Embodied_insulation = pe.Var(
            initialize=0, doc="Embodied emissions of insulation materials",
        )

        self.m.Cost_conv = pe.Var(
            self.m.Conversion_tech,
            within=pe.NonNegativeReals,
            initialize=0,
            doc="Embodied emissions of conversion technologies",
        )

        self.m.Cost_stor = pe.Var(
            self.m.Storage_tech,
            within=pe.NonNegativeReals,
            initialize=0,
            doc="Embodied emissions of storage technologies",
        )

        self.m.Cost_insulation = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="Embodied emissions of insulation materials",
        )

        self.m.Elec_import_cost = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="The operating cost for the consumption of electricity",
        )

        self.m.Others_import_cost = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="The operating cost for the consumption of energy carriers that are not electricity",
        )

        self.m.Elec_import_emissions = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="The operational emissions for the consumption of electricity",
        )

        self.m.Others_import_emissions = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="The operational emissions for the consumption of energy carriers that are not electricity",
        )

        self.m.Elec_export_emissions = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="The operational emissions for the export of electricity, accounted as negative in total balance",
        )

        self.m.Elec_export_cost = pe.Var(
            within=pe.NonNegativeReals,
            initialize=0,
            doc="The profit for the export of electricity",
        )

        # Bilinear term reformulations
        # ----------------------------
        # These variables need to be defined in order to linearize the products of binary and continuous variables in the model.
        # These are then in turn caused by the fact that when typical days are used,
        # the "Number of typical days" parameter and the "P_solar" parameter differ per retrofit scenario

        self.m.z1 = pe.Var(
            self.m.Energy_carriers_imp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            doc="Variable to represent the product: P_import[ec_imp, d, t] * y_retrofit[ret] and avoid non-linearity",
        )
        self.m.z2 = pe.Var(
            self.m.Energy_carriers_exp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            doc="Variable to represent the product: P_export[ec_exp, d, t] * y_retrofit[ret]",
        )
        self.m.z3 = pe.Var(
            self.m.Solar_tech,
            self.m.Retrofit_scenarios,
            doc="Variable to represent the product: Conv_cap[sol] * y_retrofit[ret]",
        )
        self.m.z4 = pe.Var(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            doc="Variable to represent the product Qin[stor_tech, d, t] * y_retrofit[ret] | Useful only when temp_res = 3 and the C_to_T parameter is used",
        )
        self.m.z5 = pe.Var(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            doc="Variable to represent the product Qout[stor_tech, d, t] * y_retrofit[ret] | Useful only when temp_res = 3 and the C_to_T parameter is used",
        )
        # Added variables to avoid simultaneous charging and discharging
        # ----------------------

        self.m.er = pe.Param(initialize=1e-6, doc="",)
        self.m.QIin = pe.Var(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.Binary,
            doc="",
        )
        self.m.QIout = pe.Var(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.Binary,
            doc="",
        )

        self.m.z_scd = pe.Var(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            initialize=0,
            within=pe.Binary,
            doc="",
        )

        #%% Model constraints
        # =================

        # Energy demand balances
        # ----------------------

        def Load_balance_rule(m, ec, d, t):
            conv_tech_tmp = [tech for tech in m.Conversion_tech if tech != "PV"]
            return (m.P_import[ec, d, t] if ec in m.Energy_carriers_imp else 0) + sum(
                m.P_conv[conv_tech, d, t] * m.Conv_factor[conv_tech, ec]
                for conv_tech in conv_tech_tmp  # without PV, which is added separately as self_consumption
            ) + sum(
                m.Storage_tech_coupling[stor_tech, ec]
                * (m.Qout[stor_tech, d, t] - m.Qin[stor_tech, d, t])
                for stor_tech in m.Storage_tech
            ) + sum(
                m.Solar_self_consumption[sol, d, t]
                for sol in m.PV_tech
                if m.Conv_factor[sol, ec] > 0
            ) == sum(
                m.y_retrofit[ret] * m.Demands[ec, ret, d, t]
                for ret in m.Retrofit_scenarios
                if ec in m.Energy_carriers_dem
            ) / (
                m.Network_efficiency[ec] if ec in m.Energy_carriers_dem else 1
            )

        self.m.Load_balance = pe.Constraint(
            self.m.Energy_carriers,
            self.m.Days,
            self.m.Time_steps,
            rule=Load_balance_rule,
            doc="Energy balance for the energy hub including solar self_consumption, conversion, storage, losses, exchange and export flows",
        )

        # Generation constraints
        # ----------------------
        def Capacity_constraint_rule(m, disp, ec, d, t):
            if m.Conv_factor[disp, ec] > 0:
                return (
                    m.P_conv[disp, d, t] * m.Conv_factor[disp, ec] <= m.Conv_cap[disp]
                )
            else:
                return pe.Constraint.Skip

        self.m.Capacity_constraint = pe.Constraint(
            self.m.Dispatchable_tech,
            self.m.Energy_carriers,
            self.m.Days,
            self.m.Time_steps,
            rule=Capacity_constraint_rule,
            doc="Constraint preventing capacity violation for the generation technologies of the energy hub",
        )

        def Min_installable_capacity_rule(m, conv_tech):
            return m.Conv_cap[conv_tech] >= m.min_install * m.y_conv[conv_tech]

        self.m.Min_installable_capacity = pe.Constraint(
            self.m.Conversion_tech,
            rule=Min_installable_capacity_rule,
            doc="Minimum capacity that needs to be installed if a conversion technology is used",
        )

        def Min_installable_capacity_rule_2(m, stor_tech):
            return m.Storage_cap[stor_tech] >= m.min_install * m.y_stor[stor_tech]

        self.m.Min_installable_capacity_2 = pe.Constraint(
            self.m.Storage_tech,
            rule=Min_installable_capacity_rule_2,
            doc="Minimum capacity that needs to be installed if a storage technology is used",
        )

        def Solar_input_rule_initial(m, sol, d, t):
            # return m.P_conv[sol, d, t] == m.P_solar[ret, d, t] * m.Conv_cap[sol] * y_retrofit[ret]
            return m.P_conv[sol, d, t] == sum(
                m.P_solar[ret, d, t] * m.z3[sol, ret] for ret in m.Retrofit_scenarios
            )

        self.m.Solar_input_inital = pe.Constraint(
            self.m.Solar_tech,
            self.m.Days,
            self.m.Time_steps,
            rule=Solar_input_rule_initial,
            doc="Constraint connecting the solar radiation per m2 with the area of solar PV technologies",
        )

        def Solar_self_consumption_rule(m, ec_exp, d, t):
            # solar_tmp = [tech for tech in m.Conversion_tech if tech == "PV"]
            return (
                m.P_export["Elec", d, t]
                == m.P_conv["PV", d, t] - m.Solar_self_consumption["PV", d, t]
            )

        self.m.Solar_input = pe.Constraint(
            self.m.Energy_carriers_exp,
            self.m.Days,
            self.m.Time_steps,
            rule=Solar_self_consumption_rule,
            doc="Constraint describing that solar generated electricity can be exported",
        )

        def Minimum_part_load_constr_rule1(m, disp, ec, d, t):
            return (
                m.P_conv[disp, d, t] * m.Conv_factor[disp, ec]
                <= m.BigM * m.y_on[disp, d, t]
            )

        def Minimum_part_load_constr_rule2(m, disp, ec, d, t):
            return (
                m.P_conv[disp, d, t] * m.Conv_factor[disp, ec]
                + m.BigM * (1 - m.y_on[disp, d, t])
                >= m.Minimum_part_load[disp] * m.Conv_cap[disp]
            )

        self.m.Mininum_part_rule_constr1 = pe.Constraint(
            (
                (disp, ec, d, t)
                for disp in self.m.Dispatchable_tech
                for ec in self.m.Energy_carriers
                for d in self.m.Days
                for t in self.m.Time_steps
                if self.m.Conv_factor[disp, ec] > 0
            ),
            rule=Minimum_part_load_constr_rule1,
            doc="Constraint enforcing a minimum load during the operation of a dispatchable energy technology",
        )

        self.m.Mininum_part_rule_constr2 = pe.Constraint(
            (
                (disp, ec, d, t)
                for disp in self.m.Dispatchable_tech
                for ec in self.m.Energy_carriers
                for d in self.m.Days
                for t in self.m.Time_steps
                if self.m.Conv_factor[disp, ec] > 0
            ),
            rule=Minimum_part_load_constr_rule2,
            doc="Constraint enforcing a minimum load during the operation of a dispatchable energy technology",
        )

        def Fixed_cost_constr_rule(m, conv_tech):
            return m.Conv_cap[conv_tech] <= m.BigM_2 * m.y_conv[conv_tech]

        self.m.Fixed_cost_constr = pe.Constraint(
            self.m.Conversion_tech,
            rule=Fixed_cost_constr_rule,
            doc="Constraint for the formulation of the fixed cost in the objective function",
        )

        def Roof_area_non_violation_rule(m):
            return sum(m.Conv_cap[sol] for sol in m.Solar_tech) <= m.Roof_area

        self.m.Roof_area_non_violation = pe.Constraint(
            rule=Roof_area_non_violation_rule,
            doc="Non violation of the maximum roof area for solar installations",
        )

        # CO2 certificates constraint
        # -------------------
        def Number_CO2_certificates_rule(m):
            return m.Number_CO2_certificates <= 150

        self.m.Number_CO2_certificates_rule = pe.Constraint(
            rule=Number_CO2_certificates_rule,
            doc="Definition of a limit to the number of purchasable CO2 certificates",
        )

        # Storage constraints
        # -------------------
        def Storage_balance_rule(m, stor_tech, d, t):
            if self.temp_res == 1:
                if t != 1:
                    return (
                        m.SoC[stor_tech, d, t]
                        == (1 - m.Storage_standing_losses[stor_tech])
                        * m.SoC[stor_tech, d, t - 1]
                        + m.Storage_charging_eff[stor_tech] * m.Qin[stor_tech, d, t]
                        - (1 / m.Storage_discharging_eff[stor_tech])
                        * m.Qout[stor_tech, d, t]
                    )
                else:
                    return (
                        m.SoC[stor_tech, d, t]
                        == (1 - m.Storage_standing_losses[stor_tech])
                        * m.SoC[stor_tech, d, t + max(m.Time_steps) - 1]
                        + m.Storage_charging_eff[stor_tech] * m.Qin[stor_tech, d, t]
                        - (1 / m.Storage_discharging_eff[stor_tech])
                        * m.Qout[stor_tech, d, t]
                    )
            elif self.temp_res == 2:
                if t != 1:
                    return (
                        m.SoC[stor_tech, d, t]
                        == (1 - m.Storage_standing_losses[stor_tech])
                        * m.SoC[stor_tech, d, t - 1]
                        + m.Storage_charging_eff[stor_tech] * m.Qin[stor_tech, d, t]
                        - (1 / m.Storage_discharging_eff[stor_tech])
                        * m.Qout[stor_tech, d, t]
                    )
                else:
                    if d != 1:
                        return (
                            m.SoC[stor_tech, d, t]
                            == (1 - m.Storage_standing_losses[stor_tech])
                            * m.SoC[stor_tech, d - 1, t + max(m.Time_steps) - 1]
                            + m.Storage_charging_eff[stor_tech] * m.Qin[stor_tech, d, t]
                            - (1 / m.Storage_discharging_eff[stor_tech])
                            * m.Qout[stor_tech, d, t]
                        )
                    else:
                        return (
                            m.SoC[stor_tech, d, t]
                            == (1 - m.Storage_standing_losses[stor_tech])
                            * m.SoC[stor_tech, d + 364, t + max(m.Time_steps) - 1]
                            + m.Storage_charging_eff[stor_tech] * m.Qin[stor_tech, d, t]
                            - (1 / m.Storage_discharging_eff[stor_tech])
                            * m.Qout[stor_tech, d, t]
                        )
            elif self.temp_res == 3:
                if t != 1:
                    return m.SoC[stor_tech, d, t] == (
                        1 - m.Storage_standing_losses[stor_tech]
                    ) * m.SoC[stor_tech, d, t - 1] + m.Storage_charging_eff[
                        stor_tech
                    ] * sum(
                        # m.Qin[stor_tech, m.C_to_T[ret, d], t] * m.y_retrofit[ret]
                        m.z4[stor_tech, ret, m.C_to_T[ret, d], t]
                        for ret in m.Retrofit_scenarios
                    ) - (
                        1 / m.Storage_discharging_eff[stor_tech]
                    ) * sum(
                        # m.Qout[stor_tech, m.C_to_T[ret, d], t] * m.y_retrofit[ret]
                        m.z5[stor_tech, ret, m.C_to_T[ret, d], t]
                        for ret in m.Retrofit_scenarios
                    )
                else:
                    if d != 1:
                        return m.SoC[stor_tech, d, t] == (
                            1 - m.Storage_standing_losses[stor_tech]
                        ) * m.SoC[
                            stor_tech, d - 1, t + max(m.Time_steps) - 1
                        ] + m.Storage_charging_eff[
                            stor_tech
                        ] * sum(
                            # m.Qin[stor_tech, m.C_to_T[ret, d], t] * m.y_retrofit[ret]
                            m.z4[stor_tech, ret, m.C_to_T[ret, d], t]
                            for ret in m.Retrofit_scenarios
                        ) - (
                            1 / m.Storage_discharging_eff[stor_tech]
                        ) * sum(
                            # m.Qout[stor_tech, m.C_to_T[ret, d], t] * m.y_retrofit[ret]
                            m.z5[stor_tech, ret, m.C_to_T[ret, d], t]
                            for ret in m.Retrofit_scenarios
                        )
                    else:
                        return m.SoC[stor_tech, d, t] == (
                            1 - m.Storage_standing_losses[stor_tech]
                        ) * m.SoC[
                            stor_tech,
                            d + max(m.Calendar_days) - 1,
                            t + max(m.Time_steps) - 1,
                        ] + m.Storage_charging_eff[
                            stor_tech
                        ] * sum(
                            # m.Qin[stor_tech, m.C_to_T[ret, d], t] * m.y_retrofit[ret]
                            m.z4[stor_tech, ret, m.C_to_T[ret, d], t]
                            for ret in m.Retrofit_scenarios
                        ) - (
                            1 / m.Storage_discharging_eff[stor_tech]
                        ) * sum(
                            # m.Qout[stor_tech, m.C_to_T[ret, d], t] * m.y_retrofit[ret]
                            m.z5[stor_tech, ret, m.C_to_T[ret, d], t]
                            for ret in m.Retrofit_scenarios
                        )

        if self.temp_res == 1 or self.temp_res == 2:
            self.m.Storage_balance = pe.Constraint(
                self.m.Storage_tech,
                self.m.Days,
                self.m.Time_steps,
                rule=Storage_balance_rule,
                doc="Energy balance for the storage modules considering incoming and outgoing energy flows",
            )
        elif self.temp_res == 3:
            self.m.Storage_balance = pe.Constraint(
                self.m.Storage_tech,
                self.m.Calendar_days,
                self.m.Time_steps,
                rule=Storage_balance_rule,
                doc="Energy balance for the storage modules considering incoming and outgoing energy flows",
            )

        def Storage_charg_rate_constr_rule(m, stor_tech, d, t):
            return (
                m.Qin[stor_tech, d, t]
                <= m.Storage_max_charge[stor_tech] * m.Storage_cap[stor_tech]
            )

        self.m.Storage_charg_rate_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            rule=Storage_charg_rate_constr_rule,
            doc="Constraint for the maximum allowable charging rate of the storage technologies",
        )

        def Storage_discharg_rate_constr_rule(m, stor_tech, d, t):
            return (
                m.Qout[stor_tech, d, t]
                <= m.Storage_max_charge[stor_tech] * m.Storage_cap[stor_tech]
            )

        self.m.Storage_discharg_rate_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            rule=Storage_discharg_rate_constr_rule,
            doc="Constraint for the maximum allowable discharging rate of the storage technologies",
        )

        def Storage_cap_constr_rule(m, stor_tech, d, t):
            return m.SoC[stor_tech, d, t] <= m.Storage_cap[stor_tech]

        if self.temp_res == 1 or self.temp_res == 2:
            self.m.Storage_cap_constr = pe.Constraint(
                self.m.Storage_tech,
                self.m.Days,
                self.m.Time_steps,
                rule=Storage_cap_constr_rule,
                doc="Constraint for non-violation of the capacity of the storage",
            )
        elif self.temp_res == 3:
            self.m.Storage_cap_constr = pe.Constraint(
                self.m.Storage_tech,
                self.m.Calendar_days,
                self.m.Time_steps,
                rule=Storage_cap_constr_rule,
                doc="Constraint for non-violation of the capacity of the storage",
            )

        def Max_allowable_storage_cap_rule(m, stor_tech):
            return m.Storage_cap[stor_tech] <= m.Storage_max_cap[stor_tech]

        self.m.Max_allowable_storage_cap = pe.Constraint(
            self.m.Storage_tech,
            rule=Max_allowable_storage_cap_rule,
            doc="Constraint enforcing the maximum allowable storage capacity per type of storage technology",
        )

        def Fixed_cost_storage_rule(m, stor_tech):
            return m.Storage_cap[stor_tech] <= m.BigM * m.y_stor[stor_tech]

        self.m.Fixed_cost_storage = pe.Constraint(
            self.m.Storage_tech,
            rule=Fixed_cost_storage_rule,
            doc="Constraint for the formulation of the fixed cost in the objective function",
        )

        # Added constraints to avoid simultaneous charging and discharging (scd)
        # --------------------
        def scd1_rule(m, stor_tech, d, t):
            return m.QIin[stor_tech, d, t] >= m.Qin[stor_tech, d, t] / 100

        self.m.scd1_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd1_rule, doc="",
        )

        def scd2_rule(m, stor_tech, d, t):
            return m.QIout[stor_tech, d, t] >= m.Qout[stor_tech, d, t] / 100

        self.m.scd2_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd2_rule, doc="",
        )

        def scd3_rule(m, stor_tech, d, t):
            return m.QIin[stor_tech, d, t] <= 1 - m.er + m.Qin[stor_tech, d, t] / 100

        self.m.scd3_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd3_rule, doc="",
        )

        def scd4_rule(m, stor_tech, d, t):
            return m.QIout[stor_tech, d, t] <= 1 - m.er + m.Qout[stor_tech, d, t] / 100

        self.m.scd4_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd4_rule, doc="",
        )

        def scd5_rule(m, stor_tech, d, t):
            return m.z_scd[stor_tech, d, t] >= 1 - m.QIin[stor_tech, d, t]

        self.m.scd5_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd5_rule, doc="",
        )

        def scd6_rule(m, stor_tech, d, t):
            return m.z_scd[stor_tech, d, t] >= 1 - m.QIout[stor_tech, d, t]

        self.m.scd6_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd6_rule, doc="",
        )

        def scd7_rule(m, stor_tech, d, t):
            return (
                m.z_scd[stor_tech, d, t]
                <= 2 - m.QIin[stor_tech, d, t] - m.QIout[stor_tech, d, t]
            )

        self.m.scd7_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd7_rule, doc="",
        )

        def scd8_rule(m, stor_tech, d, t):
            return m.z_scd[stor_tech, d, t] == 1

        self.m.scd8_constr = pe.Constraint(
            self.m.Storage_tech, self.m.Days, self.m.Time_steps, rule=scd8_rule, doc="",
        )

        # Retrofit constraints
        # --------------------
        def One_retrofit_state_rule(m):
            return sum(m.y_retrofit[ret] for ret in m.Retrofit_scenarios) == 1

        self.m.One_retrofit_state_def = pe.Constraint(
            rule=One_retrofit_state_rule,
            doc="Constraint to impose that one retrofit state out of all possible must be selected",
        )

        # Added constraints for sanity check
        # ----------------

        def Embodied_insulation_rule(m):
            return m.Embodied_insulation == sum(
                m.Embodied_emissions_insulation[ret]
                * m.y_retrofit[ret]
                / m.Lifetime_retrofit[ret]
                for ret in m.Retrofit_scenarios
            )

        self.m.Embodied_insulation_constr = pe.Constraint(
            rule=Embodied_insulation_rule,
            doc="Calculation of embodied emissions of insulation material",
        )

        def Cost_insulation_rule(m):
            return m.Cost_insulation == sum(
                m.y_retrofit[ret] * m.Retrofit_inv_costs[ret] * m.CRF_retrofit[ret]
                for ret in m.Retrofit_scenarios
            )

        self.m.Cost_insulation_constr = pe.Constraint(
            rule=Cost_insulation_rule, doc="Calculation of cost of insulation material",
        )

        def Embodied_conv_rule(m, conv_tech):
            return (
                m.Embodied_conv[conv_tech]
                == m.y_conv[conv_tech]
                * m.Embodied_emissions_conversion_tech_fixed[conv_tech]
                + m.Embodied_emissions_conversion_tech_linear[conv_tech]
                * m.Conv_cap[conv_tech]
            )

        self.m.Embodied_conv_constr = pe.Constraint(
            self.m.Conversion_tech,
            rule=Embodied_conv_rule,
            doc="Calculation of embodied emissions of conversion technologies",
        )

        def Embodied_stor_rule(m, stor_tech):
            return (
                m.Embodied_stor[stor_tech]
                == m.y_stor[stor_tech]
                * m.Embodied_emissions_storage_tech_fixed[stor_tech]
                + m.Embodied_emissions_storage_tech_linear[stor_tech]
                * m.Storage_cap[stor_tech]
            )

        self.m.Embodied_stor_constr = pe.Constraint(
            self.m.Storage_tech,
            rule=Embodied_stor_rule,
            doc="Calculation of embodied emissions of storage technologies",
        )

        def Cost_conv_rule(m, conv_tech):
            return (
                m.Cost_conv[conv_tech]
                == (
                    m.Fixed_conv_costs[conv_tech] * m.y_conv[conv_tech]
                    + m.Linear_conv_costs[conv_tech] * m.Conv_cap[conv_tech]
                )
                * m.CRF_tech[conv_tech]
            )

        self.m.Cost_conv_constr = pe.Constraint(
            self.m.Conversion_tech,
            rule=Cost_conv_rule,
            doc="Calculation of cost of conversion technologies",
        )

        def Cost_stor_rule(m, stor_tech):
            return (
                m.Cost_stor[stor_tech]
                == (
                    m.Fixed_stor_costs[stor_tech] * m.y_stor[stor_tech]
                    + m.Linear_stor_costs[stor_tech] * m.Storage_cap[stor_tech]
                )
                * m.CRF_stor[stor_tech]
            )

        self.m.Cost_stor_constr = pe.Constraint(
            self.m.Storage_tech,
            rule=Cost_stor_rule,
            doc="Calculation of cost of storage technologies",
        )

        def Elec_import_cost_rule(m):
            return m.Elec_import_cost == sum(
                m.Elec_import_prices[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp == "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        def Others_import_cost_rule(m):
            return m.Others_import_cost == sum(
                m.Import_prices[ec_imp]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp != "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Elec_import_cost_cstr = pe.Constraint(
            rule=Elec_import_cost_rule,
            doc="Definition of the electricity operating cost",
        )

        self.m.Others_import_cost_cstr = pe.Constraint(
            rule=Others_import_cost_rule,
            doc="Definition of the energy carriers operating cost that are not electricity",
        )

        def Elec_export_cost_rule(m):
            return m.Elec_export_cost == sum(
                m.Elec_export_prices[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z2[ec_exp, ret, d, t]
                for ec_exp in m.Energy_carriers_exp
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Elec_export_cost_cstr = pe.Constraint(
            rule=Elec_export_cost_rule,
            doc="Definition of the electricity export profit",
        )

        def Elec_import_emissions_rule(m):
            return m.Elec_import_emissions == sum(
                m.Grid_intensity[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp == "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Elec_import_emissions_cstr = pe.Constraint(
            rule=Elec_import_emissions_rule,
            doc="Definition of the electricity operating emissions",
        )

        def Others_import_emissions_rule(m):
            return m.Others_import_emissions == sum(
                # m.Carbon_factors_import[ec_imp] * m.P_import[ec_imp, d, t] * m.Number_of_days[ret, d] * m.y_retrofit[ret]
                m.Carbon_factors_import[ec_imp]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp != "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Others_import_emissions_cstr = pe.Constraint(
            rule=Others_import_emissions_rule,
            doc="Definition of the energy carriers operating emissions that are not electricity",
        )

        def Elec_export_emissions_rule(m):
            return m.Elec_export_emissions == sum(
                m.Grid_intensity[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z2[ec_exp, ret, d, t]
                for ec_exp in m.Energy_carriers_exp
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Elec_export_emissions_cstr = pe.Constraint(
            rule=Elec_export_emissions_rule,
            doc="Definition of the electricity export emissiosn",
        )

        # Objective function definitions
        # ------------------------------
        def Import_cost_rule(m):
            return m.Import_cost == sum(
                # m.Import_prices[ec_imp] * m.P_import[ec_imp, d, t] * m.Number_of_days[ret, d] * m.y_retrofit[ret]
                m.Elec_import_prices[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp == "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            ) + sum(
                # m.Import_prices[ec_imp] * m.P_import[ec_imp, d, t] * m.Number_of_days[ret, d] * m.y_retrofit[ret]
                m.Import_prices[ec_imp]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp != "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Import_cost_def = pe.Constraint(
            rule=Import_cost_rule,
            doc="Definition of the operating cost component of the total energy system cost",
        )

        def Export_profit_rule(m):
            return m.Export_profit == sum(
                # m.Export_prices[ec_imp] * m.P_export[ec_imp, d, t] * m.Number_of_days[ret, d] * m.y_retrofit[ret]
                m.Elec_export_prices[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z2[ec_exp, ret, d, t]
                for ec_exp in m.Energy_carriers_exp
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Export_profit_def = pe.Constraint(
            rule=Export_profit_rule,
            doc="Definition of the income due to electricity exports component of the total energy system cost",
        )

        def Investment_cost_rule(m):
            return m.Investment_cost == sum(
                (
                    m.Fixed_conv_costs[conv_tech] * m.y_conv[conv_tech]
                    + m.Linear_conv_costs[conv_tech] * m.Conv_cap[conv_tech]
                )
                * m.CRF_tech[conv_tech]
                for conv_tech in m.Conversion_tech
            ) + sum(
                (
                    m.Fixed_stor_costs[stor_tech] * m.y_stor[stor_tech]
                    + m.Linear_stor_costs[stor_tech] * m.Storage_cap[stor_tech]
                )
                * m.CRF_stor[stor_tech]
                for stor_tech in m.Storage_tech
            ) + m.Network_inv_cost_per_m * m.Network_length * m.CRF_network + sum(
                m.y_retrofit[ret] * m.Retrofit_inv_costs[ret] * m.CRF_retrofit[ret]
                for ret in m.Retrofit_scenarios
            )

        self.m.Investment_cost_def = pe.Constraint(
            rule=Investment_cost_rule,
            doc="Definition of the investment cost component of the total energy system cost",
        )

        def Total_cost_rule(m):
            return (
                m.Total_cost
                == m.Investment_cost
                + m.Import_cost
                - m.Export_profit
                + m.Number_CO2_certificates * m.Cost_CO2_certificates
            )

        self.m.Total_cost_def = pe.Constraint(
            rule=Total_cost_rule,
            doc="Definition of the total cost model objective function",
        )

        def Total_carbon_rule(m):
            return m.Total_carbon == sum(
                m.Grid_intensity[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp == "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            ) + sum(
                # m.Carbon_factors_import[ec_imp] * m.P_import[ec_imp, d, t] * m.Number_of_days[ret, d] * m.y_retrofit[ret]
                m.Carbon_factors_import[ec_imp]
                * m.Number_of_days[ret, d]
                * m.z1[ec_imp, ret, d, t]
                for ec_imp in m.Energy_carriers_imp
                if ec_imp != "Elec"
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            ) - sum(
                # m.Carbon_factors_import[ec_imp] * m.P_export[ec_exp, d, t] * m.Number_of_days[ret, d] * m.y_retrofit[ret]
                m.Grid_intensity[ret, d, t]
                * m.Number_of_days[ret, d]
                * m.z2[ec_exp, ret, d, t]
                for ec_exp in m.Energy_carriers_exp
                for ret in m.Retrofit_scenarios
                for d in m.Days
                for t in m.Time_steps
            ) - m.Number_CO2_certificates * m.Carbon_benefit_CO2_certificates + sum(
                m.Embodied_emissions_insulation[ret]
                / m.Lifetime_retrofit[ret]
                * m.y_retrofit[ret]
                for ret in m.Retrofit_scenarios
            ) + sum(
                (
                    m.Embodied_emissions_conversion_tech_fixed[conv_tech]
                    * m.y_conv[conv_tech]
                    + m.Embodied_emissions_conversion_tech_linear[conv_tech]
                    * m.Conv_cap[conv_tech]
                )
                / m.Lifetime_tech[conv_tech]
                for conv_tech in m.Conversion_tech
            ) + sum(
                (
                    m.Embodied_emissions_storage_tech_fixed[stor_tech]
                    * m.y_stor[stor_tech]
                    + m.Embodied_emissions_storage_tech_linear[stor_tech]
                    * m.Storage_cap[stor_tech]
                )
                / m.Lifetime_stor[stor_tech]
                for stor_tech in m.Storage_tech
            )

        self.m.Total_carbon_def = pe.Constraint(
            rule=Total_carbon_rule,
            doc="Definition of the total carbon emissions model objective function",
        )

        # Carbon constraint
        # -----------------

        def Carbon_constraint_rule(m):
            return m.Total_carbon <= m.epsilon

        self.m.Carbon_constraint = pe.Constraint(
            rule=Carbon_constraint_rule,
            doc="Constraint setting an upper limit to the total carbon emissions of the system",
        )

        # Bilinear terms - auxiliary definitions
        # --------------------------------------

        def z1_rule_1(m, ec_imp, ret, d, t):
            return m.z1[ec_imp, ret, d, t] >= 0

        self.m.z1_rule_1_constr = pe.Constraint(
            self.m.Energy_carriers_imp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z1_rule_1,
            doc="Auxiliary constraint for variable z1",
        )

        def z1_rule_2(m, ec_imp, ret, d, t):
            return m.z1[ec_imp, ret, d, t] <= m.BigM * m.y_retrofit[ret]

        self.m.z1_rule_2_constr = pe.Constraint(
            self.m.Energy_carriers_imp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z1_rule_2,
            doc="Auxiliary constraint for variable z1",
        )

        def z1_rule_3(m, ec_imp, ret, d, t):
            return m.P_import[ec_imp, d, t] - m.z1[ec_imp, ret, d, t] >= 0

        self.m.z1_rule_3_constr = pe.Constraint(
            self.m.Energy_carriers_imp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z1_rule_3,
            doc="Auxiliary constraint for variable z1",
        )

        def z1_rule_4(m, ec_imp, ret, d, t):
            return m.P_import[ec_imp, d, t] - m.z1[ec_imp, ret, d, t] <= m.BigM * (
                1 - m.y_retrofit[ret]
            )

        self.m.z1_rule_4_constr = pe.Constraint(
            self.m.Energy_carriers_imp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z1_rule_4,
            doc="Auxiliary constraint for variable z1",
        )

        def z2_rule_1(m, ec_exp, ret, d, t):
            return m.z2[ec_exp, ret, d, t] >= 0

        self.m.z2_rule_1_constr = pe.Constraint(
            self.m.Energy_carriers_exp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z2_rule_1,
            doc="Auxiliary constraint for variable z2",
        )

        def z2_rule_2(m, ec_exp, ret, d, t):
            return m.z2[ec_exp, ret, d, t] <= m.BigM * m.y_retrofit[ret]

        self.m.z2_rule_2_constr = pe.Constraint(
            self.m.Energy_carriers_exp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z2_rule_2,
            doc="Auxiliary constraint for variable z2",
        )

        def z2_rule_3(m, ec_exp, ret, d, t):
            return m.P_export[ec_exp, d, t] - m.z2[ec_exp, ret, d, t] >= 0

        self.m.z2_rule_3_constr = pe.Constraint(
            self.m.Energy_carriers_exp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z2_rule_3,
            doc="Auxiliary constraint for variable z2",
        )

        def z2_rule_4(m, ec_exp, ret, d, t):
            return m.P_export[ec_exp, d, t] - m.z2[ec_exp, ret, d, t] <= m.BigM * (
                1 - m.y_retrofit[ret]
            )

        self.m.z2_rule_4_constr = pe.Constraint(
            self.m.Energy_carriers_exp,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z2_rule_4,
            doc="Auxiliary constraint for variable z2",
        )

        def z3_rule_1(m, sol, ret):
            return m.z3[sol, ret] >= 0

        self.m.z3_rule_1_constr = pe.Constraint(
            self.m.Solar_tech,
            self.m.Retrofit_scenarios,
            rule=z3_rule_1,
            doc="Auxiliary constraint for variable z3",
        )

        def z3_rule_2(m, sol, ret):
            return m.z3[sol, ret] <= m.BigM * m.y_retrofit[ret]

        self.m.z3_rule_2_constr = pe.Constraint(
            self.m.Solar_tech,
            self.m.Retrofit_scenarios,
            rule=z3_rule_2,
            doc="Auxiliary constraint for variable z3",
        )

        def z3_rule_3(m, sol, ret):
            return m.Conv_cap[sol] - m.z3[sol, ret] >= 0

        self.m.z3_rule_3_constr = pe.Constraint(
            self.m.Solar_tech,
            self.m.Retrofit_scenarios,
            rule=z3_rule_3,
            doc="Auxiliary constraint for variable z3",
        )

        def z3_rule_4(m, sol, ret):
            return m.Conv_cap[sol] - m.z3[sol, ret] <= m.BigM * (1 - m.y_retrofit[ret])

        self.m.z3_rule_4_constr = pe.Constraint(
            self.m.Solar_tech,
            self.m.Retrofit_scenarios,
            rule=z3_rule_4,
            doc="Auxiliary constraint for variable z3",
        )

        def z4_rule_1(m, stor_tech, ret, d, t):
            return m.z4[stor_tech, ret, d, t] >= 0

        self.m.z4_rule_1_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z4_rule_1,
            doc="Auxiliary constraint for variable z4",
        )

        def z4_rule_2(m, stor_tech, ret, d, t):
            return m.z4[stor_tech, ret, d, t] <= m.BigM * m.y_retrofit[ret]

        self.m.z4_rule_2_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z4_rule_2,
            doc="Auxiliary constraint for variable z4",
        )

        def z4_rule_3(m, stor_tech, ret, d, t):
            return m.Qin[stor_tech, d, t] - m.z4[stor_tech, ret, d, t] >= 0

        self.m.z4_rule_3_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z4_rule_3,
            doc="Auxiliary constraint for variable z4",
        )

        def z4_rule_4(m, stor_tech, ret, d, t):
            return m.Qin[stor_tech, d, t] - m.z4[stor_tech, ret, d, t] <= m.BigM * (
                1 - m.y_retrofit[ret]
            )

        self.m.z4_rule_4_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z4_rule_4,
            doc="Auxiliary constraint for variable z4",
        )

        def z5_rule_1(m, stor_tech, ret, d, t):
            return m.z5[stor_tech, ret, d, t] >= 0

        self.m.z5_rule_1_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z5_rule_1,
            doc="Auxiliary constraint for variable z5",
        )

        def z5_rule_2(m, stor_tech, ret, d, t):
            return m.z5[stor_tech, ret, d, t] <= m.BigM * m.y_retrofit[ret]

        self.m.z5_rule_2_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z5_rule_2,
            doc="Auxiliary constraint for variable z5",
        )

        def z5_rule_3(m, stor_tech, ret, d, t):
            return m.Qout[stor_tech, d, t] - m.z5[stor_tech, ret, d, t] >= 0

        self.m.z5_rule_3_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z5_rule_3,
            doc="Auxiliary constraint for variable z5",
        )

        def z5_rule_4(m, stor_tech, ret, d, t):
            return m.Qout[stor_tech, d, t] - m.z5[stor_tech, ret, d, t] <= m.BigM * (
                1 - m.y_retrofit[ret]
            )

        self.m.z5_rule_4_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Retrofit_scenarios,
            self.m.Days,
            self.m.Time_steps,
            rule=z5_rule_4,
            doc="Auxiliary constraint for variable z5",
        )

        #%% Objective functions
        # ===================

        # Cost objective
        # --------------
        def cost_obj_rule(m):
            return m.Total_cost

        self.m.Cost_obj = pe.Objective(rule=cost_obj_rule, sense=pe.minimize)

        # Carbon objective
        # ----------------
        def carbon_obj_rule(m):
            return m.Total_carbon

        self.m.Carbon_obj = pe.Objective(rule=carbon_obj_rule, sense=pe.minimize)

    def solve(self, mip_gap=0.001, time_limit=10 ** 8, results_folder=".\\"):
        """
        Solves the model and outputs model results

        Two types of model outputs are generated:

            1. All the model definitions, constraints, parameter and variable values are given for each objective/Pareto point in the self.results object.
            2. The key objective function, design and operation results are given as follows:
                * obj: Contains the total cost, cost breakdown, and total carbon results. It is a data frame for all optim_mode settings.
                * dsgn: Contains the generation and storage capacities of all candidate technologies. It is a data frame for all optim_mode settings.
                * oper: Contains the generation, export and storage energy flows for all time steps considered. It is a single dataframe when optim_mode is 1 or 2 (single-objective) and a list of dataframes for each Pareto point when optim_mode is set to 3 (multi-objective).
        """

        import Output_functions as of
        import pickle as pkl

        # ====================================|
        # Main optimization solving procedure |
        # ====================================|

        # Solver definition
        # -----------------
        optimizer = pyomo.opt.SolverFactory("gurobi")
        optimizer.options["MIPGap"] = mip_gap
        optimizer.options["TimeLimit"] = time_limit

        # Excel file with all variable values
        from EHret_example_final_no_certificates import i

        if self.optim_mode == 1:

            # Cost minimization
            # =================
            all_vars = [None]

            self.m.Carbon_obj.deactivate()
            results = optimizer.solve(
                self.m, tee=True, keepfiles=True, logfile="gur.log"
            )

            # Save results
            # ------------
            self.m.solutions.store_to(results)
            all_vars[0] = of.get_all_vars(self.m)

            # JSON file with results
            results.write(
                filename=results_folder + "\cost_min_solver_results.json", format="json"
            )

            # Pickle file with all variable values
            file = open(results_folder + "\cost_min.p", "wb")
            pkl.dump(all_vars, file)
            file.close()

            of.write_all_vars_to_excel(
                all_vars[0], results_folder + "\cost_min_scenario_" + str(i)
            )

        elif self.optim_mode == 2:

            # Carbon minimization
            # ===================
            all_vars = [None]

            self.m.Carbon_obj.activate()
            self.m.Cost_obj.deactivate()
            optimizer.solve(self.m, tee=True, keepfiles=True, logfile="gur.log")
            carb_min = pe.value(self.m.Total_carbon) * 1.01

            self.m.epsilon = carb_min
            self.m.Carbon_obj.deactivate()
            self.m.Cost_obj.activate()
            results = optimizer.solve(
                self.m, tee=True, keepfiles=True, logfile="gur.log"
            )

            # Save results
            # ------------
            self.m.solutions.store_to(results)
            all_vars[0] = of.get_all_vars(self.m)

            # JSON file with results
            results.write(
                filename=results_folder + "\carb_min_solver_results.json", format="json"
            )

            # Pickle file with all variable values
            file = open(results_folder + "\carb_min.p", "wb")
            pkl.dump(all_vars, file)
            file.close()

            of.write_all_vars_to_excel(
                all_vars[0], results_folder + "\carb_min_scenario_" + str(i)
            )

        else:

            # Multi-objective optimization
            # ============================
            all_vars = [None] * (self.num_of_pfp + 2)

            # Cost minimization
            # -----------------
            self.m.Carbon_obj.deactivate()
            results = optimizer.solve(
                self.m, tee=True, keepfiles=True, logfile="gur.log"
            )
            carb_max = pe.value(self.m.Total_carbon)

            # Save results
            # ------------
            self.m.solutions.store_to(results)
            all_vars[0] = of.get_all_vars(self.m)

            # JSON file with results
            results.write(
                filename=results_folder + "\MO_solver_results_1.json", format="json"
            )

            # Pickle file with all variable values
            file = open(results_folder + "\multi_obj_1.p", "wb")
            pkl.dump([all_vars[0]], file)
            file.close()

            # Excel file with all variable values
            of.write_all_vars_to_excel(all_vars[0], results_folder + "\multi_obj_1")

            # Carbon minimization
            # -------------------
            self.m.Carbon_obj.activate()
            self.m.Cost_obj.deactivate()
            optimizer.solve(self.m, tee=True, keepfiles=True, logfile="gur.log")
            carb_min = pe.value(self.m.Total_carbon) * 1.01

            # Pareto points
            # -------------
            if self.num_of_pfp == 0:
                self.m.epsilon = carb_min
                self.m.Carbon_obj.deactivate()
                self.m.Cost_obj.activate()
                results = optimizer.solve(
                    self.m, tee=True, keepfiles=True, logfile="gur.log"
                )

                # Save results
                # ------------
                self.m.solutions.store_to(results)
                all_vars[1] = of.get_all_vars(self.m)

                # JSON file with results
                results.write(
                    filename=results_folder + "\MO_solver_results_2.json", format="json"
                )

                # Pickle file with all variable values
                file = open(results_folder + "\multi_obj_2.p", "wb")
                pkl.dump([all_vars[1]], file)
                file.close()

                # Excel file with all variable values
                of.write_all_vars_to_excel(all_vars[1], results_folder + "\multi_obj_2")

            else:
                self.m.Carbon_obj.deactivate()
                self.m.Cost_obj.activate()

                interval = (carb_max - carb_min) / (self.num_of_pfp + 1)
                steps = list(np.arange(carb_min, carb_max, interval))
                steps.reverse()
                print(steps)

                for i in range(1, self.num_of_pfp + 1 + 1):
                    self.m.epsilon = steps[i - 1]
                    print(self.m.epsilon.extract_values())
                    results = optimizer.solve(
                        self.m, tee=True, keepfiles=True, logfile="gur.log"
                    )

                    # Save results
                    # ------------
                    self.m.solutions.store_to(results)
                    all_vars[i] = of.get_all_vars(self.m)

                    # JSON file with results
                    results.write(
                        filename=results_folder
                        + "\MO_solver_results_"
                        + str(i + 1)
                        + ".json",
                        format="json",
                    )

                    # Pickle file with all variable values
                    file = open(
                        results_folder + "\multi_obj_" + str(i + 1) + ".p", "wb"
                    )
                    pkl.dump([all_vars[i]], file)
                    file.close()

                    # Excel file with all variable values
                    of.write_all_vars_to_excel(
                        all_vars[i], results_folder + "\multi_obj_" + str(i + 1)
                    )

            # Pickle file with all variable values for all multi-objective runs
            file = open(results_folder + "\multi_obj_all_points.p", "wb")
            pkl.dump(all_vars, file)
            file.close()


if __name__ == "__main__":
    pass
