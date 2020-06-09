# -*- coding: utf-8 -*-
"""
Single energy hub - single stage model for the optimal design of a distributed energy system
Author: Georgios Mavromatidis (ETH Zurich, gmavroma@ethz.ch)
"""

import pyomo
import pyomo.opt
import pyomo.environ as pe

# import pandas as pd
import numpy as np


class EnergyHub:
    """This class implements a standard energy hub model for the optimal design and operation of distributed multi-energy systems"""

    def __init__(self, input_file, temp_res=1, optim_mode=3, num_of_pareto_points=5):
        """
        __init__ function to read in the input data and begin the model creation process

        Inputs to the function:
        -----------------------
            * input_file: .py file where the values for all model parameters are defined
            * temp_res (default = 1): 1: typical days optimization, 2: full horizon optimization (8760 hours), 3: typical days with continous storage state-of-charge
            * optim_mode (default = 3): 1: for cost minimization, 2: for carbon minimization, 3: for multi-objective optimization
            * num_of_pareto_points (default = 5): In case optim_mode is set to 3, then this specifies the number of Pareto points
        """
        import importlib

        self.inp = importlib.import_module(input_file)
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
        """Create the Pyomo energy hub model given the input data specified in the input_file"""

        self.m = pe.ConcreteModel()

        # Model sets
        # ==========

        # Temporal dimensions
        # -------------------
        self.m.Days = pe.Set(
            initialize=self.inp.Days,
            ordered=True,
            doc="The number of days considered in each year of the model | Index: d",
        )
        self.m.Time_steps = pe.Set(
            initialize=self.inp.Time_steps,
            # ordererd=True,
            doc="Time steps considered in the model | Index: t",
        )
        self.m.Calendar_days = pe.Set(
            initialize=list(range(1, 365 + 1)),
            ordered=True,
            doc="Set for each calendar day of a full year",
        )

        # Energy carriers
        # ---------------
        self.m.Energy_carriers = pe.Set(
            initialize=self.inp.Energy_carriers,
            doc="The set of all energy carriers considered in the model | Index : ec",
        )
        self.m.Energy_carriers_imp = pe.Set(
            initialize=self.inp.Energy_carriers_imp,
            within=self.m.Energy_carriers,
            doc="The set of energy carriers for which imports are possible | Index : ec_imp",
        )
        self.m.Energy_carriers_exp = pe.Set(
            initialize=self.inp.Energy_carriers_exp,
            within=self.m.Energy_carriers,
            doc="The set of energy carriers for which exports are possible | Index : ec_exp",
        )
        self.m.Energy_carriers_dem = pe.Set(
            initialize=self.inp.Energy_carriers_dem,
            within=self.m.Energy_carriers,
            doc="The set of energy carriers for which end-user demands are defined | Index : ec_dem",
        )

        # Technologies
        # ------------
        self.m.Conversion_tech = pe.Set(
            initialize=self.inp.Conversion_tech,
            doc="The energy conversion technologies of each energy hub candidate site | Index : conv_tech",
        )
        self.m.Solar_tech = pe.Set(
            initialize=self.inp.Solar_tech,
            within=self.m.Conversion_tech,
            doc="Subset for solar technologies | Index : sol",
        )
        self.m.Dispatchable_tech = pe.Set(
            initialize=self.inp.Dispatchable_tech,
            within=self.m.Conversion_tech,
            doc="Subset for dispatchable/controllable technologies | Index : disp",
        )
        self.m.Storage_tech = pe.Set(
            initialize=self.inp.Storage_tech,
            doc="The set of energy storage technologies for each energy hub candidate site | Index : strg_tech",
        )

        # Model parameters
        # ================

        # Load parameters
        # ---------------
        self.m.Demands = pe.Param(
            self.m.Energy_carriers_dem,
            self.m.Days,
            self.m.Time_steps,
            default=0,
            initialize=self.inp.Demands,
            doc="Time-varying energy demand patterns for the energy hub",
        )
        if self.temp_res == 1 or self.temp_res == 3:
            self.m.Number_of_days = pe.Param(
                self.m.Days,
                default=1,
                initialize=self.inp.Number_of_days,
                doc="The number of days that each time step of typical day corresponds to",
            )
        else:
            self.m.Number_of_days = pe.Param(
                self.m.Days,
                self.m.Time_steps,
                default=1,
                initialize=1,
                doc="Parameter equal to 1 for each time step, because full horizon optimization is performed (temp_res == 2)",
            )
        if self.temp_res == 3:
            self.m.C_to_T = pe.Param(
                self.m.Calendar_days,
                initialize=self.inp.C_to_T,
                within=self.m.Days,
                doc="Parameter to match each calendar day of a full year to a typical day for optimization",
            )

        # Technical parameters
        # --------------------
        self.m.Conv_factor = pe.Param(
            self.m.Conversion_tech,
            self.m.Energy_carriers,
            default=0,
            initialize=self.inp.Conv_factor,
            doc="The conversion factors of the technologies in the energy hub",
        )
        self.m.Minimum_part_load = pe.Param(
            self.m.Dispatchable_tech,
            default=0,
            initialize=self.inp.Minimum_part_load,
            doc="Minimum allowable part-load during the operation of dispatchable technologies",
        )
        self.m.Lifetime_tech = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp.Lifetime_tech,
            doc="Lifetime of energy generation technologies",
        )
        self.m.Storage_tech_coupling = pe.Param(
            self.m.Storage_tech,
            self.m.Energy_carriers,
            initialize=self.inp.Storage_tech_coupling,
            default=0,
            doc="Par",
        )
        self.m.Storage_max_charge = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Storage_max_charge,
            doc="Maximum charging rate in % of the total capacity for the storage technologies",
        )
        self.m.Storage_max_discharge = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Storage_max_discharge,
            doc="Maximum discharging rate in % of the total capacity for the storage technologies",
        )
        self.m.Storage_standing_losses = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Storage_standing_losses,
            doc="Standing losses for the storage technologies",
        )
        self.m.Storage_charging_eff = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Storage_charging_eff,
            doc="Efficiency of charging process for the storage technologies",
        )
        self.m.Storage_discharging_eff = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Storage_discharging_eff,
            doc="Efficiency of discharging process for the storage technologies",
        )
        self.m.Storage_max_cap = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Storage_max_cap,
            doc="Maximum allowable energy storage capacity per technology type",
        )
        self.m.Lifetime_stor = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Lifetime_stor,
            doc="Lifetime of energy storage technologies",
        )
        self.m.Network_efficiency = pe.Param(
            self.m.Energy_carriers_dem,
            default=1,
            initialize=self.inp.Network_efficiency,
            doc="The efficiency of the energy networks used by the energy hub",
        )
        self.m.Network_length = pe.Param(
            initialize=self.inp.Network_length,
            doc="The length of the thermal network for the energy hub",
        )
        self.m.Network_lifetime = pe.Param(
            initialize=self.inp.Network_lifetime,
            doc="The lifetime of the thermal network used by the energy hub",
        )

        # Cost parameters
        # ---------------
        self.m.Import_prices = pe.Param(
            self.m.Energy_carriers_imp,
            initialize=self.inp.Import_prices,
            default=0,
            doc="Prices for importing energy carriers from the grid",
        )
        self.m.Export_prices = pe.Param(
            self.m.Energy_carriers_exp,
            initialize=self.inp.Export_prices,
            default=0,
            doc="Feed-in tariffs for exporting energy carriers back to the grid",
        )
        self.m.Linear_conv_costs = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp.Linear_conv_costs,
            doc="Linear part of the investment cost (per kW or m2) for the generation technologies in the energy hub",
        )
        self.m.Fixed_conv_costs = pe.Param(
            self.m.Conversion_tech,
            initialize=self.inp.Fixed_conv_costs,
            doc="Fixed part of the investment cost (per kW or m2) for the generation technologies in the energy hub",
        )
        self.m.Linear_stor_costs = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Linear_stor_costs,
            doc="Linear part of the investment cost (per kWh) for the storage technologies in the energy hub",
        )
        self.m.Fixed_stor_costs = pe.Param(
            self.m.Storage_tech,
            initialize=self.inp.Fixed_stor_costs,
            doc="Fixed part of the investment cost (per kWh) for the storage technologies in the energy hub",
        )
        self.m.Network_inv_cost_per_m = pe.Param(
            initialize=self.inp.Network_inv_cost_per_m,
            doc="Investment cost per pipe m of the thermal network of the energy hub",
        )
        self.m.Discount_rate = pe.Param(
            initialize=self.inp.Discount_rate,
            doc="The interest rate used for the CRF calculation",
        )

        def CRF_tech_rule(m, conv_tech):
            return (
                self.m.Discount_rate
                * (1 + self.m.Discount_rate) ** self.m.Lifetime_tech[conv_tech]
            ) / ((1 + self.m.Discount_rate) ** self.m.Lifetime_tech[conv_tech] - 1)

        self.m.CRF_tech = pe.Param(
            self.m.Conversion_tech,
            initialize=CRF_tech_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of generation technologies",
        )

        def CRF_stor_rule(m, strg_tech):
            return (
                self.m.Discount_rate
                * (1 + self.m.Discount_rate) ** self.m.Lifetime_stor[strg_tech]
            ) / ((1 + self.m.Discount_rate) ** self.m.Lifetime_stor[strg_tech] - 1)

        self.m.CRF_stor = pe.Param(
            self.m.Storage_tech,
            initialize=CRF_stor_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of storage technologies",
        )

        def CRF_network_rule(m):
            return (
                self.m.Discount_rate
                * (1 + self.m.Discount_rate) ** self.m.Network_lifetime
            ) / ((1 + self.m.Discount_rate) ** self.m.Network_lifetime - 1)

        self.m.CRF_network = pe.Param(
            initialize=CRF_network_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of the networks used by the energy hub",
        )

        # Environmental parameters
        # ------------------------
        self.m.Carbon_factors_import = pe.Param(
            self.m.Energy_carriers_imp,
            initialize=self.inp.Carbon_factors_import,
            doc="Energy carrier CO2 emission factors",
        )
        self.m.epsilon = pe.Param(
            initialize=10 ** 8,
            mutable=True,
            doc="Epsilon value used for the multi-objective epsilon-constrained optimization",
        )

        # Misc parameters
        # ---------------
        self.m.Roof_area = pe.Param(
            initialize=self.inp.Roof_area,
            doc="Available roof area for the installation of solar technologies",
        )
        self.m.P_solar = pe.Param(
            self.m.Days,
            self.m.Time_steps,
            initialize=self.inp.P_solar,
            doc="Incoming solar radiation patterns (kWh/m2) for solar technologies",
        )
        self.m.BigM = pe.Param(default=10 ** 6, doc="Big M: Sufficiently large value")

        # Model variables
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
            self.m.Conversion_tech,
            self.m.Days,
            self.m.Time_steps,
            within=pe.Binary,
            doc="Binary variable indicating the on (=1) or off (=0) state of a dispatchable technology",
        )
        self.m.y_conv = pe.Var(
            self.m.Conversion_tech,
            within=pe.Binary,
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
            doc="Total carbon emissions due to the operation of the energy hub",
        )

        # Model constraints
        # =================

        # Energy demand balances
        # ----------------------
        def Load_balance_rule(m, ec, d, t):
            return (m.P_import[ec, d, t] if ec in m.Energy_carriers_imp else 0) + sum(
                m.P_conv[conv_tech, d, t] * m.Conv_factor[conv_tech, ec]
                for conv_tech in m.Conversion_tech
            ) + sum(
                m.Storage_tech_coupling[strg_tech, ec]
                * (m.Qout[strg_tech, d, t] - m.Qin[strg_tech, d, t])
                for strg_tech in m.Storage_tech
            ) == (
                m.Demands[ec, d, t] if ec in m.Energy_carriers_dem else 0
            ) / (
                m.Network_efficiency[ec] if ec in m.Energy_carriers_dem else 1
            ) + (
                m.P_export[ec, d, t] if ec in m.Energy_carriers_exp else 0
            )

        self.m.Load_balance = pe.Constraint(
            self.m.Energy_carriers,
            self.m.Days,
            self.m.Time_steps,
            rule=Load_balance_rule,
            doc="Energy balance for the energy hub including conversion, storage, losses, exchange and export flows",
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

        def Solar_input_rule(m, sol, d, t):
            return m.P_conv[sol, d, t] == m.P_solar[d, t] * m.Conv_cap[sol]

        self.m.Solar_input = pe.Constraint(
            self.m.Solar_tech,
            self.m.Days,
            self.m.Time_steps,
            rule=Solar_input_rule,
            doc="Constraint connecting the solar radiation per m2 with the area of solar PV technologies",
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
            return m.Conv_cap[conv_tech] <= m.BigM * m.y_conv[conv_tech]

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

        # Storage constraints
        # -------------------
        def Storage_balance_rule(m, strg_tech, d, t):
            if self.temp_res == 1:
                if t != 1:
                    return (
                        m.SoC[strg_tech, d, t]
                        == (1 - m.Storage_standing_losses[strg_tech])
                        * m.SoC[strg_tech, d, t - 1]
                        + m.Storage_charging_eff[strg_tech] * m.Qin[strg_tech, d, t]
                        - (1 / m.Storage_discharging_eff[strg_tech])
                        * m.Qout[strg_tech, d, t]
                    )
                else:
                    return (
                        m.SoC[strg_tech, d, t]
                        == (1 - m.Storage_standing_losses[strg_tech])
                        * m.SoC[strg_tech, d, t + 23]
                        + m.Storage_charging_eff[strg_tech] * m.Qin[strg_tech, d, t]
                        - (1 / m.Storage_discharging_eff[strg_tech])
                        * m.Qout[strg_tech, d, t]
                    )
            elif self.temp_res == 2:
                if t != 1:
                    return (
                        m.SoC[strg_tech, d, t]
                        == (1 - m.Storage_standing_losses[strg_tech])
                        * m.SoC[strg_tech, d, t - 1]
                        + m.Storage_charging_eff[strg_tech] * m.Qin[strg_tech, d, t]
                        - (1 / m.Storage_discharging_eff[strg_tech])
                        * m.Qout[strg_tech, d, t]
                    )
                else:
                    if d != 1:
                        return (
                            m.SoC[strg_tech, d, t]
                            == (1 - m.Storage_standing_losses[strg_tech])
                            * m.SoC[strg_tech, d - 1, t + 23]
                            + m.Storage_charging_eff[strg_tech] * m.Qin[strg_tech, d, t]
                            - (1 / m.Storage_discharging_eff[strg_tech])
                            * m.Qout[strg_tech, d, t]
                        )
                    else:
                        return (
                            m.SoC[strg_tech, d, t]
                            == (1 - m.Storage_standing_losses[strg_tech])
                            * m.SoC[strg_tech, d + 364, t + 23]
                            + m.Storage_charging_eff[strg_tech] * m.Qin[strg_tech, d, t]
                            - (1 / m.Storage_discharging_eff[strg_tech])
                            * m.Qout[strg_tech, d, t]
                        )
            elif self.temp_res == 3:
                if t != 1:
                    return (
                        m.SoC[strg_tech, d, t]
                        == (1 - m.Storage_standing_losses[strg_tech])
                        * m.SoC[strg_tech, d, t - 1]
                        + m.Storage_charging_eff[strg_tech]
                        * m.Qin[strg_tech, m.C_to_T[d], t]
                        - (1 / m.Storage_discharging_eff[strg_tech])
                        * m.Qout[strg_tech, m.C_to_T[d], t]
                    )
                else:
                    if d != 1:
                        return (
                            m.SoC[strg_tech, d, t]
                            == (1 - m.Storage_standing_losses[strg_tech])
                            * m.SoC[strg_tech, d - 1, t + 23]
                            + m.Storage_charging_eff[strg_tech]
                            * m.Qin[strg_tech, m.C_to_T[d], t]
                            - (1 / m.Storage_discharging_eff[strg_tech])
                            * m.Qout[strg_tech, m.C_to_T[d], t]
                        )
                    else:
                        return (
                            m.SoC[strg_tech, d, t]
                            == (1 - m.Storage_standing_losses[strg_tech])
                            * m.SoC[strg_tech, d + 364, t + 23]
                            + m.Storage_charging_eff[strg_tech]
                            * m.Qin[strg_tech, m.C_to_T[d], t]
                            - (1 / m.Storage_discharging_eff[strg_tech])
                            * m.Qout[strg_tech, m.C_to_T[d], t]
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

        def Storage_charg_rate_constr_rule(m, strg_tech, d, t):
            return (
                m.Qin[strg_tech, d, t]
                <= m.Storage_max_charge[strg_tech] * m.Storage_cap[strg_tech]
            )

        self.m.Storage_charg_rate_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            rule=Storage_charg_rate_constr_rule,
            doc="Constraint for the maximum allowable charging rate of the storage technologies",
        )

        def Storage_discharg_rate_constr_rule(m, strg_tech, d, t):
            return (
                m.Qout[strg_tech, d, t]
                <= m.Storage_max_charge[strg_tech] * m.Storage_cap[strg_tech]
            )

        self.m.Storage_discharg_rate_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            rule=Storage_discharg_rate_constr_rule,
            doc="Constraint for the maximum allowable discharging rate of the storage technologies",
        )

        def Storage_cap_constr_rule(m, strg_tech, d, t):
            return m.SoC[strg_tech, d, t] <= m.Storage_cap[strg_tech]

        self.m.Storage_cap_constr = pe.Constraint(
            self.m.Storage_tech,
            self.m.Days,
            self.m.Time_steps,
            rule=Storage_cap_constr_rule,
            doc="Constraint for non-violation of the capacity of the storage",
        )

        def Max_allowable_storage_cap_rule(m, strg_tech):
            return m.Storage_cap[strg_tech] <= m.Storage_max_cap[strg_tech]

        self.m.Max_allowable_storage_cap = pe.Constraint(
            self.m.Storage_tech,
            rule=Max_allowable_storage_cap_rule,
            doc="Constraint enforcing the maximum allowable storage capacity per type of storage technology",
        )

        def Fixed_cost_storage_rule(m, strg_tech):
            return m.Storage_cap[strg_tech] <= m.BigM * m.y_stor[strg_tech]

        self.m.Fixed_cost_storage = pe.Constraint(
            self.m.Storage_tech,
            rule=Fixed_cost_storage_rule,
            doc="Constraint for the formulation of the fixed cost in the objective function",
        )

        # Objective function definitions
        # ------------------------------
        def Import_cost_rule(m):
            return m.Import_cost == sum(
                m.Import_prices[ec_imp] * m.P_import[ec_imp, d, t] * m.Number_of_days[d]
                for ec_imp in m.Energy_carriers_imp
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Import_cost_def = pe.Constraint(
            rule=Import_cost_rule,
            doc="Definition of the operating cost component of the total energy system cost",
        )

        def Export_profit_rule(m):
            return m.Export_profit == sum(
                m.Export_prices[ec_exp] * m.P_export[ec_exp, d, t] * m.Number_of_days[d]
                for ec_exp in m.Energy_carriers_exp
                for d in m.Days
                for t in m.Time_steps
            )

        self.m.Export_profit_def = pe.Constraint(
            rule=Export_profit_rule,
            doc="Definition of the income due to electricity exports component of the total energy system cost",
        )

        def Investment_cost_rule(m):
            return (
                m.Investment_cost
                == sum(
                    (
                        m.Fixed_conv_costs[conv_tech] * m.y_conv[conv_tech]
                        + m.Linear_conv_costs[conv_tech] * m.Conv_cap[conv_tech]
                    )
                    * m.CRF_tech[conv_tech]
                    for conv_tech in m.Conversion_tech
                )
                + sum(
                    (
                        m.Fixed_stor_costs[strg_tech] * m.y_stor[strg_tech]
                        + m.Linear_stor_costs[strg_tech] * m.Storage_cap[strg_tech]
                    )
                    * m.CRF_stor[strg_tech]
                    for strg_tech in m.Storage_tech
                )
                + m.Network_inv_cost_per_m * m.Network_length * m.CRF_network
            )

        self.m.Investment_cost_def = pe.Constraint(
            rule=Investment_cost_rule,
            doc="Definition of the investment cost component of the total energy system cost",
        )

        def Total_cost_rule(m):
            return m.Total_cost == m.Investment_cost + m.Import_cost - m.Export_profit

        self.m.Total_cost_def = pe.Constraint(
            rule=Total_cost_rule,
            doc="Definition of the total cost model objective function",
        )

        def Total_carbon_rule(m):
            return m.Total_carbon == sum(
                m.Carbon_factors_import[ec_imp]
                * m.P_import[ec_imp, d, t]
                * m.Number_of_days[d]
                for ec_imp in m.Energy_carriers_imp
                for d in m.Days
                for t in m.Time_steps
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

        # Objective functions
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

    def solve(self):
        """
        Solves the model and outputs model results

        Usage:
        ------
            (obj, des, oper) = solve()

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
        # optimizer.options["MIPGap"] = 0.05
        # optimizer.options["TimeLimit"] = 3600*96-300

        if self.optim_mode == 1:

            # Cost minimization
            # =================
            obj = [None]
            dsgn = [None]
            oper = [None]
            all_vars = [None]

            self.m.Carbon_obj.deactivate()
            results = optimizer.solve(
                self.m, tee=True, keepfiles=True, logfile="gur.log"
            )

            # Save results
            # ------------
            self.m.solutions.store_to(results)
            results.write(
                filename="cost_min_solver_results.json", format="json"
            )  # JSON file with results
            of.pickle_solver_results(
                self.m, "cost_min_solver_results.p"
            )  # Write a pickle file with the SolverResults object

            obj[0] = of.get_obj_results(self.m)
            dsgn[0] = of.get_design_results(self.m)
            oper[0] = of.get_oper_results(self.m)

            all_vars[0] = of.get_all_vars(self.m)
            file = open("cost_min.p", "wb")
            pkl.dump(all_vars, file)
            file.close()

        elif self.optim_mode == 2:

            # Carbon minimization
            # ===================
            obj = [None]
            dsgn = [None]
            oper = [None]
            all_vars = [None]

            self.m.Carbon_obj.activate()
            self.m.Cost_obj.deactivate()
            optimizer.solve(self.m, tee=True, keepfiles=True, logfile="gur.log")
            carb_min = pe.value(self.m.Total_carbon)

            self.m.epsilon = carb_min
            self.m.Carbon_obj.deactivate()
            self.m.Cost_obj.activate()
            results = optimizer.solve(
                self.m, tee=True, keepfiles=True, logfile="gur.log"
            )

            # Save results
            # ------------
            self.m.solutions.store_to(results)
            results.write(
                filename="carb_min_solver_results.json", format="json"
            )  # JSON file with results
            of.pickle_solver_results(
                self.m, "carb_min_solver_results.p"
            )  # Write a pickle file with the SolverResults object

            obj[0] = of.get_obj_results(self.m)
            dsgn[0] = of.get_design_results(self.m)
            oper[0] = of.get_oper_results(self.m)

            all_vars[0] = of.get_all_vars(self.m)
            file = open("carb_min.p", "wb")
            pkl.dump(all_vars, file)
            file.close()

        else:

            obj = [None] * (self.num_of_pfp + 2)
            dsgn = [None] * (self.num_of_pfp + 2)
            oper = [None] * (self.num_of_pfp + 2)
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
            results.write(
                filename="MO_solver_results_1.json", format="json"
            )  # JSON file with results
            of.pickle_solver_results(
                self.m, "MO_solver_results_1.p"
            )  # Write a pickle file with the SolverResults object

            obj[0] = of.get_obj_results(self.m)
            dsgn[0] = of.get_design_results(self.m)
            oper[0] = of.get_oper_results(self.m)
            all_vars[0] = of.get_all_vars(self.m)

            # Carbon minimization
            # -------------------
            self.m.Carbon_obj.activate()
            self.m.Cost_obj.deactivate()
            optimizer.solve(self.m, tee=True, keepfiles=True, logfile="gur.log")
            carb_min = pe.value(self.m.Total_carbon)

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
                results.write(
                    filename="MO_solver_results_2.json", format="json"
                )  # JSON file with results
                of.pickle_solver_results(
                    sp.m, "MO_solver_results_2.p"
                )  # Write a pickle file with the SolverResults object

                obj[1] = of.get_obj_results(self.m)
                dsgn[1] = of.get_design_results(self.m)
                oper[1] = of.get_oper_results(self.m)
                all_vars[1] = of.get_all_vars(self.m)

            else:
                self.m.Carbon_obj.deactivate()
                self.m.Cost_obj.activate()

                interval = (carb_max - carb_min) / (self.num_of_pfp + 1)
                steps = list(np.arange(carb_min, carb_max, interval))
                steps.reverse()
                print(steps)

                for i in range(1, self.num_of_pfp + 1 + 1):
                    self.m.epsilon = steps[i - 1]
                    results = optimizer.solve(
                        self.m, tee=True, keepfiles=True, logfile="gur.log"
                    )

                    # Save results
                    # ------------
                    self.m.solutions.store_to(results)
                    results.write(
                        filename="MO_solver_results_" + str(i + 1) + ".json",
                        format="json",
                    )  # JSON file with results
                    of.pickle_solver_results(
                        sp.m, "MO_solver_results_" + str(i + 1) + ".p"
                    )  # Write a pickle file with the SolverResults object

                    obj[i] = of.get_obj_results(self.m)
                    dsgn[i] = of.get_design_results(self.m)
                    oper[i] = of.get_oper_results(self.m)
                    all_vars[i] = of.get_all_vars(self.m)

            file = open("multi_obj.p", "wb")
            pkl.dump(all_vars, file)
            file.close()

        return obj, dsgn, oper


if __name__ == "__main__":
    sp = EnergyHub("Input_data", 1, 1, 1)
    sp.create_model()  # Create model
    (obj, dsgn, oper) = sp.solve()
