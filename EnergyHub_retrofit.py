# -*- coding: utf-8 -*-
"""
Single energy hub - single stage model for the optimal design of a distributed energy system including building retrofit options

@author: Georgios Mavromatidis (ETH Zurich, gmavroma@ethz.ch)

"""

import pyomo
import pyomo.opt
import pyomo.environ as pe
import pyomoio as pyio
from pyomo.opt import SolverResults

import pandas as pd
import numpy as np


class EnergyHub_retrofit:
    """This class implements a standard energy hub model for the optimal design and operation of distributed multi-energy systems"""

    def __init__(self, input_file, temp_res=1, optim_mode=3, num_of_pareto_points=5):
        """
        __init__ function to read in the input data and begin the model creation process

        Inputs to the function:
        -----------------------
            * input_file: .py file where the values for all model parameters are defined
            * temp_res (default = 1): 1: typical days optimization, 2: full horizon optimization (8760 hours)
            * optim_mode (default = 3): 1: for cost minimization, 2: for carbon minimization, 3: for multi-objective optimization
            * num_of_pareto_points (default = 5): In case optim_mode is set to 3, then this specifies the number of Pareto points
        """
        import importlib

        self.InputFile = input_file
        self.inp = importlib.import_module(self.InputFile)
        self.temp_res = temp_res
        self.optim_mode = optim_mode
        if self.optim_mode == 1 or self.optim_mode == 2:
            self.num_of_pfp = 0
            print(
                "Warning: Number of Pareto front points specified is ignored. Single-objective optimization will be performed."
            )
        else:
            self.num_of_pfp = num_of_pareto_points
        self.createModel()

    def createModel(self):
        """Create the Pyomo energy hub model given the input data specified in the self.InputFile"""

        self.m = pe.ConcreteModel()

        # Model sets
        # ==========
        self.m.Time = pe.Set(
            initialize=self.inp.Time, doc="Time steps considered in the model"
        )

        def First_hour_rule(m):
            if self.temp_res == 1:
                return list(range(1, self.inp.Number_of_time_steps + 1, 24))
            else:
                return 1

        self.m.First_hour = pe.Set(
            initialize=First_hour_rule,
            within=self.m.Time,
            doc="if temp_res == 1: The first hour of each typical day considered in the model | if temp_res == 2, it is the first hour of the year (i.e. First_hour = 1)",
        )
        self.m.Inputs = pe.Set(
            initialize=self.inp.Inputs,
            doc="The energy conversion technologies of the energy hub",
        )
        self.m.Buildings = pe.RangeSet(
            1,
            self.inp.Number_of_buildings,
            doc="Number of buildings considered for solar installations",
        )
        self.m.Solar_pv_inputs = pe.Set(
            initialize=self.inp.Solar_pv_inputs,
            within=self.m.Inputs,
            doc="Solar PV technologies",
        )
        self.m.Solar_th_inputs = pe.Set(
            initialize=self.inp.Solar_th_inputs,
            within=self.m.Inputs,
            doc="Solar thermal technologies",
        )
        self.m.Inputs_wo_grid = pe.Set(
            initialize=self.inp.Inputs_wo_grid,
            doc="Subset of input energy streams without the grid",
        )
        self.m.Dispatchable_Tech = pe.Set(
            initialize=self.inp.Dispatchable_Tech,
            within=self.m.Inputs,
            doc="Subset for dispatchable/controllable technologies",
        )
        self.m.CHP_Tech = pe.Set(
            initialize=self.inp.CHP_Tech,
            within=self.m.Inputs,
            doc="Subset of CHP engine technologies",
        )
        self.m.Outputs = pe.Set(
            initialize=self.inp.Outputs,
            doc="Types of energy demands that the energy hub must supply",
        )
        self.m.Retrofit_scenarios = pe.Set(
            initialize=self.inp.Retrofit_scenarios,
            doc="Retrofit scenarios considered for the buildings connected to the energy hub",
        )

        # Model parameters
        # ================

        # Load parameters
        # ---------------
        self.m.Loads = pe.Param(
            self.m.Time,
            self.m.Retrofit_scenarios,
            self.m.Outputs,
            default=0,
            initialize=self.inp.Loads,
            doc="Time-varying energy demand patterns for the energy hub",
        )
        if self.temp_res == 1:
            self.m.Number_of_days = pe.Param(
                self.m.Time,
                self.m.Retrofit_scenarios,
                default=1,
                initialize=self.inp.Number_of_days,
                doc="The number of days that each time step of typical day corresponds to",
            )
        else:
            self.m.Number_of_days = pe.Param(
                self.m.Time,
                self.m.Retrofit_scenarios,
                default=1,
                initialize=1,
                doc="Parameter equal to 1 for each time step, because full horizon optimization is performed (temp_res == 2)",
            )

        # Technical parameters
        # --------------------
        self.m.Cmatrix = pe.Param(
            self.m.Outputs,
            self.m.Inputs,
            default=0,
            initialize=self.inp.Cmatrix,
            doc="The coupling matrix of the energy hub",
        )
        self.m.Minimum_part_load = pe.Param(
            self.m.Dispatchable_Tech,
            default=0,
            initialize=self.inp.Minimum_part_load,
            doc="Minimum allowable part-load during the operation of dispatchable technologies",
        )
        self.m.Storage_max_charge = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Storage_max_charge,
            doc="Maximum charging rate in % of the total capacity for the storage technologies",
        )
        self.m.Storage_max_discharge = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Storage_max_discharge,
            doc="Maximum discharging rate in % of the total capacity for the storage technologies",
        )
        self.m.Storage_standing_losses = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Storage_standing_losses,
            doc="Standing losses for the storage technologies",
        )
        self.m.Storage_charging_eff = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Storage_charging_eff,
            doc="Efficiency of charging process for the storage technologies",
        )
        self.m.Storage_discharging_eff = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Storage_discharging_eff,
            doc="Efficiency of discharging process for the storage technologies",
        )
        self.m.Storage_max_cap = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Storage_max_cap,
            doc="Maximum allowable energy storage capacity per technology type",
        )
        self.m.Lifetime_tech = pe.Param(
            self.m.Inputs,
            initialize=self.inp.Lifetime_tech,
            doc="Lifetime of energy generation technologies",
        )
        self.m.Lifetime_stor = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Lifetime_stor,
            doc="Lifetime of energy storage technologies",
        )
        self.m.Lifetime_retrofit = pe.Param(
            self.m.Retrofit_scenarios,
            initialize=self.inp.Lifetime_retrofit,
            doc="Lifetime considered for each retrofit scenario",
        )
        self.m.Network_efficiency = pe.Param(
            self.m.Outputs,
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
        self.m.Operating_costs = pe.Param(
            self.m.Inputs,
            initialize=self.inp.Operating_costs,
            doc="Energy carrier costs at the input of the energy hub",
        )
        self.m.Linear_inv_costs = pe.Param(
            self.m.Inputs,
            initialize=self.inp.Linear_inv_costs,
            doc="Linear part of the investment cost (per kW or m2) for the generation technologies in the energy hub",
        )
        self.m.Fixed_inv_costs = pe.Param(
            self.m.Inputs,
            initialize=self.inp.Fixed_inv_costs,
            doc="Fixed part of the investment cost (per kW or m2) for the generation technologies in the energy hub",
        )
        self.m.Linear_stor_costs = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Linear_stor_costs,
            doc="Linear part of the investment cost (per kWh) for the storage technologies in the energy hub",
        )
        self.m.Fixed_stor_costs = pe.Param(
            self.m.Outputs,
            initialize=self.inp.Fixed_stor_costs,
            doc="Fixed part of the investment cost (per kWh) for the storage technologies in the energy hub",
        )
        self.m.Network_inv_cost_per_m = pe.Param(
            initialize=self.inp.Network_inv_cost_per_m,
            doc="Investment cost per pipe m of the thermal network of the energy hub",
        )
        self.m.Retrofit_inv_costs = pe.Param(
            self.m.Retrofit_scenarios,
            initialize=self.inp.Retrofit_inv_costs,
            doc="Investment cost for each of the considered retrofit scenarios",
        )
        self.m.FiT = pe.Param(
            self.m.Outputs,
            initialize=self.inp.FiT,
            doc="Feed-in tariffs for exporting electricity back to the grid",
        )
        self.m.Interest_rate = pe.Param(
            initialize=self.inp.Interest_rate,
            doc="The interest rate used for the CRF calculation",
        )

        def CRF_tech_rule(m, inp):
            return (
                self.m.Interest_rate
                * (1 + self.m.Interest_rate) ** self.m.Lifetime_tech[inp]
            ) / ((1 + self.m.Interest_rate) ** self.m.Lifetime_tech[inp] - 1)

        self.m.CRF_tech = pe.Param(
            self.m.Inputs,
            initialize=CRF_tech_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of generation technologies",
        )

        def CRF_stor_rule(m, out):
            return (
                self.m.Interest_rate
                * (1 + self.m.Interest_rate) ** self.m.Lifetime_stor[out]
            ) / ((1 + self.m.Interest_rate) ** self.m.Lifetime_stor[out] - 1)

        self.m.CRF_stor = pe.Param(
            self.m.Outputs,
            initialize=CRF_stor_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of storage technologies",
        )

        def CRF_network_rule(m):
            return (
                self.m.Interest_rate
                * (1 + self.m.Interest_rate) ** self.m.Network_lifetime
            ) / ((1 + self.m.Interest_rate) ** self.m.Network_lifetime - 1)

        self.m.CRF_network = pe.Param(
            initialize=CRF_network_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of the networks used by the energy hub",
        )

        def CRF_retrofit_rule(m, ret):
            return (
                self.m.Interest_rate
                * (1 + self.m.Interest_rate) ** self.m.Lifetime_retrofit[ret]
            ) / ((1 + self.m.Interest_rate) ** self.m.Lifetime_retrofit[ret] - 1)

        self.m.CRF_retrofit = pe.Param(
            self.m.Retrofit_scenarios,
            initialize=CRF_retrofit_rule,
            doc="Capital Recovery Factor (CRF) used to annualise the investment cost of the considered retrofit scenarios",
        )

        # Environmental parameters
        # ------------------------
        self.m.Carbon_factors = pe.Param(
            self.m.Inputs,
            initialize=self.inp.Carbon_factors,
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
            self.m.Buildings,
            initialize=self.inp.Roof_area,
            doc="Available roof area for the installation of solar technologies",
        )
        self.m.P_solar = pe.Param(
            self.m.Time,
            self.m.Retrofit_scenarios,
            self.m.Buildings,
            initialize=self.inp.P_solar,
            doc="Incoming solar radiation patterns (kWh/m2) for solar technologies",
        )
        self.m.BigM = pe.Param(default=10 ** 6, doc="Big M: Sufficiently large value")

        # Model variables
        # ===============

        # Generation technologies
        # -----------------------
        self.m.P = pe.Var(
            self.m.Time,
            self.m.Inputs,
            within=pe.NonNegativeReals,
            doc="The input energy stream at each generation device of the energy hub at each time step",
        )
        self.m.P_export = pe.Var(
            self.m.Time,
            self.m.Outputs,
            within=pe.NonNegativeReals,
            doc="Exported energy (in this case only electricity exports are allowed)",
        )
        self.m.y_on = pe.Var(
            self.m.Dispatchable_Tech,
            self.m.Time,
            within=pe.Binary,
            doc="Binary variable indicating the on (=1) or off (=0) state of a dispatchable technology",
        )
        self.m.y = pe.Var(
            self.m.Inputs,
            within=pe.Binary,
            doc="Binary variable denoting the installation (=1) of energy generation technology",
        )
        self.m.Capacity = pe.Var(
            self.m.Inputs,
            within=pe.NonNegativeReals,
            doc="Installed capacity for energy generation technology",
        )
        # self.m.P_export = pe.Var(((t, out) for t in self.m.Time for out in self.m.Outputs if out == 'Elec'), within = pe.NonNegativeReals,
        #                          doc = 'Exported energy (in this case only electricity exports are allowed)')
        # self.m.Capacity= pe.Var((out,inp) for out in self.m.Outputs for inp in self.m.Inputs if self.m.Cmatrix[out,inp] > 0, within = pe.NonNegativeReals,
        #                         doc = 'Installed capacity for energy generation technology')

        # Storage technologies
        # --------------------
        self.m.Qin = pe.Var(
            self.m.Time,
            self.m.Outputs,
            within=pe.NonNegativeReals,
            doc="Storage charging rate",
        )
        self.m.Qout = pe.Var(
            self.m.Time,
            self.m.Outputs,
            within=pe.NonNegativeReals,
            doc="Storage discharging rate",
        )
        self.m.E = pe.Var(
            self.m.Time,
            self.m.Outputs,
            within=pe.NonNegativeReals,
            doc="Storage state of charge",
        )
        self.m.y_stor = pe.Var(
            self.m.Outputs,
            within=pe.Binary,
            doc="Binary variable denoting the installation (=1) of energy storage technology",
        )
        self.m.Storage_cap = pe.Var(
            self.m.Outputs,
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

        # Objective function components
        # -----------------------------
        self.m.Operating_cost = pe.Var(
            within=pe.NonNegativeReals,
            doc="The operating cost for the consumption of energy carriers",
        )
        self.m.Income_via_exports = pe.Var(
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

        # Bilinear term reformulations
        # ----------------------------
        # These variables need to be defined in order to linearize the products of binary and continuous variables in the model. These are then in turn caused by the fact that when typical days are used, the "Number of typical days" parameter and the "P_solar" parameter differ per retrofit scenario

        self.m.z1 = pe.Var(
            self.m.Time,
            self.m.Inputs,
            self.m.Retrofit_scenarios,
            doc="Variable to represent the product: P[t,inp] * y_retrofit[ret]",
        )
        self.m.z2 = pe.Var(
            self.m.Time,
            self.m.Outputs,
            self.m.Retrofit_scenarios,
            doc="Variable to represent the product: P_export[t,out] * y_retrofit[ret]",
        )
        self.m.z3 = pe.Var(
            self.m.Solar_pv_inputs,
            self.m.Retrofit_scenarios,
            doc="Variable to represent the product: Capacity[pv] * y_retrofit[ret]",
        )
        self.m.z4 = pe.Var(
            self.m.Solar_th_inputs,
            self.m.Retrofit_scenarios,
            doc="Variable to represent the product: Capacity[st] * y_retrofit[ret]",
        )

        # Model constraints
        # =================

        # Energy demand balances
        # ----------------------
        def Load_balance_rule(m, t, out):
            return (
                sum(m.P[t, inp] * m.Cmatrix[out, inp] for inp in m.Inputs)
                + m.Qout[t, out]
                - m.Qin[t, out]
                == sum(
                    m.y_retrofit[ret] * m.Loads[t, ret, out] for ret in self.m.Retrofit_scenarios
                )
                / m.Network_efficiency[out]
                + m.P_export[t, out]
            )

        self.m.Load_balance = pe.Constraint(
            self.m.Time, self.m.Outputs, rule=Load_balance_rule
        )

        def No_heat_export_rule(m, t, out):
            return m.P_export[t, out] == 0

        self.m.No_heat_export = pe.Constraint(
            ((t, out) for t in self.m.Time for out in self.m.Outputs if out == "Heat"),
            rule=No_heat_export_rule,
            doc="No heat exports allowed",
        )

        # Generation constraints
        # ----------------------
        def Capacity_constraint_rule(m, t, disp, out):
            return m.P[t, disp] * m.Cmatrix[out, disp] <= m.Capacity[disp]

        self.m.Capacity_constraint = pe.Constraint(
            self.m.Time,
            self.m.Dispatchable_Tech,
            self.m.Outputs,
            rule=Capacity_constraint_rule,
            doc="Constraint preventing capacity violation for the generation technologies of the energy hub",
        )

        def Solar_pv_input_rule(m, t, sol_pv, bldg):
            #            return m.P[t, sol_pv] == sum(
            #                m.P_solar[t, bldg] * y_retrofit[ret] * m.Capacity[sol_pv]
            #                for ret in m.Retrofit_scenarios
            #            )
            return m.P[t, sol_pv] == sum(
                m.P_solar[t, ret, bldg] * m.z3[sol_pv, ret] for ret in m.Retrofit_scenarios
            )

        self.m.Solar_pv_input = pe.Constraint(
            (
                (t, sol_pv, bldg)
                for t in self.m.Time
                for i, sol_pv in enumerate(self.m.Solar_pv_inputs)
                for k, bldg in enumerate(self.m.Buildings)
                if i == k
            ),
            rule=Solar_pv_input_rule,
            doc="Constraint connecting the solar radiation per m2 with the area of solar PV technologies",
        )

        def Solar_th_input_rule(m, t, sol_th, bldg):
            #            return m.P[t, sol_pv] == sum(
            #                m.P_solar[t, bldg] * y_retrofit[ret] * m.Capacity[sol_th]
            #                for ret in m.Retrofit_scenarios
            #            )
            return m.P[t, sol_th] == sum(
                m.P_solar[t, ret, bldg] * m.z4[sol_th, ret] for ret in m.Retrofit_scenarios
            )

        self.m.Solar_th_input = pe.Constraint(
            (
                (t, sol_th, bldg)
                for t in self.m.Time
                for i, sol_th in enumerate(self.m.Solar_th_inputs)
                for k, bldg in enumerate(self.m.Buildings)
                if i == k
            ),
            rule=Solar_th_input_rule,
            doc="Constraint connecting the solar radiation per m2 with the area of solar thermal technologies",
        )

        def Minimum_part_load_constr_rule1(m, t, disp, out):
            return m.P[t, disp] * m.Cmatrix[out, disp] <= m.BigM * m.y_on[disp, t]

        def Minimum_part_load_constr_rule2(m, t, disp, out):
            return (
                m.P[t, disp] * m.Cmatrix[out, disp] + m.BigM * (1 - m.y_on[disp, t])
                >= m.Minimum_part_load[disp] * m.Capacity[disp]
            )

        self.m.Mininum_part_rule_constr1 = pe.Constraint(
            (
                (t, disp, out)
                for t in self.m.Time
                for disp in self.m.Dispatchable_Tech
                for out in self.m.Outputs
                if self.m.Cmatrix[out, disp] > 0
            ),
            rule=Minimum_part_load_constr_rule1,
            doc="Constraint enforcing a minimum load during the operation of a dispatchable energy technology",
        )

        self.m.Mininum_part_rule_constr2 = pe.Constraint(
            (
                (t, disp, out)
                for t in self.m.Time
                for disp in self.m.Dispatchable_Tech
                for out in self.m.Outputs
                if self.m.Cmatrix[out, disp] > 0
            ),
            rule=Minimum_part_load_constr_rule2,
            doc="Constraint enforcing a minimum load during the operation of a dispatchable energy technology",
        )

        def Fixed_cost_constr_rule(m, inp):
            return m.Capacity[inp] <= m.BigM * m.y[inp]

        self.m.Fixed_cost_constr = pe.Constraint(
            self.m.Inputs,
            rule=Fixed_cost_constr_rule,
            doc="Constraint for the formulation of the fixed cost in the objective function",
        )

        def Roof_area_non_violation_rule(m, sol_pv, sol_st, bldg):
            return m.Capacity[sol_pv] + m.Capacity[sol_st] <= m.Roof_area[bldg]

        self.m.Roof_area_non_violation = pe.Constraint(
            (
                (pv, st, bldg)
                for i, pv in enumerate(self.m.Solar_pv_inputs)
                for j, st in enumerate(self.m.Solar_th_inputs)
                for k, bldg in enumerate(self.m.Buildings)
                if i == j == k
            ),
            rule=Roof_area_non_violation_rule,
            doc="Non violation of the maximum roof area for solar installations",
        )

        # Storage constraints
        # -------------------

        def Storage_balance_rule(m, t, out):
            return (
                m.E[t, out]
                == (1 - m.Storage_standing_losses[out]) * m.E[t - 1, out]
                + m.Storage_charging_eff[out] * m.Qin[t, out]
                - (1 / m.Storage_discharging_eff[out]) * m.Qout[t, out]
            )

        self.m.Storage_balance = pe.Constraint(
            self.m.Time - self.m.First_hour,
            self.m.Outputs,
            rule=Storage_balance_rule,
            doc="Energy balance for the storage modules considering incoming and outgoing energy flows",
        )

        def Storage_balance_cycling_rule(m, t, out):
            if self.temp_res == 1:
                return (
                    m.E[t, out]
                    == (1 - m.Storage_standing_losses[out]) * m.E[t + 23, out]
                    + m.Storage_charging_eff[out] * m.Qin[t, out]
                    - (1 / m.Storage_discharging_eff[out]) * m.Qout[t, out]
                )
            else:
                return (
                    m.E[t, out]
                    == (1 - m.Storage_standing_losses[out]) * m.E[t + 8759, out]
                    + m.Storage_charging_eff[out] * m.Qin[t, out]
                    - (1 / m.Storage_discharging_eff[out]) * m.Qout[t, out]
                )

        self.m.Storage_balance_cycling = pe.Constraint(
            self.m.First_hour,
            self.m.Outputs,
            rule=Storage_balance_cycling_rule,
            doc="Energy balance for the storage modules considering incoming and outgoing energy flows",
        )

        def Storage_charg_rate_constr_rule(m, t, out):
            return m.Qin[t, out] <= m.Storage_max_charge[out] * m.Storage_cap[out]

        self.m.Storage_charg_rate_constr = pe.Constraint(
            self.m.Time,
            self.m.Outputs,
            rule=Storage_charg_rate_constr_rule,
            doc="Constraint for the maximum allowable charging rate of the storage technologies",
        )

        def Storage_discharg_rate_constr_rule(m, t, out):
            return m.Qout[t, out] <= m.Storage_max_charge[out] * m.Storage_cap[out]

        self.m.Storage_discharg_rate_constr = pe.Constraint(
            self.m.Time,
            self.m.Outputs,
            rule=Storage_discharg_rate_constr_rule,
            doc="Constraint for the maximum allowable discharging rate of the storage technologies",
        )

        def Storage_cap_constr_rule(m, t, out):
            return m.E[t, out] <= m.Storage_cap[out]

        self.m.Storage_cap_constr = pe.Constraint(
            self.m.Time,
            self.m.Outputs,
            rule=Storage_cap_constr_rule,
            doc="Constraint for non-violation of the capacity of the storage",
        )

        def Max_allowable_storage_cap_rule(m, out):
            return m.Storage_cap[out] <= m.Storage_max_cap[out]

        self.m.Max_allowable_storage_cap = pe.Constraint(
            self.m.Outputs,
            rule=Max_allowable_storage_cap_rule,
            doc="Constraint enforcing the maximum allowable storage capacity per type of storage technology",
        )

        def Fixed_cost_storage_rule(m, out):
            return m.Storage_cap[out] <= m.BigM * m.y_stor[out]

        self.m.Fixed_cost_storage = pe.Constraint(
            self.m.Outputs,
            rule=Fixed_cost_storage_rule,
            doc="Constraint for the formulation of the fixed cost in the objective function",
        )

        # Retrofit constraints
        # --------------------
        def One_retrofit_state_rule(m):
            return sum(m.y_retrofit[ret] for ret in m.Retrofit_scenarios) == 1

        self.m.One_retrofit_state_def = pe.Constraint(
            rule = One_retrofit_state_rule,
            doc = "Constraint to impose that one retrofit state out of all possible must be selected"
        )

        # Objective function definitions
        # ------------------------------

        def Operating_cost_rule(m):
            return m.Operating_cost == sum(
                # m.Operating_costs[inp] * m.P[t, inp] * m.Number_of_days[t, ret] * m.y_retrofit[ret]
                m.Operating_costs[inp] * m.Number_of_days[t, ret] * m.z1[t, inp, ret]
                for t in m.Time
                for inp in m.Inputs
                for ret in m.Retrofit_scenarios
            )

        self.m.Operating_cost_def = pe.Constraint(
            rule=Operating_cost_rule,
            doc="Definition of the operating cost component of the total energy system cost",
        )

        def Income_via_exports_rule(m):
            return m.Income_via_exports == sum(
                # m.FiT[out] * m.P_export[t, out] * m.Number_of_days[t, ret] * m.y_retrofit[ret]
                m.FiT[out] * m.Number_of_days[t, ret] * m.z2[t, out, ret]
                for t in m.Time
                for out in m.Outputs
                for ret in m.Retrofit_scenarios
            )

        self.m.Income_via_exports_def = pe.Constraint(
            rule=Income_via_exports_rule,
            doc="Definition of the income due to electricity exports component of the total energy system cost",
        )

        def Investment_cost_rule(m):
            return m.Investment_cost == sum(
                (
                    m.Fixed_inv_costs[inp] * m.y[inp]
                    + m.Linear_inv_costs[inp] * m.Capacity[inp]
                )
                * m.CRF_tech[inp]
                for inp in m.Inputs
            ) + sum(
                (
                    m.Fixed_stor_costs[out] * m.y_stor[out]
                    + m.Linear_stor_costs[out] * m.Storage_cap[out]
                )
                * m.CRF_stor[out]
                for out in m.Outputs
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
                == m.Investment_cost + m.Operating_cost - m.Income_via_exports
            )

        self.m.Total_cost_def = pe.Constraint(
            rule=Total_cost_rule,
            doc="Definition of the total cost model objective function",
        )

        def Total_carbon_rule(m):
            return m.Total_carbon == sum(
                # m.Carbon_factors[inp] * m.P[t, inp] * m.Number_of_days[t, ret] * y_retrofit[ret]
                m.Carbon_factors[inp] * m.Number_of_days[t, ret] * m.z1[t, inp, ret]
                for t in m.Time
                for inp in m.Inputs
                for ret in m.Retrofit_scenarios
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

        def z1_rule_1(m, t, inp, ret):
            return m.z1[t, inp, ret] >= 0

        self.m.z1_rule_1_constr = pe.Constraint(
            self.m.Time,
            self.m.Inputs,
            self.m.Retrofit_scenarios,
            rule=z1_rule_1,
            doc="Auxiliary constraint for variable z1",
        )

        def z1_rule_2(m, t, inp, ret):
            return m.z1[t, inp, ret] <= m.BigM * m.y_retrofit[ret]

        self.m.z1_rule_2_constr = pe.Constraint(
            self.m.Time,
            self.m.Inputs,
            self.m.Retrofit_scenarios,
            rule=z1_rule_2,
            doc="Auxiliary constraint for variable z1",
        )

        def z1_rule_3(m, t, inp, ret):
            return m.P[t, inp] - m.z1[t, inp, ret] >= 0

        self.m.z1_rule_3_constr = pe.Constraint(
            self.m.Time,
            self.m.Inputs,
            self.m.Retrofit_scenarios,
            rule=z1_rule_3,
            doc="Auxiliary constraint for variable z1",
        )

        def z1_rule_4(m, t, inp, ret):
            return m.P[t, inp] - m.z1[t, inp, ret] <= m.BigM * (1 - m.y_retrofit[ret])

        self.m.z1_rule_4_constr = pe.Constraint(
            self.m.Time,
            self.m.Inputs,
            self.m.Retrofit_scenarios,
            rule=z1_rule_4,
            doc="Auxiliary constraint for variable z1",
        )

        def z2_rule_1(m, t, out, ret):
            return m.z2[t, out, ret] >= 0

        self.m.z2_rule_1_constr = pe.Constraint(
            self.m.Time,
            self.m.Outputs,
            self.m.Retrofit_scenarios,
            rule=z2_rule_1,
            doc="Auxiliary constraint for variable z2",
        )

        def z2_rule_2(m, t, out, ret):
            return m.z2[t, out, ret] <= m.BigM * m.y_retrofit[ret]

        self.m.z2_rule_2_constr = pe.Constraint(
            self.m.Time,
            self.m.Outputs,
            self.m.Retrofit_scenarios,
            rule=z2_rule_2,
            doc="Auxiliary constraint for variable z2",
        )

        def z2_rule_3(m, t, out, ret):
            return m.P_export[t, out] - m.z2[t, out, ret] >= 0

        self.m.z2_rule_3_constr = pe.Constraint(
            self.m.Time,
            self.m.Outputs,
            self.m.Retrofit_scenarios,
            rule=z2_rule_3,
            doc="Auxiliary constraint for variable z2",
        )

        def z2_rule_4(m, t, out, ret):
            return m.P_export[t, out] - m.z2[t, out, ret] <= m.BigM * (
                1 - m.y_retrofit[ret]
            )

        self.m.z2_rule_4_constr = pe.Constraint(
            self.m.Time,
            self.m.Outputs,
            self.m.Retrofit_scenarios,
            rule=z2_rule_4,
            doc="Auxiliary constraint for variable z2",
        )

        def z3_rule_1(m, pv, ret):
            return m.z3[pv, ret] >= 0

        self.m.z3_rule_1_constr = pe.Constraint(
            self.m.Solar_pv_inputs,
            self.m.Retrofit_scenarios,
            rule=z3_rule_1,
            doc="Auxiliary constraint for variable z3",
        )

        def z3_rule_2(m, pv, ret):
            return m.z3[pv, ret] <= m.BigM * m.y_retrofit[ret]

        self.m.z3_rule_2_constr = pe.Constraint(
            self.m.Solar_pv_inputs,
            self.m.Retrofit_scenarios,
            rule=z3_rule_2,
            doc="Auxiliary constraint for variable z3",
        )

        def z3_rule_3(m, pv, ret):
            return m.Capacity[pv] - m.z3[pv, ret] >= 0

        self.m.z3_rule_3_constr = pe.Constraint(
            self.m.Solar_pv_inputs,
            self.m.Retrofit_scenarios,
            rule=z3_rule_3,
            doc="Auxiliary constraint for variable z3",
        )

        def z3_rule_4(m, pv, ret):
            return m.Capacity[pv] - m.z3[pv, ret] <= m.BigM * (1 - m.y_retrofit[ret])

        self.m.z3_rule_4_constr = pe.Constraint(
            self.m.Solar_pv_inputs,
            self.m.Retrofit_scenarios,
            rule=z3_rule_4,
            doc="Auxiliary constraint for variable z3",
        )

        def z4_rule_1(m, st, ret):
            return m.z4[st, ret] >= 0

        self.m.z4_rule_1_constr = pe.Constraint(
            self.m.Solar_th_inputs,
            self.m.Retrofit_scenarios,
            rule=z4_rule_1,
            doc="Auxiliary constraint for variable z4",
        )

        def z4_rule_2(m, st, ret):
            return m.z4[st, ret] <= m.BigM * m.y_retrofit[ret]

        self.m.z4_rule_2_constr = pe.Constraint(
            self.m.Solar_th_inputs,
            self.m.Retrofit_scenarios,
            rule=z4_rule_2,
            doc="Auxiliary constraint for variable z4",
        )

        def z4_rule_3(m, st, ret):
            return m.Capacity[st] - m.z4[st, ret] >= 0

        self.m.z4_rule_3_constr = pe.Constraint(
            self.m.Solar_th_inputs,
            self.m.Retrofit_scenarios,
            rule=z4_rule_3,
            doc="Auxiliary constraint for variable z4",
        )

        def z4_rule_4(m, st, ret):
            return m.Capacity[st] - m.z4[st, ret] <= m.BigM * (1 - m.y_retrofit[ret])

        self.m.z4_rule_4_constr = pe.Constraint(
            self.m.Solar_th_inputs,
            self.m.Retrofit_scenarios,
            rule=z4_rule_4,
            doc="Auxiliary constraint for variable z4",
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
        solver = pyomo.opt.SolverFactory("gurobi")

        def get_design_results(model_instance):
            dsgn1 = pyio.get_entity(model_instance, "Capacity")
            dsgn2 = pyio.get_entity(model_instance, "Storage_cap")
            dsgn = pd.concat([dsgn1, dsgn2])
            dsgn = pd.DataFrame(dsgn)
            dsgn = dsgn.T
            return dsgn

        def get_oper_results(model_instance):
            oper1 = pyio.get_entities(model_instance, "P").unstack()
            oper2 = pyio.get_entities(
                model_instance, ["P_export", "Qin", "Qout", "E"]
            ).unstack()
            oper = pd.merge(oper1, oper2, left_index=True, right_index=True)
            return oper

        def get_obj_results(model_instance):
            obj = pyio.get_entities(
                model_instance,
                [
                    "Total_cost",
                    "Investment_cost",
                    "Operating_cost",
                    "Income_via_exports",
                    "Total_carbon",
                ],
            )
            return obj

        if self.optim_mode == 1:

            # Cost minimization
            # -----------------
            self.results = [None]

            self.m.Carbon_obj.deactivate()
            solver.solve(self.m, tee=False, keepfiles=False)

            # Save results
            self.results[0] = self.m.clone()
            obj = get_obj_results(self.m)
            dsgn = get_design_results(self.m)
            oper = get_oper_results(self.m)
        elif self.optim_mode == 2:

            # Carbon minimization
            # -------------------
            self.results = [None]

            self.m.Carbon_obj.activate()
            self.m.Cost_obj.deactivate()
            solver.solve(self.m, tee=False, keepfiles=False)
            carb_min = pe.value(self.m.Total_carbon)

            self.m.epsilon = carb_min
            self.m.Carbon_obj.deactivate()
            self.m.Cost_obj.activate()
            solver.solve(self.m, tee=False, keepfiles=False)

            # Save results
            self.results[0] = self.m.clone()
            obj = get_obj_results(self.m)
            dsgn = get_design_results(self.m)
            oper = get_oper_results(self.m)
        else:

            self.results = [None] * (self.num_of_pfp + 2)
            oper = [None] * (self.num_of_pfp + 2)

            # Cost minimization
            # -----------------
            self.m.Carbon_obj.deactivate()
            solver.solve(self.m, tee=False, keepfiles=False)
            self.results[0] = self.m.clone()
            carb_max = pe.value(self.m.Total_carbon)

            # Save results
            self.results[0] = self.m.clone()
            obj = get_obj_results(self.m)
            dsgn = get_design_results(self.m)
            oper[0] = get_oper_results(self.m)
            oper[0].index.name = "Time"

            # Carbon minimization
            # -------------------
            self.m.Carbon_obj.activate()
            self.m.Cost_obj.deactivate()
            solver.solve(self.m, tee=False, keepfiles=False)
            carb_min = pe.value(self.m.Total_carbon)

            # Pareto points
            # -------------
            if self.num_of_pfp == 0:
                self.m.epsilon = carb_min
                self.m.Carbon_obj.deactivate()
                self.m.Cost_obj.activate()
                solver.solve(self.m, tee=False, keepfiles=False)

                # Save results
                self.results[1] = self.m.clone()
                obj = pd.concat([obj, get_obj_results(self.m)])
                dsgn = pd.concat([dsgn, get_design_results(self.m)])
                oper[1] = get_oper_results(self.m)
                oper[1].index.name = "Time"

            else:
                self.m.Carbon_obj.deactivate()
                self.m.Cost_obj.activate()

                interval = (carb_max - carb_min) / (self.num_of_pfp + 1)
                steps = list(np.arange(carb_min, carb_max, interval))
                steps.reverse()
                print(steps)

                for i in range(1, self.num_of_pfp + 1 + 1):
                    self.m.epsilon = steps[i - 1]
                    solver.solve(self.m, tee=False, keepfiles=False)

                    # Save results
                    self.results[i] = self.m.clone()
                    obj = pd.concat([obj, get_obj_results(self.m)])
                    dsgn = pd.concat([dsgn, get_design_results(self.m)])
                    oper[i] = get_oper_results(self.m)
                    oper[i].index.name = "Time"

        # Beautification
        # --------------
        obj.index.name = "Pareto front points"
        dsgn.rename(columns={"Heat": "Heat stor", "Elec": "Elec stor"}, inplace=True)
        dsgn.index.name = "Pareto front points"

        if self.optim_mode == 1:
            obj.index = ["Min_cost"]
            dsgn.index = ["Min_cost"]
            oper.index.name = "Time"
        elif self.optim_mode == 2:
            obj.index = ["Min_carb"]
            dsgn.index = ["Min_carb"]
            oper.index.name = "Time"
        else:
            obj.index = (
                ["Min_cost"]
                + ["PFP" + str(i) for i in range(2, self.num_of_pfp + 1 + 1)]
                + ["Min_carb"]
            )
            dsgn.index = (
                ["Min_cost"]
                + ["PFP" + str(i) for i in range(2, self.num_of_pfp + 1 + 1)]
                + ["Min_carb"]
            )
        return obj, dsgn, oper


if __name__ == "__main__":
    sp = EnergyHub_retrofit("Input_data_retrofit", 1, 3, 5)
    (obj, dsgn, oper) = sp.solve()
