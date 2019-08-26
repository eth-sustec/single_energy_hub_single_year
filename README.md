# Simple single-stage energy hub model

## About

This repository contains a simple implementation of a standard "energy hub" model for the optimal design and operation of a single distributed multi-energy system.

The model is built in Pyomo (http://pyomo.readthedocs.io).

## Detailed model description

The model is formulated as a Mixed-Integer Linear Programming (MILP) model for the optimal design and operation of an energy system.

Model characteristics:

* The design is also performed in one-stage i.e. we assume that the investment in the energy system happens in one stage.
* The model considers various generation and storage technologies and supplies different types of energy demands.
* For simplicity reasons, advanced constraints, such as part-load efficiencies and minimum part-loads are not included.
* Single-objective (minimum cost (investment + operation), minimum CO<sub>2</sub> emissions) and multi-objective modes are included.

## How to use the EnergyHub model:

First, import the EnergyHub class (defined in the `EnergyHub.py` file):
```
import EnergyHub as eh
```
Then, an `EnergyHub` object can be defined using the following inputs:

1. A file specifying the values of the model parameters (the model formulation is decoupled from the attribution of parameter values)
2. A parameter specifying the optimization objectives (1: total cost minimization, 2: CO<sub>2</sub> minimization, 3: multi-objective optimization)
3. A parameter specifying the number of Pareto points to be considered (only if multi-objective optimization is chosen)
  
For instance:
```python
# Create your model
mod = eh.EnergyHub('Input_data', 3, 5)
```
The final step is to solve the model and get the model results:
```python
# Solve the model and get the results
(obj, dsgn, oper) = mod.solve()
```
The key objective function, design and operation results are returned by the model using three objects:
1. obj: Contains the total cost, cost breakdown, and total carbon results. It is a data frame for all optim_mode settings.
2. dsgn: Contains the generation and storage capacities of all candidate technologies. It is a data frame for all optim_mode settings.
3. oper: Contains the generation, export and storage energy flows for all time steps considered. It is a single dataframe when optim_mode is 1 or 2 (single-objective) and a list of dataframes for each Pareto point when optim_mode is set to 3 (multi-objective).
