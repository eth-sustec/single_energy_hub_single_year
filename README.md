# Simple single-stage energy hub model

## About

This repository contains a simple implementation of a standard "energy hub" model for the optimal design and operation of a single decentralized multi-energy system (D-MES) considering also building retrofit options.

The model is built in Pyomo (http://pyomo.readthedocs.io).

## Detailed model description

The model is formulated as a Mixed-Integer Linear Programming (MILP) model for the optimal design and operation of an energy system.

Model characteristics:

* The design is also performed in one-stage i.e. we assume that the investment in the energy system happens in one stage.
* The model considers various generation and storage technologies to compose the D-MES.
* The model is also capable of optimizing the selection of the optimal retrofit scenario for the building(s), which in turns determines the energy demands that the D-MES will need to satisfy. 
* For simplicity reasons, advanced constraints, such as part-load efficiencies and minimum part-loads are not included.
* Single-objective (minimum cost (investment + operation), minimum CO<sub>2</sub> emissions) and multi-objective modes are included.

## How to use the EnergyHub model:

First, import the EnergyHub class (defined in the `EnergyHub.py` file):
```
import EnergyHubRetrofit as ehr
```
Then, an `EnergyHubRetrofit` object can be defined using the following inputs:

1. A dictionary that holds all the values for the model parameters (the model formulation is decoupled from the attribution of parameter values)
2. A parameter specifying the temporal resolution of the model (1: typical days optimization, 2: full yearly horizon optimization (8760 hours), 3: typical days with continuous storage state-of-charge)
3. A parameter specifying the optimization objectives (1: total cost minimization, 2: CO<sub>2</sub> minimization, 3: multi-objective optimization)
4. A parameter specifying the number of Pareto points to be considered (only if multi-objective optimization is chosen)
  
For instance:
```python
# Create your model
mod = ehr.EnergyHubRetrofit(ehr_inp, 1, 3, 5) # Initialize the model
```
The final step is to solve the model and get the model results:
```python
# Solve the model and get the results
mod.solve()
```
The key outputs of the model are:
1. A JSON file containing key model details (number of variables, constraints etc.) and the values of all model variables
2. An Excel file containing all values of model variables in a separate worksheet per variable
3. A pickle file containing a list (for each Pareto point) of dictionaries, with each dictionary's key value pair corresponding to the variable's name and optimal values.