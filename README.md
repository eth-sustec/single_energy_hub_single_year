# Simple single-stage energy hub model

## About

This repository contains a simple implementation of a standard "energy hub" model for the optimal design and operation of a single distributed multi-energy system.

The model is built in Pyomo (http://pyomo.readthedocs.io).

## Detailed model description

This repository contains a Mixed-Integer Linear Programming (MILP) model for the optimal design and operation of an energy system.

The design is also performed in one-stage i.e. we assume that the investment in the energy system happens in one stage.

The formulation of the MILP model itself is kept by choice rather simple and advanced constraints, such as part-load efficiencies and minimum part-loads have not been considered in order to keep the computational requirements of the model low.

The model considers various generation and storage technologies and supplies different types of energy demands.

Single-objective (minimum cost (investment + operation), minimum CO<sub>2</sub> emissions) and multi-objective modes are included.

## How to use the EnergyHub model:

An `EnergyHub` class is defined in the `EnergyHub.py` file. The class takes the following inputs:

1. A file specifying the values of the model parameters.
2. A parameter specifying 

```
import EnergyHub as eh

# Create your model
mod = eh.EnergyHub('Input_data', 3, 5)

# Solve the model and get the results
(obj, dsgn, oper) = mod.solve()
```
