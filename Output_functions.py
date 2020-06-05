import pandas as pd
import pyomoio as pyio
from pyomo.opt import SolverResults
import pickle as pkl

def get_design_results(model_instance):
    """
    Gets a Pyomo model_instance as input and returns a Pandas Series with all the results regarding the design of the energy system
    """
    dsgn1 = pyio.get_entity(model_instance, "Conv_cap")
    dsgn2 = pyio.get_entity(model_instance, "Storage_cap")
    dsgn = pd.concat([dsgn1, dsgn2])
    dsgn = pd.DataFrame(dsgn)
    dsgn = dsgn.T
    return dsgn

def get_oper_results(model_instance):
    """
    Gets a Pyomo model_instance as input and returns a Pandas DataFrame with all the results regarding the operation of the energy system
    """
    oper1 = pyio.get_entities(model_instance, "P_conv").unstack()
    oper2 = pyio.get_entities(
        model_instance, ["P_import", "P_conv", "P_export", "Qin", "Qout", "SoC"]
    ).unstack()
    oper = pd.merge(oper1, oper2, left_index=True, right_index=True)
    return oper

def get_obj_results(model_instance):
    """
    Gets a Pyomo model_instance as input and returns a Pandas Series with all the results regarding the objective functions and their components
    """
    obj = pyio.get_entities(
        model_instance,
        [
            "Total_cost",
            "Investment_cost",
            "Import_cost",
            "Export_profit",
            "Total_carbon",
        ],
    )
    return obj

def get_all_vars(model_instance):
    var_list = [
        "y_conv",
        "Conv_cap",
        "y_stor",
        "Storage_cap"
        "P_import",
        "P_conv",
        "P_export",
        "P_exchange",
        "Qin",
        "Qout",
        "SoC",
        "Total_cost",
        "Investment_cost",
        "Import_cost",
        "Export_profit",
        "Total_carbon",
    ]

    res = dict()

    for i in range(len(var_list)):
        v = getattr(model_instance, var_list[i])
        d = v.extract_values()
        res[var_list[i]] = pd.DataFrame(d, index=["Value"]).T

    return res

def pickle_solver_results(model_instance, filename):
    myResults = SolverResults()
    model_instance.solutions.store_to(myResults)
    with open(filename, mode='wb') as file:
        pkl.dump(myResults, file)
    file.close()