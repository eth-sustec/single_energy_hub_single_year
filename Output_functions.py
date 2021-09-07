import pandas as pd
# from pyomo.opt import SolverResults
# import pickle as pkl
import pyomo.environ as pe

def get_all_vars(model_instance):
    # List of all variable names in model_instance
    var_list = [
        i.name for i in list(model_instance.component_objects(pe.Var, active=True))
    ]

    res = dict()

    for i in range(len(var_list)):
        v = getattr(model_instance, var_list[i])
        d = v.extract_values()
        res[var_list[i]] = pd.DataFrame(d, index=["Value"]).T

    return res

def write_all_vars_to_excel(all_vars, filename):
    writer = pd.ExcelWriter(filename + ".xlsx", engine="openpyxl")
    for key in all_vars:
        all_vars[key].to_excel(
            writer, sheet_name=key[0 : min(len(key), 31)], merge_cells=False
        )
    writer.save()