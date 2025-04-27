import pandas as pd
import os
from styling import Readability, Options
from global_classes import VariableOutput

# var1 = VariableOutput("gdp", "GDP", "GLB", "ref", pd.DataFrame(data = {"Run #": [1, 2, 3, 4, 5]}))
# var2 = VariableOutput("GDP", "GDP", "GLB", "2c", pd.DataFrame(data = {"Run #": [1, 2, 3, 4, 5]}))

# l = [var1, var2]

# var1_copy = VariableOutput("gdp", "GDP", "GLB", "ref", pd.DataFrame(data = {"Run #": [1, 2, 3, 4, 5]}))

# print(var1_copy in l)
print("total_emissions_CO2eq_million_ton_CO2eq" in Options().all_outputs)