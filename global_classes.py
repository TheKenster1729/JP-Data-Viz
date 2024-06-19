import pandas as pd
import numpy as np
import random
import json
from sql_utils import SQLConnection, DataRetrieval

NUM_SAMPLES_PER_TIMESTEP = 400
SAMPLE_IDS = list(range(1, NUM_SAMPLES_PER_TIMESTEP + 1))
SAMPLE_ID_COLUMN = "Run #"
DATA_COLUMN = "Value"
TIME_COLUMN = "Year"

class GlobalVariables:
    def __init__(self, number_of_samples_per_timestep = NUM_SAMPLES_PER_TIMESTEP, sample_ids = SAMPLE_IDS, sample_id_column = SAMPLE_ID_COLUMN,
                 data_column = DATA_COLUMN, time_column = TIME_COLUMN) -> None:
        self.global_number_of_samples_per_timestep = number_of_samples_per_timestep
        self.global_sample_ids = sample_ids
        self.global_sample_id_column = sample_id_column
        self.global_data_column = data_column
        self.global_time_column = time_column

class VariableOutput(GlobalVariables):
    def __init__(self, original_name, display_name, region, scenario, df, year = None) -> None:
        super().__init__()
        self.original_name = original_name
        self.display_name = display_name
        self.region = region
        self.scenario = scenario
        self.df = df
        self.year = year
        self.sample_ids_set = set(self.df[self.global_sample_id_column])

    def __add__(self, other):
        other_sample_ids = other.sample_ids_set
        df_to_return = self.df.copy().drop(columns=[self.global_data_column])

        if self.sample_ids_set == other_sample_ids:
            if self.year:
                df_to_return[self.global_data_column] = self.df[self.global_data_column] + other.df[self.global_data_column]
            else:
                # adding up over all years, so make sure dataframe is sorted properly - year as the top sort and run number as the next level
                df_to_return = df_to_return.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                other_df = other.df.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                df_to_return[self.global_data_column] = self.df[self.global_data_column] + other_df[self.global_data_column]
            return df_to_return
        else:
            sample_ids_to_use = self.sample_ids_set.intersection(other_sample_ids)
            if self.year:
                df_to_return[self.global_data_column] = self.df[self.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column] + other.df[other.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column]
            else:
                # adding up over all years, so make sure dataframe is sorted properly - year as the top sort and run number as the next level
                df_to_return = df_to_return.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                other_df = other.df.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                df_to_return[self.global_data_column] = self.df[self.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column] + other.df[other.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column]
            return df_to_return

    def __sub__(self, other):
        other_sample_ids = other.sample_ids_set
        df_to_return = self.df.copy().drop(columns=[self.global_data_column])

        if self.sample_ids_set == other_sample_ids:
            if self.year:
                df_to_return[self.global_data_column] = self.df[self.global_data_column] - other.df[self.global_data_column]
            else:
                # adding up over all years, so make sure dataframe is sorted properly - year as the top sort and run number as the next level
                df_to_return = df_to_return.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                other_df = other.df.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                df_to_return[self.global_data_column] = self.df[self.global_data_column] - other_df[self.global_data_column]
            return df_to_return
        else:
            sample_ids_to_use = self.sample_ids_set.intersection(other_sample_ids)
            if self.year:
                df_to_return[self.global_data_column] = self.df[self.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column] - other.df[other.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column]
            else:
                # adding up over all years, so make sure dataframe is sorted properly - year as the top sort and run number as the next level
                df_to_return = df_to_return.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                other_df = other.df.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
                df_to_return[self.global_data_column] = self.df[self.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column] - other.df[other.df[self.global_sample_id_column].isin(sample_ids_to_use)][self.global_data_column]
            return df_to_return

    def __mul__(self, other):
        if self.sample_ids_set == other.sample_ids_set:
            df_to_return = self.df.copy().drop(columns=[self.global_data_column])
            df_to_return[self.global_data_column] = self.df[self.global_data_column] * other.df[self.global_data_column]

            return df_to_return

        else:
            # first take into account crashed runs
            sample_ids_to_use = self.sample_ids_set.intersection(other.sample_ids_set)
            self_values_to_use = self.df[self.df[self.global_sample_id_column].isin(sample_ids_to_use)]
            other_values_to_use = other.df[other.df[self.global_sample_id_column].isin(sample_ids_to_use)]

            df_to_return = self.df.copy().drop(columns=[self.global_data_column])
            df_to_return[self.global_data_column] = self_values_to_use[self.global_data_column] * other_values_to_use[self.global_data_column]

            return df_to_return
        
    def __truediv__(self, other):
        if self.sample_ids_set == other.sample_ids_set:
            # no crashed runs or other funny business, so now check for potential division by zero
            zero_values_other = other.df[other.df[self.global_data_column] == 0].index
            self_values_to_use = self.df[~self.df.index.isin(zero_values_other)]
            other_values_to_use = other.df[~other.df.index.isin(zero_values_other)]

            df_to_return = self.df.copy().drop(columns=[self.global_data_column]).drop(index=zero_values_other)
            df_to_return[self.global_data_column] = self_values_to_use[self.global_data_column] / other_values_to_use[self.global_data_column]

            return df_to_return

        else:
            # first take into account crashed runs
            sample_ids_to_use = self.sample_ids_set.intersection(other.sample_ids_set)
            self_values_to_use = self.df[self.df[self.global_sample_id_column].isin(sample_ids_to_use)]
            other_values_to_use = other.df[other.df[self.global_sample_id_column].isin(sample_ids_to_use)]

            # now take into account division by zero
            zero_values_other = other_values_to_use[other_values_to_use[self.global_data_column] == 0].index
            self_values_to_use = self_values_to_use[~self_values_to_use.index.isin(zero_values_other)]
            other_values_to_use = other_values_to_use[~other_values_to_use.index.isin(zero_values_other)]

            df_to_return = self.df.copy().drop(columns=[self.global_data_column])
            df_to_return[self.global_data_column] = self_values_to_use[self.global_data_column] / other_values_to_use[self.global_data_column]

            return df_to_return

class Region:
    pass

class Scenario:
    pass

class OutputOperations:
    pass

class DivideOutputs:
    def __init__(self, output1, output2) -> None:
        self.output1 = output1
        self.output2 = output2

    def consistency_check(self):
        # basic check: make sure the two have the same run numbers, and if they don't, only use the run numbers that are in both
        sample_id_condition = self.output1.sample_ids == self.output2.sample_ids
        if not sample_id_condition:
            samples_in_common = set(self.output1.sample_ids).intersection(self.output2.sample_ids)
            self.output1.sample_ids = list(samples_in_common)
            self.output2.sample_ids = list(samples_in_common)
        assert self.output1.original_name == self.output2.original_name
        assert self.output1.display_name == self.output2.display_name
        assert self.output1.number_of_samples_per_time == self.output2.number_of_samples_per_time
        assert self.output1.sample_ids == self.output2.sample_ids

class SumOutputs(GlobalVariables):
    def __init__(self, outputs: list) -> None:
        super().__init__()
        self.outputs = outputs

    def find_valid_run_numbers(self):
        self.run_numbers = [output.sample_ids_set for output in self.outputs]
        self.valid_run_numbers = set.intersection(*self.run_numbers)

    def sum_outputs(self):
        self.find_valid_run_numbers()
        result_df = self.outputs[0].df[self.outputs[0].df[self.global_sample_id_column].isin(self.valid_run_numbers)].copy()
        # ensure the dataframe is sorted properly - year as the top sort and run number as the next level
        result_df = result_df.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
        for output in self.outputs[1:]:
            output_data_to_use = output.df[output.df[self.global_sample_id_column].isin(self.valid_run_numbers)]
            output_data_to_use = output_data_to_use.sort_values(by=[self.global_time_column, self.global_sample_id_column], ascending=[True, True])
            result_df[self.global_data_column] = result_df[self.global_data_column] + output_data_to_use[self.global_data_column]
        return result_df

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.dataframe = None

    def add_child(self, child):
        self.children.append(child)

    def add_dataframe(self, df):
        self.dataframe = df

class SyntheticData(GlobalVariables):
    def __init__(self, n_regions = None, n_scenarios = None, n_outputs = None, n_runs = None, years = None) -> None:
        super().__init__()
        if not n_regions:
            self.regions = ["Region " + str(i) for i in range(1, 19)]
        else:
            self.regions = n_regions

        if not n_scenarios:
            self.scenarios = ["Scenario " + str(i) for i in range(1, 13)]
        else:
            self.scenarios = n_scenarios

        if not n_outputs:
            self.outputs = ["Output " + str(i) for i in range(1, 10)]
        else:
            self.outputs = n_outputs

        if not n_runs:
            self.n_runs = list(range(1, self.global_number_of_samples_per_timestep + 1))
        else:
            self.n_runs = n_runs

        if not years:
            self.years = list(range(2020, 2105, 5))
        else:
            self.years = years

        self.tree_data_structure = self.create_tree_data_structure()
        self.dataset = []

    def create_data(self):
        distributions = ["normal", "powerlaw", "gamma"]

        # randomly select among distributions
        dist = random.choice(distributions)
        if dist == "normal":
            samples = np.random.normal(100, 15, size = len(self.n_runs))
        elif dist == "powerlaw":
            samples = np.random.power(10, size = len(self.n_runs))*100
        elif dist == "gamma":
            samples = np.random.gamma(2, 10, size = len(self.n_runs))*100

        return samples

    def create_tree_data_structure(self):
        root = TreeNode("Root")
        for output in self.outputs:
            output_node = TreeNode(output)
            root.add_child(output_node)
            for scenario in self.scenarios:
                scenario_node = TreeNode(scenario)
                output_node.add_child(scenario_node)
                for region in self.regions:                    
                    region_node = TreeNode(region)
                    scenario_node.add_child(region_node)
                    df = pd.DataFrame()
                    for year in self.years:
                        df_to_concat = pd.DataFrame()
                        samples = self.create_data()
                        df_to_concat[self.global_sample_id_column] = self.n_runs
                        df_to_concat[self.global_data_column] = samples
                        df_to_concat["Year"] = [year]*len(samples)
                        df = pd.concat([df, df_to_concat], ignore_index=True)
                    region_node.add_dataframe(df)

        return root

    def export_entire_tree_to_json(self, root, filename):

        def serialize_node(node):
            # This function will recursively serialize each node and its children
            node_dict = {
                "name": node.name,
                "children": [serialize_node(child) for child in node.children]
            }
            if node.dataframe is not None:
                # Convert the DataFrame to a dictionary
                node_dict["dataframe"] = node.dataframe.to_dict(orient='records')
            return node_dict

        # Serialize the entire tree starting from the root
        tree_dict = serialize_node(root)
        
        # Convert the tree dictionary to a JSON string
        json_string = json.dumps(tree_dict, indent=4)
        
        # Write JSON string to a file
        with open(filename, "w") as json_file:
            json_file.write(json_string)

    def load_tree_from_json(self):
        pass

    def bfs(self, root, output, region, scenario):
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.name == output:
                for child in node.children:
                    if child.name == scenario:
                        for grandchild in child.children:
                            if grandchild.name == region:
                                return grandchild
            for child in node.children:
                queue.append(child)
        return None

    def get_dataframe(self, output, region, scenario, year = None):
        node = self.bfs(self.tree_data_structure, output, region, scenario)
        if year:
            return node.dataframe.loc[node.dataframe["Year"] == year]
        else:
            return node.dataframe

if __name__ == "__main__":
    # # test operations
    db = SQLConnection("all_data_jan_2024")
    df1 = DataRetrieval(db, "elec_prod_Renewables_TWh_pol", "GLB", "2C_med").single_output_df()
    output1 = VariableOutput("elec_prod_Renewables_TWh_pol", "Renewable", "GLB", "2C_med", df1)
    df2 = DataRetrieval(db, "elec_prod_Biomass_CCS_TWh_pol", "GLB", "2C_med").single_output_df()
    output2 = VariableOutput("elec_prod_Biomass_CCS_TWh_pol", "Biomass", "GLB", "2C_med", df2)
    # df3 = DataRetrieval(db, "emissions_CO2eq_total_million_ton_CO2eq", "CHN", "Ref").single_output_df()
    # output3 = VariableOutput("emissions_CO2eq_total_million_ton_CO2eq", "CO2 Emissions", "CHN", "Ref", df3)
    # sum_outputs = SumOutputs([output1, output2, output3]).sum_outputs()
    print(df2)
    renew_share = output1 / output2
    print(renew_share)

    # s = SyntheticData()
    # df1 = s.get_dataframe("Output 1", "Region 1", "Scenario 1")
    # output1 = VariableOutput("Output 1", "Output 1", "Region 1", "Scenario 1", df1)
    # df2 = s.get_dataframe("Output 1", "Region 1", "Scenario 1")
    # output2 = VariableOutput("Output 1", "Output 1", "Region 1", "Scenario 1", df2)