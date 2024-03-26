import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table
import os
from anytree.importer import DictImporter
from anytree import RenderTree
import json
from anytree.search import findall
from itertools import product
from styling import Options
import sqlalchemy
from sqlalchemy import Integer, Float
import mysql.connector
from random import choices

class SQLConnection:
    def __init__(self, dbname):
        self.dbname = dbname
        self.retrieval_engine = create_engine('mysql+mysqlconnector://root:password@localhost:3306/{}'.format(self.dbname))
        self.engine = mysql.connector.connect(
                host = "localhost",
                user = "root",
                password = "password",
                database = "all_data_jan_2024"
            )
        self.cursor = self.engine.cursor()

class CustomVariable:
    def __init__(self) -> None:
        pass

class DatabaseModification(SQLConnection):
    def __init__(self, dbname, path_to_scenarios = r"Raw Data\Scenarios", scenarios = "all", files = "all"):
        super().__init__(dbname)
        self.path_to_scenarios = path_to_scenarios
        if scenarios == "all":
            self.scenarios = os.listdir(path_to_scenarios)
        else: # expects a list of scenario folders
            self.scenarios = scenarios
        self.files = files

    def name_table(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'

        c = choices(alphabet, k = 20)
        return ''.join(c)

    def update_name_mapping_table(self, full_output_name, assigned_name):
        mapping_df = pd.DataFrame({
            "Full Output Name": [full_output_name],
            "Assigned Name": [assigned_name]
        })
        mapping_df.to_sql(name = 'name_mappings', con = self.retrieval_engine, if_exists = 'append')
        print(f"Updated name mapping: {full_output_name} -> {assigned_name}")

    def main(self):
        for folder in self.scenarios:
            path_to_this_folder = os.path.join(self.path_to_scenarios, folder)
            for filename in os.listdir(path_to_this_folder):
                if filename.endswith('.xlsx') or filename.endswith('.xls'):
                    # Define the table name based on the filename (without the extension)
                    for region in Options().region_names:
                        table_name = self.name_table()

                        # Load the Excel file into a Pandas DataFrame
                        file_path = os.path.join(path_to_this_folder, filename)
                        if self.files == "all":
                            df = pd.read_excel(file_path, sheet_name = region, usecols = "B:S").rename(columns = {"Unnamed: 1": "Run #"})
                            df_to_use = df.melt(id_vars = "Run #", value_name = "Value", var_name = "Year")
                        else:
                            if filename in self.files:
                                df = pd.read_excel(file_path, sheet_name = region, usecols = "B:S").rename(columns = {"Unnamed: 1": "Run #"})
                                df_to_use = df.melt(id_vars = "Run #", value_name = "Value", var_name = "Year")

                                cleaned_spreadsheet_name = '_'.join(filename.split('.')[0].split('_')[1:])
                                folder_no_period = folder.replace('.', '')
                                full_output_name = cleaned_spreadsheet_name[:-len(folder_no_period)] + region + "_" + cleaned_spreadsheet_name[-len(folder_no_period):]

                                sql_dtypes = {
                                    "Run #": Integer,
                                    "Year": Integer,
                                    "Value": Float
                                }
                                df_to_use.to_sql(name = table_name, con = self.retrieval_engine, if_exists = 'replace', index = False, dtype = sql_dtypes)
                                self.update_name_mapping_table(full_output_name, table_name)
                                print(f"Loaded {filename} into {table_name} table in the database.")

class DataRetrieval:
    def __init__(self, db_connection_obj, output, region, scenario, year = None):
        self.db = db_connection_obj
        self.output = output
        self.region = region
        self.scenario = scenario
        self.year = year

    def number_to_ordinal(self, n):
        """
        Convert an integer from 1 to 100 into its English ordinal representation.
        
        Args:
        n (int): Integer from 1 to 100
        
        Returns:
        str: The ordinal representation of n
        """
        if 11 <= n <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return str(n) + suffix

    def get_long_name(self):
        return self.output + "_" + self.region.lower() + "_" + self.scenario.lower()

    def single_output_df(self):
        if self.output not in Options().outputs:
            # indicates this call is for a custom variable, handle accordingly
            # there is a much better way, probably a few but this will do for now
            parsed_string = self.output.split("-")
            output1, output2 = parsed_string[0], parsed_string[2]
            long_name_1 = output1 + "_" + self.region.lower() + "_" + self.scenario.lower()
            long_name_2 = output2 + "_" + self.region.lower() + "_" + self.scenario.lower()
            df1_query = text("SELECT `Assigned Name` FROM name_mappings WHERE `Full Output Name`=:long_name_1")
            df2_query = text("SELECT `Assigned Name` FROM name_mappings WHERE `Full Output Name`=:long_name_2")

            with self.db.retrieval_engine.connect() as conn:
                sql_table_name_1 = conn.execute(df1_query, parameters = {"long_name_1": long_name_1}).fetchall()[0][0]
                sql_table_name_2 = conn.execute(df2_query, parameters = {"long_name_2": long_name_2}).fetchall()[0][0]
            
            try:
                df1 = pd.read_sql_table(sql_table_name_1, con = self.db.retrieval_engine).drop(columns = "index_name")
            except KeyError:
                df1 = pd.read_sql_table(sql_table_name_1, con = self.db.retrieval_engine)

            try:
                df2 = pd.read_sql_table(sql_table_name_2, con = self.db.retrieval_engine).drop(columns = "index_name")
            except KeyError:
                df2 = pd.read_sql_table(sql_table_name_2, con = self.db.retrieval_engine)

            df1 = df1.dropna()
            df2 = df2.dropna()

            df1["Value"] = df1["Value"].replace("Eps", 0)
            df2["Value"] = df2["Value"].replace("Eps", 0)

            if parsed_string[1] == "division":
                assert len(df1) == len(df2)
                df = df1.copy()
                df["Value"] = df["Value"].div(df2["Value"])
                df = df.replace([np.inf, -np.inf], np.nan).dropna()

        else:
            long_name = self.get_long_name()
            query = text("SELECT `Assigned Name` FROM name_mappings WHERE `Full Output Name`=:long_name")

            with self.db.retrieval_engine.connect() as conn:
                sql_table_name = conn.execute(query, parameters = {"long_name": long_name}).fetchall()[0][0]

            # a new table implementation resulted in some tables not having the index_name column
            try:
                df = pd.read_sql_table(sql_table_name, con = self.db.retrieval_engine).drop(columns = "index_name")
            except KeyError:
                df = pd.read_sql_table(sql_table_name, con = self.db.retrieval_engine)
            
            df = df.dropna()
            if self.output == "percapita_consumption_loss_percent":
                df["Value"] = df["Value"]*100

            df["Value"] = df["Value"].replace("Eps", 0)

        return df

    def single_output_df_to_graph(self, lower_bound, upper_bound):
        df = self.single_output_df()
        df_to_graph = df.groupby(["Year"])["Value"].agg([
            lambda x: np.percentile(x, lower_bound),
            np.median,
            lambda x: np.percentile(x, upper_bound)
            ]
        )
        df_to_graph.columns = ['{} Percentile'.format(self.number_to_ordinal(lower_bound)), 'Median', '{} Percentile'.format(self.number_to_ordinal(upper_bound))]

        return df_to_graph

    def output_df(self, output, regions, scenarios):
        combinations = product(regions, scenarios)

        df_to_return = pd.DataFrame()
        for combo in combinations:
            long_name = self.get_long_name(output, combo[0], combo[1])
            df = self.single_output_df(long_name)  
        #     df = pd.read_sql_table(table, con = self.engine).drop(columns = "index")
        #     df_to_return = pd.concat([df_to_return, df], ignore_index = True)

        # return df_to_return

    def mapping_df(self):
        return self.single_output_df().query("Year==@self.year")

    def choropleth_map_df(self, lower_bound, upper_bound):
        df_to_return = pd.DataFrame(columns = ['Region', '{} Percentile'.format(self.number_to_ordinal(lower_bound)), 'Median', '{} Percentile'.format(self.number_to_ordinal(upper_bound))])
        for region in Options().region_names[1:]:
            self.region = region
            regional_result = self.single_output_df_to_graph(lower_bound, upper_bound).loc[self.year]
            easy_to_use_data = [self.region] + list(regional_result.values)
            regional_result_reshaped = pd.DataFrame(data = {'Region': [self.region], '{} Percentile'.format(self.number_to_ordinal(lower_bound)): [easy_to_use_data[1]],
                                                            'Median': [easy_to_use_data[2]], '{} Percentile'.format(self.number_to_ordinal(upper_bound)): [easy_to_use_data[3]]})
            df_to_return = pd.concat([df_to_return, regional_result_reshaped])

        return df_to_return.reset_index(drop = True)
    
'''
import os
import pandas as pd
from styling import Options
from sql_utils import SQLConnection
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
from pprint import pprint

folders = ["2C", "Ref"]
root = r"Cleaned Data"

connection = SQLConnection("jp_data")
filename_dict = Options().filenames_to_sql_tables

outputs = [x[:-4] for x in os.listdir(os.path.join(root, "2C"))]
regions = Options().region_names
scenarios = ["2C", "Ref"]
root_node = Node("jp_data_tree")
for output in outputs:
    output_node = Node("{}".format(filename_dict[output]), parent = root_node)
    for region in regions:
        region_node = Node("{}".format(region), parent = output_node)
        for scenario in scenarios:
            scenario_node = Node("{}".format(scenario), parent = region_node, sql_table_name = filename_dict[output] + "_{}_{}".format(region.lower(), scenario.lower()))
'''
if __name__ == "__main__":
    db = SQLConnection("all_data_jan_2024")
    # print(DataRetrieval(db, "sectoral_output_Electricity_billion_USD2007", "GLB", "Ref", 2050).input_output_mapping_df())
    # SQLConnection("jp_data").input_output_mapping_df("sectoral_output_Electricity_billion_USD2007", "USA", "2C", 2050)
    # DataRetrieval(db, "sectoral_output_Electricity_billion_USD2007", "GLB", "15C_med", 2050).choropleth_map_df(5, 95)
    # for output in Options().outputs:
    #     df = DataRetrieval(db, output, "GLB", "15C_med", 2050).input_output_mapping_df()
    #     print(len(df) > 350)

    print(list(Options().outputs).pop("emissions_CO2eq_total_million_ton_CO2eq"))
    # DatabaseModification("all_data_jan_2024", scenarios = ["About1.5C_pes", "15C_med", "2C_pes"], files = ["1_percapita_consumption_loss_percent_About15C_pes.xlsx", "1_percapita_consumption_loss_percent_15C_med.xlsx", "1_percapita_consumption_loss_percent_2C_pes.xlsx"]).main()