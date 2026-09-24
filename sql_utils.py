"""Backwards-compatible entry points over the normalized schema.

SQLConnection, DataRetrieval and MultiOutputRetrieval keep the constructor
signatures and return shapes they have always had. Everything underneath them
now goes through data.repository, which reads series_values instead of the
thousands of randomly-named per-series tables the old code walked through
name_mappings.

Around thirty call sites in app.py, figure.py and analysis.py still use these
three names; they are migrated to the repository separately. DatabaseModification
is the legacy Excel ingest and is unchanged: it still writes the old layout, and its
__main__ call is deliberately commented out. New data goes through
scripts/ingest_excel.py, which writes series_values directly.
"""

import os
from itertools import product

import mysql.connector
import pandas as pd
import numpy as np
from sqlalchemy import Float, Integer, text
from random import choices

import config
from data.repository import ordinal, repository
from styling import Options


class SQLConnection:
    """A named database plus the repository bound to it.

    The engine and raw connector are kept because legacy tooling may still open
    a raw connection to the database. Both are created on first use.
    """

    def __init__(self, dbname, pool_size=10, max_overflow=20):
        self.dbname = dbname
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._engine = None
        self._cursor = None

    @property
    def repository(self):
        return repository(self.dbname)

    @property
    def retrieval_engine(self):
        return config.engine(self.dbname, self.pool_size, self.max_overflow)

    @property
    def engine(self):
        if self._engine is None:
            self._engine = mysql.connector.connect(**config.connector_kwargs(self.dbname))
        return self._engine

    @property
    def cursor(self):
        if self._cursor is None:
            self._cursor = self.engine.cursor()
        return self._cursor


class CustomVariable:
    # Dead: never had a body and is not referenced. Superseded by
    # data.expressions. Left in place pending a decision to remove it.
    def __init__(self) -> None:
        pass

class DatabaseModification(SQLConnection):
    def __init__(self, dbname, path_to_scenarios = r"Raw Data\Scenarios", scenarios = "all", files = "all"):
        super().__init__(dbname)
        self.path_to_scenarios = path_to_scenarios
        if scenarios == "all":
            self.scenarios = list(os.listdir(path_to_scenarios))
            self.scenarios.remove(".DS_Store")
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

    def determine_column_range(self, df):
        first_year_column = list(df.columns).index("2020")
        alphanumeric = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}

        return alphanumeric[first_year_column - 1] + ":" + alphanumeric[len(df.columns) - 1]

    def main(self):
        count = 1
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
                            starting_column = self.determine_column_range(pd.read_excel(file_path, sheet_name = region))
                            df = pd.read_excel(file_path, sheet_name = region, usecols = starting_column + ":S")
                            df = df.rename(columns = {df.columns[0]: "Run #"})
                            df_to_use = df.melt(id_vars = "Run #", value_name = "Value", var_name = "Year")
                        else:
                            if filename in self.files:
                                starting_column = self.determine_column_range(pd.read_excel(file_path, sheet_name = region))
                                df = pd.read_excel(file_path, sheet_name = region, usecols = starting_column + ":S")
                                df = df.rename(columns = {df.columns[0]: "Run #"})
                                df_to_use = df.melt(id_vars = "Run #", value_name = "Value", var_name = "Year")

                        # need to process Eps - not float
                        df_to_use["Value"] = df_to_use["Value"].replace("Eps", 0)

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

            print(f"Finished scenario {folder} ({count} of {len(self.scenarios)})")
            count += 1

class MultiOutputRetrieval:
    def __init__(self, db_connection_obj, outputs, region, scenario, year = None):
        self.db = db_connection_obj
        self.outputs = outputs
        self.region = region
        self.scenario = scenario
        self.year = year

    def construct_df(self):
        self.df = self.db.repository.multi_output(
            self.outputs, self.region, self.scenario, self.year)
        return self.df

class DataRetrieval:
    def __init__(self, db_connection_obj, output, region, scenario, year = None):
        self.db = db_connection_obj
        self.output = output
        self.region = region
        self.scenario = scenario
        self.year = year

    @property
    def repository(self):
        return self.db.repository

    def number_to_ordinal(self, n):
        """
        Convert an integer from 1 to 100 into its English ordinal representation.

        Args:
        n (int): Integer from 1 to 100

        Returns:
        str: The ordinal representation of n
        """
        return ordinal(n)

    def get_long_name(self, name = None):
        if not name: # this is the default
            name = self.output
        return name + "_" + self.region + "_" + self.scenario

    def parse_custom_vars_division(self, parsed_string):
        # Dead: lowercases the region and scenario, so the name_mappings lookup
        # it builds can never match a key. Superseded by data.expressions.
        # Left in place pending a decision to remove it.
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

        return df

    def parse_custom_vars_addition(self):
        # Dead: never had a body. Superseded by data.expressions.Addition.
        pass

    def single_output_df(self):
        return self.repository.series(self.output, self.region, self.scenario, self.year)

    def single_output_df_to_graph(self, lower_bound, upper_bound):
        return self.repository.percentile_band(
            self.output, self.region, self.scenario, lower_bound, upper_bound, self.year)

    def output_df(self, output, regions, scenarios):
        # Dead: get_long_name takes one argument, not three, and
        # single_output_df takes none. This cannot ever have run.
        # Left in place pending a decision to remove it.
        combinations = product(regions, scenarios)

        df_to_return = pd.DataFrame()
        for combo in combinations:
            long_name = self.get_long_name(output, combo[0], combo[1])
            df = self.single_output_df(long_name)
        #     df = pd.read_sql_table(table, con = self.engine).drop(columns = "index")
        #     df_to_return = pd.concat([df_to_return, df], ignore_index = True)

        # return df_to_return

    def mapping_df(self):
        if self.year is None:
            # The old implementation filtered the full series with
            # query("Year==@self.year"), which matches nothing when year is
            # None. Preserved so callers see the same empty frame.
            return self.single_output_df().iloc[0:0]
        return self.single_output_df()

    def choropleth_map_df(self, lower_bound, upper_bound, max_workers=8):
        """
        All regions except GLB, in one query.

        max_workers is accepted and ignored: the old implementation fanned one
        query out per region across a thread pool, and a single query bound by
        region_id IN (...) is faster than any number of those.
        """
        return self.repository.choropleth(
            self.output, Options().region_names[1:], self.scenario, self.year,
            lower_bound, upper_bound)

class DatabaseModificationForNewStructure(SQLConnection):
    # Dead: an empty stub. The migration it describes is
    # scripts/migrate_schema.py. Left in place pending a decision to remove it.
    def __init__(self, dbname, path_to_scenarios = r"Raw Data\Scenarios"):
        super().__init__(dbname)
        self.path_to_scenarios = path_to_scenarios

    def main(self):
        pass

if __name__ == "__main__":
    # Leave this commented. Running it re-ingests every Excel file as brand new
    # randomly-named tables and appends duplicate rows to name_mappings.
    # DatabaseModification("publication", path_to_scenarios = r"Raw Data/Archive").main()
    pass
