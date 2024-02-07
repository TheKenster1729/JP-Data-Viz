import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from anytree.importer import DictImporter
from anytree import RenderTree
import json
from anytree.search import findall
from itertools import product
from styling import Options
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy

class SQLConnection:
    def __init__(self, dbname, engine = None, use_cloud_db = False):
        self.dbname = dbname
        if engine:
            self.engine = engine
        else:
            if not use_cloud_db:
                self.engine = create_engine('mysql+mysqlconnector://root:password@localhost:3306/{}'.format(self.dbname))
            else:
                # initialize Connector object
                connector = Connector()

                # function to return the database connection
                def getconn():
                    conn = connector.connect(
                        "jp-data-viz:us-central1:jp-data-sql", # Cloud SQL Instance Connection Name
                        "pg8000",
                        user="root",
                        password="u4V2h2hdG`bC-r>}",
                        db="jp_data",
                        ip_type= IPTypes.PUBLIC  # IPTypes.PRIVATE for private IP
                    )
                    return conn

                # create connection pool
                pool = sqlalchemy.create_engine(
                    "mysql+pymysql://",
                    creator=getconn,
                )

                self.engine = pool

    def test_cloud_connection(self):
        query = "SELECT * FROM jp_data"
        with self.engine.connect() as db_conn:
            # insert into database
            db_conn.execute(query)

            # commit transaction (SQLAlchemy v2.X.X is commit as you go)
            db_conn.commit()

            # query database
            result = db_conn.execute(sqlalchemy.text("SELECT * from my_table")).fetchall()

            # Do something with the results
            for row in result:
                print(row)

    def output_df(self, output, regions, scenarios):
        combinations = product(regions, scenarios)
        long_names = [Options().filenames_to_sql_tables[output] + "_" + x[0].lower() + "_" + x[1].lower() for x in combinations]

        df_to_return = pd.DataFrame()
        for name in long_names:
            query = "SELECT `Assigned Name` FROM name_mappings WHERE `Full Output Name` = {}".format(name)
            print(self.engine.connect().execute(query))
        #     df = pd.read_sql_table(table, con = self.engine).drop(columns = "index")
        #     df_to_return = pd.concat([df_to_return, df], ignore_index = True)

        # return df_to_return

    def input_output_mapping_df(self, output, region, scenario, year):
        sql_table = Options().filenames_to_sql_tables[output] + "_" + region.lower() + "_" + scenario.lower()
        df = pd.read_sql_table(sql_table, con = self.engine).query("Year == @year").drop(columns = "index").reset_index()

        return df
    
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
    SQLConnection("all_data_jan_2024").output_df("consumption_billion_usd2007", ["USA", "GLB"], ["15C_med", "Above2C_med"])
    # SQLConnection("jp_data").input_output_mapping_df("sectoral_output_Electricity_billion_USD2007", "USA", "2C", 2050)
