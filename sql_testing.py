from sql_utils import SQLConnection

connection = SQLConnection("test_db", "fulldataset")
query = "SELECT * FROM fulldataset WHERE Region='GLB' AND OutputName='total_emissions_CO2_million_ton_CO2'"
df = connection.read_query(query)
print(df)
