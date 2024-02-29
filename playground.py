from sqlalchemy import create_engine, MetaData, Table, text

engine = create_engine('mysql+mysqlconnector://root:password@localhost:3306/{}'.format("all_data_jan_2024"))
full_output_name = "percapita_consumption_loss_percent_glb_2c_pes"
query = text("SELECT `Assigned Name` FROM name_mappings WHERE `Full Output Name`=:full_output_name")

with engine.connect() as conn:
    res = conn.execute(query, parameters = {"full_output_name": full_output_name})

print(res.fetchall()[0][0])