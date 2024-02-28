import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from sql_utils import SQLConnection
from figure import InputOutputMappingPlot
import numpy as np
from styling import Options
from itertools import product
import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd
<<<<<<< HEAD
from shapely import wkt
import plotly.express as px

regions = ["USA", "EUR", "CHN"]
eppa_map = gp.read_file("assets\Eppa countries\eppa6_regions.shp")
values = []
for region in regions:
    data = SQLConnection("jp_data").input_output_mapping_df("elec_prod_Renewables_TWh", region, "Ref", 2050)["Value"].median()
    values.append(data)

data_df = pd.DataFrame(data = dict(EPPA6_Regi = regions, Value = values))
merged_df = pd.merge(eppa_map, data_df, on = "EPPA6_Regi")
print(merged_df)

gdf = gp.GeoDataFrame(merged_df, geometry = "geometry")

fig = px.choropleth(gdf,
                    geojson = gdf.geometry,
                    locations = gdf.index,
                    color = "Value",
                    color_continuous_scale = "Viridis",
                    hover_name = "EPPA6_Regi",
                    title = "Renewables Production 2050 (Ref), TWh")
# Update the layout
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Show the figure
fig.show()

# fig = go.Figure(
#     go.Choropleth(
#         locations = eppa_map.geometry,
#         z = data["Value"],
#         colorscale = "Greens"
#     )
# )

# fig.show()
=======

print("1, 2 \n 3")
>>>>>>> dev
