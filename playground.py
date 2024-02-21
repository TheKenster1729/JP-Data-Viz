import pandas as pd
import geopandas as gpd
import plotly.express as px
from sql_utils import SQLConnection, DataRetrieval

db = SQLConnection("all_data_jan_2024")

def get_data(output_name, scenario, year):
    # This is a placeholder for your actual get_data function.
    # It should return a pandas DataFrame with 'EPPA Regions' and 'Value' columns.
    df = DataRetrieval(db, output_name, "GLB", scenario, year).choropleth_map_df(5, 95)
    return df

def generate_choropleth_map(output_name, scenario, year):
    # Retrieve the data for the specified parameters
    df = get_data(output_name, scenario, year)
    
    # Load the spatial data
    gdf = gpd.read_file(r"assets\Eppa countries\eppa6_regions_simplified.shp").rename(columns = {"EPPA6_Regi": "Region"})
    print(gdf)
    
    # Merge the data with the spatial data
    merged_gdf = gdf.merge(df, on = "Region")
    
    # Create a choropleth map
    fig = px.choropleth(merged_gdf,
                        geojson=merged_gdf.geometry,
                        locations=merged_gdf.index,
                        color="Median",
                        color_continuous_scale="Viridis",
                        scope="world",
                        labels={'Value': 'Value'},
                        title=f"Choropleth Map for {output_name}, {scenario}, {year}")
    
    # Update geodataframes layout
    fig.update_geos(fitbounds="locations", visible=False)
    
    # Show the figure
    fig.show()

# Example usage

generate_choropleth_map('emissions_CO2eq_total_million_ton_CO2eq', 'Ref', 2050)