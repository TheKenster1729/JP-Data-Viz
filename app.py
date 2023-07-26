# use venv when running this code

# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
from styling import Options

styling_options = Options()
dash_app = Dash(__name__)
app = dash_app.server

dash_app.layout = html.Div([
    dcc.Dropdown(id = "timeseries_regions",
                 value = ["GLB"],
                 options = [{"label": region, "value": region} for region in styling_options.region_names]),
    dcc.Dropdown(id = "timeseries_output",
                value = "GDP_billion_USD2007",
                options = [{"label": output, "value": output} for output in styling_options.outputs]),
    dcc.Graph(id = "timeseries_1")
])

@dash_app.callback(Output("timeseries_1", "figure"),  
              Input("timeseries_regions", "value"),
              Input("timeseries_output", "value"))
def timeseries_1(region, output):
    from figure import TimeSeries
    fig = TimeSeries("type1", output, [region], "Ref", styling_options.years).create_type1_timeseries()

    return fig

if __name__ == "__main__":
    dash_app.run()