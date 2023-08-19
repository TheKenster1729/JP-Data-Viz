# use venv when running this code

# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
from styling import Options
import dash_bootstrap_components as dbc
from content import *
from figure import TimeSeries
import plotly.graph_objects as go

styling_options = Options()
dash_app = Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])
app = dash_app.server

dash_app.layout = dbc.Container(
    [   
        html.H1("MIT Joint Program on the Science and Policy of Global Change"),
        html.H2("Data Visualization Dashboard"),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(type_of_plot, label = "Time Series Visualization", tab_id = "ts"),
                dbc.Tab(label = "Input Distributions", tab_id = "inputdist"),
                dbc.Tab(label = "Output Distributions", tab_id = "outputdist"),
                dbc.Tab(label = "Input-Output Mapping", tab_id = "io"),
                dbc.Tab(label = "Output-Output Mapping", tab_id = "oo")
            ],
            id = "tabs",
            active_tab = "ts"
        ),
        html.Div(id = "tab-content")
    ]
)

# callback for ts visualization options
@dash_app.callback(
    Output("ts-header-2", "children"),
    Output("ts-dropdown-2", "options"),
    Output("ts-header-3", "children"),
    Output("ts-checklist-1", "options"),
    Input("ts-plot-type", "value")
)
def display_corresponding_plot_options(value):
    if value == "type1":
        return ["Select Region", [{"label": region, "value": region} for region in Options().region_names], "Select Scenario (Up to 3)", [{"label": scenario, "value": scenario} for scenario in Options().scenarios]]
    else:
        return ["Select Scenario", [{"label": scenario, "value": scenario} for scenario in Options().scenarios], "Select Region (Up to 3)", [{"label": region, "value": region} for region in Options().region_names]]

# callback for ts visualization plot
@dash_app.callback(
    Output("ts-graph-1", "figure"),
    Input("ts-output-metric", "value"),
    Input("ts-dropdown-2", "value"),
    Input("ts-checklist-1", "value"),
    Input("ts-button-1", "n_clicks")
)
def display_ts_plot(metric, dropdown1, checklist, n_clicks):
    # only proceed if submit button is clicked
    if n_clicks:
    # if (dropdown1 and checklist):
    #     fig = TimeSeries("type1", metric, dropdown1, checklist, Options().years).create_type1_timeseries()
    #     return fig
        return TimeSeries("type1", metric, dropdown1, checklist, Options().years).create_type1_timeseries()
    else:
        return go.Figure()

# callback for time series plot
# @dash_app.callback(
#     Output("type1-plot", "figure"),
#     Input("ts-output-metric", "value"),
# )
# def display_type1_plot(metric):
#     print(metric)

# @dash_app.callback(
#     Output("tab-content", "children"),
#     Input("tabs", "active_tab")
# )
# def test_graph(active_tab):
#     return active_tab
# dash_app.layout = html.Div([
#     dcc.Dropdown(id = "timeseries_regions",
#                  value = "GLB",
#                  options = [{"label": region, "value": region} for region in styling_options.region_names]),
#     dcc.Dropdown(id = "timeseries_output",
#                 value = "GDP_billion_USD2007",
#                 options = [{"label": output, "value": output} for output in styling_options.outputs]),
#     dcc.Graph(id = "timeseries_1")
# ])

# @dash_app.callback(Output("timeseries_1", "figure"),  
#               Input("timeseries_regions", "value"),
#               Input("timeseries_output", "value"))
# def timeseries_1(region, output):
#     from figure import TimeSeries
#     fig = TimeSeries("type1", output, [region], "Ref", styling_options.years).create_type1_timeseries()

#     return fig

if __name__ == "__main__":
    dash_app.run()