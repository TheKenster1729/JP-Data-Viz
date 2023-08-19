#---TIME SERIES VISUALIZATION---#
# Select type of plot
import dash_bootstrap_components as dbc
from dash import dcc, html
from styling import Options
import plotly.graph_objects as go

type_of_plot = html.Div(
    [
        html.P(
    """Here you can visualize the full timeseries for different combinations of outputs, regions, and scenarios.
    Start by selecting an output metric, and then either display the time series for that metric for one region and multiple
    scenarios, or the time series for that metric for one scenario and multiple regions."""
    ),
    html.H4("Select Output Metric"),
    dcc.Dropdown(id = "ts-output-metric", 
                 value = "GDP_billion_USD2007",
                 options = [{"label": output, "value": output} for output in Options().outputs])
    ,
    dbc.Card(
        dbc.CardBody(
            [
                html.H4("Select Plot Type"),
                dcc.Dropdown(id = "ts-plot-type",
                    value = "type1",
                    options = [{"label": "Select Region", "value": "type1"}, {"label": "Select Scenario", "value": "type2"}]
            ),
            html.H4(id = "ts-header-2"),
            dcc.Dropdown(id = "ts-dropdown-2"),
            html.H4(id = "ts-header-3"),
            dcc.Checklist(id = "ts-checklist-1"),
            dbc.Button("Submit", color = "primary", className = "me-1", id = "ts-button-1")
            ]
        ), 
        className = "mt-3"),
        dcc.Graph(id = "ts-graph-1")
])

type1_plot_options = html.Div(
        [
            html.H4("Select Region"),
            dcc.Dropdown(id = "type1-region",
                         value = "GLB",
                         options = [{"label": region, "value": region} for region in Options().region_names]),
            html.H4("Select Scenarios (Up to 3)"),
            dcc.Checklist(id = "type1-scenario-checkbox", 
                      value = "Ref", 
                      options = [{"label": scenario, "value": scenario} for scenario in Options().scenarios]),
            dcc.Graph(id = "type1-plot")
        ]
)

type2_plot_options = html.Div(
        [
            html.H4("Select Scenario"),
            dcc.Dropdown(id = "type2-scenario",
                         value = "Ref",
                         options = [{"label": scenario, "value": scenario} for scenario in Options().scenarios])
        ]
)

type1_scenario_select = html.Div(
    [
    ]
)