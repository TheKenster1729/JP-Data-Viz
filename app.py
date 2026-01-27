# use venv when running this code
import dash
from dash.exceptions import PreventUpdate
from dash import html, dcc, callback_context
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from sql_utils import SQLConnection, DataRetrieval, MultiOutputRetrieval
from styling import Options, Readability, Color, FinishedFigure
from dash.dependencies import Input, Output, State
from figure import NewTimeSeries, InputOutputMappingPlot, ChoroplethMap, TimeSeriesClusteringPlot, \
                        OutputOutputMappingPlot, PlotTree, RegionalHeatmaps, InputDistributionAlternate, PermutationImportance, FilteredOutputOutputMappingPlot, \
                        FilteredInputOutputMappingPlot, STRESSPlatformConnection, TimeSeriesClusteringPlotCART, ModifyOutputTimeseries
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from analysis import InputOutputMapping
import json
import io
from itertools import product

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.PULSE, dbc.icons.BOOTSTRAP],
                suppress_callback_exceptions = True)

# Citation text for downloads
CITATION_TEXT = """Suggested Citation:
Cox K, Morris J. MIT EPPA Model Data Visualization Dashboard, 2026 (MIT Center for Sustainability Science and Strategy)
"""

# initialize SQL database and other UI elements
db_full = SQLConnection("all_data_aug_2024")
db_publication = SQLConnection("publication")
readability_obj = Readability()
options_obj = Options()

# construct navigation bar
jp_logo = r"assets/images/CSSS_sub-brand_lockup_three-line_rgb_mit-red.svg"
navbar = dbc.Navbar(
    class_name = "navbar navbar-expand-lg custom-navbar",
    color = "#3e8cda",
    dark = True,
    children = [
        dbc.Row(
            className = "w-100 align-items-center",
            style = {"margin-left": 0, "margin-right": 0},
            children = [
                dbc.Col(
                    width = "auto",
                    children = html.A(
                        html.Img(src = jp_logo, height = "60px"),
                        href = "https://cs3.mit.edu/",
                        target = "_blank",
                        style = {"textDecoration": "none"},
                    ),
                ),
                dbc.Col(
                    style = {"flex": "1", "textAlign": "center"},
                    children = dbc.NavbarBrand(
                        children = [html.Span("MIT EPPA Model"), html.Br(), html.Span("Data Visualization Dashboard")],
                        className = "ms-2",
                        style = {"color": "white", "fontSize": "28px", "lineHeight": "1.1", "fontWeight": "700"},
                    ),
                ),
                dbc.Col(
                    width = "auto",
                    style = {"margin-left": "auto", "margin-right": "15%"},
                    children = html.A(
                        "Dashboard Guide",
                        href = "https://www.notion.so/thekenster/MIT-JP-Data-Visualization-Dashboard-User-Guide-f696462c92bc4280a261ef67b1ab3bf3",
                        target = "_blank",
                        style = {"textDecoration": "none", "color": "white"},
                    ),
                ),
            ],
        ),
        dbc.NavbarToggler(id = "navbar-toggler", n_clicks = 0),
    ],
)

overview = html.Div(id = "overview-content", style = {"padding": 20},
    children = [
        dbc.Row(
            children = [
                dbc.Col(width = 6,
                    children = [
                        dbc.Row(html.H3("Data Selection")),
                        dbc.Row(html.P("Select the data you want to visualize.")),
                        dbc.Row(html.P("The two datasets represent two distinct ensembles of runs of the MIT EPPA Model. The Publication Dataset was used for an upcoming publication, while the Full Dataset contains a broader range of scenarios. Refer to the guide for additional information.")),
                        dcc.Dropdown(id = "overview-data-dropdown", 
                                     options = [{"label": "Full Dataset", "value": "full"}, 
                                     {"label": "Publication Dataset", "value": "publication"},
                                     ], 
                                     value = "full",
                                     style = {"width": "80%"}
                                    ),
                    ]
                )
            ]
        )
    ]
)

output_timeseries = html.Div(id = "tab-1-content", style = {"padding": 20},
    children = [
        dbc.Row(
            children = [
            dbc.Col(width = 2,
                children = [
                    dbc.Card(
                        className = "card text-white bg-primary mb-3",
                        children = [
                            html.Div(style = {'display': 'flex'},
                                children = [
                                    html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Output Visualization")
                                ]
                            )
                        ]
                    )                    
                ]
            ),
            dbc.Col(width = 10,
                    children = [
                        dbc.Card(
                            dbc.CardBody(
                                children = [
                                    dbc.Row(
                                        children = [
                                            dbc.Col(style = {},
                                                width = 9,
                                                children = [
                                                    dbc.Row(html.Div("Output Name", className = "text-primary")),
                                                    dbc.Row(
                                                        children = [
                                                            dcc.Dropdown(id = "output-dropdown", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                                                                        value = "emissions_CO2eq_total_million_ton_CO2eq")
                                                        ]
                                                    )
                                                ]
                                            ),
                                            # dbc.Col(style = {},
                                            #     width = 3,
                                            #     children = [
                                            #         dbc.Row(html.Div("View", className = "text-primary")),
                                            #         dbc.Row(
                                            #             children = [
                                            #                 dcc.Dropdown(id = "chart-options", options = [{"label": "Time Series", "value": "time-series"},
                                            #                                         {"label": "Distribution by Year", "value": "dist-by-year"}],
                                            #                             value = "time-series")
                                            #             ]
                                            #         )
                                            #     ]
                                            # )
                                        ]
                                    )
                                ]
                            )
                        )
                    ]
                )
            ]
        ),
        dbc.Row(align = "end",
            children = [
                dbc.Col(width = {"size": 1, "offset": 1},
                    children = [
                        dbc.Row(html.Div("Region"), className = "text-primary"),
                        dbc.Row(
                            children = [
                                dbc.Checklist(id = "region-dropdown", style = {"padding": 10}, options = [{"label": i, "value": i} for i in options_obj.region_names], value = ["GLB"])
                            ]
                        )
                    ]
                ),
                dbc.Col(width = 10,
                    children = [
                        dbc.Row(
                            dcc.Graph(id = "output-time-series-plot", config={'modeBarButtonsToRemove': ['toImage']})
                        ),
                        dbc.Row(
                            html.Div(style = {"display": "none"},
                                id = "slider-area",
                                children = [
                                    html.Div("Year", className = "text-primary"),
                                    dcc.Slider(
                                        id = 'year-slider',
                                        min = min(Options().years),
                                        max = max(Options().years),
                                        value = 2050,
                                        marks = {str(year): str(year) for year in Options().years[::2]},
                                        step = 5
                                    )
                                ]
                            )                        
                        ),
                    ]
                )
            ]
        ),
        dbc.Row(
            children = [
                dbc.Col(width = {"size": 9, "offset": 1},
                    children = [
                        dbc.Row(html.Div("Scenario", className = "text-primary")),
                        dbc.Row(
                            dcc.Dropdown(id = "scenario-dropdown", style = {"padding": 10}, options = [{"label":k, "value":v} for k, v in options_obj.scenario_display_names_rev.items()],
                                        value = ["2C_med"], multi = True)
                        )
                    ]
                ),
                dbc.Col(width = 2,
                    children = [
                        dbc.Row(html.Div("Color"), className = "text-primary"),
                        dbc.Row(
                            dcc.Dropdown(id = "output-color-scheme", options = [{"label": "Standard", "value": "standard"}, {"label": "By Region", "value": "by-region"}, {"label": "By Scenario", "value": "by-scenario"}],
                                        value = "standard")
                        )
                    ]
                )
            ]
        ),
        dbc.Row(style = {"margin-top": 20}, children = [
                dbc.Col(width = {"size": 10, "offset": 1},
                    children = [
                        dbc.Accordion(start_collapsed = True,
                            children = [
                                dbc.AccordionItem(title = "Plot Options",
                                    children = [
                                        html.P("Set Uncertainty Range - Upper and Lower Percentiles", className = "text-primary"),
                                        html.P("Upper Bound"),
                                        dcc.Slider(51, 99, 1, id = "time-series-plot-upper-bound", value = 95,
                                                    marks = {label: str(label) for label in range(50, 100, 5)}, tooltip = dict(always_visible = True)),
                                        html.P("Lower Bound"),
                                        dcc.Slider(1, 49, 1, id = "time-series-plot-lower-bound", value = 5, 
                                                    marks = {label: str(label) for label in range(0, 50, 5)}, tooltip = dict(always_visible = True)),
                                        dbc.Button("Set Bounds", id = "time-series-plot-apply-bound-changes", class_name = "Primary")
                                    ]
                                ),
                                dbc.AccordionItem(title = "Styling Options",
                                    children = [
                                        html.P("Set Plot Background Color", className = "text-primary"),
                                        dmc.ColorPicker(id = "time-series-plot-color-picker", format = "hex", value = "#e5ecf5"),
                                        html.Br(),
                                        html.P("Toggle Gridlines", className = "text-primary"),
                                        dmc.Switch(label = "Toggle Gridlines", onLabel = "On", offLabel = "Off", size = "lg", radius = "sm", id = "time-series-plot-toggle-gridlines", checked = True),
                                        html.Br(),
                                        dbc.Button("Apply Changes", id = "time-series-plot-apply-styling-changes", class_name = "Primary")
                                    ]
                                ),
                                dbc.AccordionItem(title = "Save Options",
                                    children = [
                                        html.P("Note: downloads may take a few seconds to complete. A citation file will also be downloaded.", className = "text-info"),
                                        dbc.Button("Download Data as CSV", id = "time-series-plot-download-data-button"),
                                        dbc.Button("Download Plot as High-Res Image", id = "time-series-plot-download-image-button", style = {"margin-left": 20, "margin-right": 20}),
                                        dbc.Button("Download Plot as SVG", id = "time-series-plot-download-svg-button"),
                                        dcc.Download(id = "time-series-plot-download-csv"),
                                        dcc.Download(id = "time-series-plot-download-image"),
                                        dcc.Download(id = "time-series-plot-download-svg"),
                                        dcc.Download(id = "time-series-plot-download-citation-csv"),
                                        dcc.Download(id = "time-series-plot-download-citation-image"),
                                        dcc.Download(id = "time-series-plot-download-citation-svg")
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

input_dists = html.Div(style = {"padding": 20},
            children = [
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                dbc.Card(
                                    className = "card text-white bg-primary mb-3",
                                    children = [
                                        html.Div(style = {"display": "flex"},
                                            children = [
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Input Distributions")
                                            ]
                                        )
                                    ]
                                )
                            ]
                        ),
                        dbc.Col(width = 10,
                            children = [
                                dbc.Card(
                                    dbc.CardBody(
                                        children = [
                                            dbc.Row(
                                                children = [
                                                    dbc.Col(width = 8,
                                                        children = [
                                                            dbc.Row(html.Div("Inputs to Compare", className = "text-primary")),
                                                            dbc.Row(
                                                                dcc.Dropdown(
                                                                    id = "input-dist-options",
                                                                    options = [{'label': i, 'value': i} for i in Options().input_names],
                                                                    value = ["wind", "oil", "gas", "WindGas", "WindBio"],
                                                                    multi = True
                                                                    ),
                                                                )
                                                            ]
                                                        )
                                                        ]
                                                    )
                                                ]
                                            )
                                        )
                                    ]
                                )
                            ]
                        ),
                html.Div(
                        children = [
                            dcc.Graph(id = "input-dist-graph")
                        ]
                )
            ]
        )

input_output_mapping = html.Div(id = "tab-4-content", style = {"padding": 20},
    children = [
        html.Div(
            children = [
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                dbc.Card(
                                    className = "card text-white bg-primary mb-3",
                                    children = [
                                        html.Div(style = {'display': 'flex'},
                                            children = [
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Input/Output Mapping")
                                            ]
                                        )
                                    ]
                                )                    
                            ]
                        ),
                        dbc.Col(width = 10,
                            children = [
                                dbc.Card(
                                    dbc.CardBody(
                                        children = [
                                            dbc.Row(
                                                children = [
                                                    dbc.Col(
                                                        children = [
                                                            dbc.Row(html.Div("Output Name", className = "text-primary")),
                                                            dbc.Row(
                                                            dcc.Dropdown(id = "input-output-mapping-output",
                                                                options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                                value = "emissions_CO2eq_total_million_ton_CO2eq")
                                                            )
                                                        ]
                                                    ),
                                                    dbc.Col(width = 2, 
                                                        children = [
                                                            dbc.Row(html.Div("Mode", className = "text-primary")),
                                                            dbc.Row(
                                                                dcc.Dropdown(id = "input-output-mapping-mode",
                                                                    options = [{'label': "Standard", 'value': "standard"}, {'label': "Filtered", 'value': "filtered"}],
                                                                    value = "standard")
                                                            )
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                html.Div("Region", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "input-output-mapping-region",
                                    options = [{'label': i, 'value': i} for i in Options().region_names],
                                    value = "GLB"
                                ),
                                html.Div("Scenario", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "input-output-mapping-scenario",
                                    options = [{'label': Options().scenario_display_names[i], 'value': i} for i in Options().scenarios],
                                    value = "Ref"
                                ),
                                html.Div("Year", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "input-output-mapping-year",
                                    options = [{'label': i, 'value': i} for i in Options().years],
                                    value = 2050),
                                html.Div("Percentile", className = "text-primary"),
                                dcc.Slider(
                                    id = "input-output-mapping-percentile",
                                    min = 1,
                                    max = 100,
                                    step = 1,
                                    value = 70,
                                    marks = {i: str(i) for i in range(15, 100, 15)},
                                    tooltip = dict(always_visible = True)
                                ),
                                html.Div("Setting", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "input-output-mapping-setting",
                                    options = [{"label": "Above Threshold", "value": "above"}, {"label": "Below Threshold", "value": "below"}],
                                    value = "above"
                                    ),
                                html.Br(),
                                dbc.Button("Update", id = "input-output-mapping-update-all-settings", className = "Primary")
                                ]
                            ),
                            dbc.Col(width = 10,
                                children = [
                                    html.Div(id = "input-output-mapping-figure-container", children = [
                                    html.Br(),
                                    dbc.Row(
                                        children = [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id = "custom-io-mapping-dropdown-1",
                                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                    value = "emissions_CO2eq_total_million_ton_CO2eq"
                                                )
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id = "custom-io-mapping-dropdown-2",
                                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                    value = "elec_prod_Renewables_TWh_pol"
                                                )
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id = "custom-io-mapping-dropdown-3",
                                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                    value = "sectoral_output_Electricity_billion_USD2007"
                                                )
                                            )
                                        ]
                                    ),
                                    html.Br(),
                                    dbc.Row(
                                        children = [
                                            dbc.Col(
                                                dcc.RangeSlider(
                                                        id = "slider-custom-io-mapping-1",
                                                        min = 0,
                                                        max = 100,
                                                        step = 1,
                                                        value = [0, 33],
                                                        marks = {i: str(i) for i in range(10, 99, 10)},
                                                        tooltip = dict(always_visible = True)
                                                    )
                                            ),
                                            dbc.Col(
                                                dcc.RangeSlider(
                                                        id = "slider-custom-io-mapping-2",
                                                        min = 0,
                                                        max = 100,
                                                        step = 1,
                                                        value = [66, 100],
                                                        marks = {i: str(i) for i in range(10, 99, 10)},
                                                        tooltip = dict(always_visible = True)
                                                    )
                                                ),
                                                dbc.Col(
                                                dcc.RangeSlider(
                                                        id = "slider-custom-io-mapping-3",
                                                        min = 0,
                                                        max = 100,
                                                        step = 1,
                                                        value = [33, 66],
                                                        marks = {i: str(i) for i in range(10, 99, 10)},
                                                        tooltip = dict(always_visible = True)
                                                    ),
                                                )
                                            ]
                                        ),
                                    ],
                                    hidden = True
                                ),
                                html.Div(id = "input-output-mapping-run-count", className = "text-primary", style = {"padding": "10px 0"}),
                                dcc.Loading([dcc.Graph(id = "input-output-mapping-figure")])
                                ]
                            ),
                            dbc.Col(width = {"size": 10, "offset": 2},
                                children = [
                                    dbc.Accordion(
                                        start_collapsed = True,
                                        children=[
                                            dbc.AccordionItem(title = "Hyperparameters",
                                                              children = [
                                                                  dbc.Row(html.Div("Number of Estimators in Ensemble")),
                                                                  html.Br(),
                                                                  dbc.Row(
                                                                      dcc.Slider(50, 500, 1, id = "input-output-mapping-n-estimators", value = 100,
                                                                                marks = {i: str(i) for i in range(50, 500, 50)}, tooltip = dict(always_visible = True))
                                                                  ),
                                                                  dbc.Row(html.Div("Max Depth of Trees in Ensemble")),
                                                                  html.Br(),
                                                                  dbc.Row(
                                                                      dcc.Slider(1, 10, 1, id = "input-output-mapping-max-depth", value = 4,
                                                                                marks = {i: str(i) for i in range(1, 10, 1)}, tooltip = dict(always_visible = True))
                                                                  )
                                                              ]),
                                            dbc.AccordionItem(
                                                title="Full Tree",
                                                children=[
                                                    dbc.Row(html.Div("Tree Depth", className = "text-primary")),
                                                    dbc.Row(style={'maxWidth': '200px'}, children = dcc.Dropdown(id = "full-cart-tree-depth-dropdown", 
                                                                options = [{"label": i, "value": i} for i in range(1, 10)], value = 4)),
                                                    dbc.Row(
                                                        children=[
                                                            dbc.Col(
                                                                dcc.Loading([dcc.Graph(id="full-cart-tree")]),
                                                                width=10
                                                            )
                                                        ]
                                                    )
                                                ],
                                            ),
                                            dbc.AccordionItem(
                                                title = "Robustness Check: Permutation Importance",
                                                children = [
                                                    dcc.Loading([dcc.Graph(id = "input-output-mapping-permutation-importance")]),
                                                    html.P("Error bars represent +/- 1 standard deviation.")
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )                        
                        ]
                    )
                ]
            )
        ]
    )

output_output_mapping = html.Div(id = "output-output-mapping-content", style = {"padding": 20},
    children = [
        html.Div(
            children = [
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                dbc.Card(
                                    className = "card text-white bg-primary mb-3",
                                    children = [
                                        html.Div(style = {'display': 'flex'},
                                            children = [
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Output/Output Mapping")
                                            ]
                                        )
                                    ]
                                )                    
                            ]
                        ),
                        dbc.Col(
                            children = [
                                dbc.Card(
                                    dbc.CardBody(
                                        children = [
                                            dbc.Row(
                                                children = [
                                                    dbc.Col(width = 10,
                                                        children = [
                                                            dbc.Row(html.Div("Output Name", className = "text-primary")),
                                                            dbc.Row(
                                                            dcc.Dropdown(id = "output-output-mapping-output",
                                                                options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                                value = "emissions_CO2eq_total_million_ton_CO2eq")
                                                            )
                                                        ]
                                                    ),
                                                    dbc.Col(width = 2, 
                                                        children = [
                                                            dbc.Row(html.Div("Mode", className = "text-primary")),
                                                            dbc.Row(
                                                                dcc.Dropdown(id = "output-output-mapping-mode",
                                                                    options = [{'label': "Standard", 'value': "standard"}, {'label': "Filtered", 'value': "filtered"}],
                                                                    value = "standard")
                                                            )
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                html.Div("Region", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "output-output-mapping-region",
                                    options = [{'label': i, 'value': i} for i in Options().region_names],
                                    value = "GLB"
                                ),
                                html.Div("Scenario", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "output-output-mapping-scenario",
                                    options = [{'label': Options().scenario_display_names[i], 'value': i} for i in Options().scenarios],
                                    value = "Ref"
                                ),
                                html.Div("Year", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "output-output-mapping-year",
                                    options = [{'label': i, 'value': i} for i in Options().years],
                                    value = 2050),
                                html.Br(),
                                dbc.Button("Update", id = "output-output-mapping-update", className = "btn btn-primary")
                                ]
                            ),
                        dbc.Col(width = 10,
                            children = [
                                html.Div(id = "output-output-mapping-figure-container", children = [
                                    html.Br(),
                                    dbc.Row(
                                        children = [
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id = "custom-oo-mapping-dropdown-1",
                                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                    value = "emissions_CO2eq_total_million_ton_CO2eq"
                                                )
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id = "custom-oo-mapping-dropdown-2",
                                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                    value = "elec_prod_Renewables_TWh_pol"
                                                )
                                            ),
                                            dbc.Col(
                                                dcc.Dropdown(
                                                    id = "custom-oo-mapping-dropdown-3",
                                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                                    value = "sectoral_output_Electricity_billion_USD2007"
                                                )
                                            )
                                        ]
                                    ),
                                    html.Br(),
                                    dbc.Row(
                                        children = [
                                            dbc.Col(
                                                dcc.RangeSlider(
                                                        id = "slider-custom-oo-mapping-1",
                                                        min = 0,
                                                        max = 100,
                                                        step = 1,
                                                        value = [0, 33],
                                                        marks = {i: str(i) for i in range(10, 99, 10)},
                                                        tooltip = dict(always_visible = True)
                                                    )
                                            ),
                                            dbc.Col(
                                                dcc.RangeSlider(
                                                        id = "slider-custom-oo-mapping-2",
                                                        min = 0,
                                                        max = 100,
                                                        step = 1,
                                                        value = [66, 100],
                                                        marks = {i: str(i) for i in range(10, 99, 10)},
                                                        tooltip = dict(always_visible = True)
                                                    )
                                                ),
                                                dbc.Col(
                                                dcc.RangeSlider(
                                                        id = "slider-custom-oo-mapping-3",
                                                        min = 0,
                                                        max = 100,
                                                        step = 1,
                                                        value = [33, 66],
                                                        marks = {i: str(i) for i in range(10, 99, 10)},
                                                        tooltip = dict(always_visible = True)
                                                    ),
                                                )
                                            ]
                                        ),
                                    ],
                                    hidden = True
                                ),
                                html.Div(id = "output-output-mapping-run-count", className = "text-primary", style = {"padding": "10px 0"}),
                                dcc.Loading([dcc.Graph(id = "output-output-mapping-figure")]),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

regional_heatmaps = html.Div(id = "regional-heatmaps", style = {"padding": 20},
    children = [
        html.Div(
            children = [
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                dbc.Card(
                                    className = "card text-white bg-primary mb-3",
                                    children = [
                                        html.Div(style = {'display': 'flex'},
                                            children = [
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Regional Heatmaps")
                                            ]
                                        )
                                    ]
                                )                    
                            ]
                        ),
                        dbc.Col(
                            children = [
                                dbc.Row(html.Div("Output Name", className = "text-primary")),
                                dbc.Row(
                                dcc.Dropdown(id = "regional-heatmaps-output",
                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                    value = "emissions_CO2eq_total_million_ton_CO2eq")
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                html.Div("Regions", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "regional-heatmaps-region",
                                    options = [{'label': i, 'value': i} for i in Options().region_names],
                                    value = ["GLB"], multi = True
                                ),
                                html.Div("Scenarios", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "regional-heatmaps-scenario",
                                    options = [{'label': Options().scenario_display_names[i], 'value': i} for i in Options().scenarios],
                                    value = ["Ref"], multi = True
                                    ),
                                html.Br(),
                                dbc.Button("Update", id = "regional-heatmaps-apply-button", className = "btn btn-primary")
                                ]
                            ),
                        dbc.Col(width = 10,
                            children = [
                                dcc.Loading([dcc.Graph(id = "regional-heatmaps-figure")]),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

choropleth_map = html.Div(style = {"padding": 20},
    children = [
        html.Div(
            children = [
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                dbc.Card(
                                    className = "card text-white bg-primary mb-3",
                                    children = [
                                        html.Div(style = {'display': 'flex'},
                                            children = [
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Choropleth Mapping")
                                            ]
                                        )
                                    ]
                                )                    
                            ]
                        ),
                        dbc.Col(
                            children = [
                                dbc.Row(html.Div("Output Name", className = "text-primary")),
                                dbc.Row(
                                dcc.Dropdown(id = "choropleth-mapping-output",
                                    options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                                    value = "emissions_CO2eq_total_million_ton_CO2eq")
                                )
                            ]
                        )
                    ]
                ),
                dbc.Row(
                    children = [
                        dbc.Col(width = 2,
                            children = [
                                html.Div("Scenario", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "choropleth-mapping-scenario",
                                    options = [{'label': Options().scenario_display_names[i], 'value': i} for i in Options().scenarios],
                                    value = "Ref"
                                ),
                                html.Div("Year", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "choropleth-mapping-year",
                                    options = [{'label': i, 'value': i} for i in Options().years],
                                    value = 2050),
                                html.Br(),
                                dbc.Button("Update", id = "choropleth-mapping-update", className = "btn btn-primary")
                                ]
                            ),
                        dbc.Col(width = 10,
                            children = [
                                dbc.Row(
                                    children = [dcc.Loading([dcc.Graph(id = "choropleth-mapping-figure")])]
                                    ),
                                dbc.Row(
                                    children = [html.P("Note: upper bound corresponds to the 95th percentile value, while lower bound corresponds to the 5th percentile value.", className = "text-primary", style = {"font-size": "16px"})]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

time_series_clustering = html.Div(id = "ts-clustering", style = {"padding": 20},
    children = [
        dbc.Row(
            children = [
            dbc.Col(width = 2,
                children = [
                    dbc.Card(
                        className = "card text-white bg-primary mb-3",
                        children = [
                            html.Div(style = {'display': 'flex'},
                                children = [
                                    html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Time Series Clustering")
                                ]
                            )
                        ]
                    )                    
                ]
            ),
            dbc.Col(width = 9,
                    children = [
                        dbc.Row(
                            children = [
                                dbc.Col(style = {},
                                    width = 9,
                                    children = [
                                        dbc.Row(html.Div("Output Name", className = "text-primary")),
                                        dbc.Row(
                                            children = [
                                                dcc.Dropdown(id = "ts-clustering-output", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                                                            value = "emissions_CO2eq_total_million_ton_CO2eq")
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            children = [
                dbc.Col(width = {"size": 2},
                    children = [
                        dbc.Row(html.Div("Region"), className = "text-primary"),
                        dbc.Row(
                            children = [
                                dcc.Dropdown(id = "ts-clustering-region", options = [{"label": i, "value": i} for i in options_obj.region_names], value = "GLB")
                            ]
                        ),
                        dbc.Row(html.Div("Scenario"), className = "text-primary"),
                        dbc.Row(
                            children = [
                            dcc.Dropdown(id = "ts-clustering-scenario", options = [{"label":k, "value":v} for k, v in options_obj.scenario_display_names_rev.items()],
                                        value = "2C_med")
                            ]
                        ),
                        dbc.Row(html.Div("Number of Clusters"), className = "text-primary"),
                        dbc.Row(
                            children = [
                            dcc.Dropdown(id = "ts-clustering-n-clusters", options = [{"label": i, "value": i} for i in range(1, 8)],
                                        value = 3)
                            ]
                        ),
                        dbc.Row(html.Div("Clustering Metric"), className = "text-primary"),
                        dbc.Row(
                            children = [
                            dcc.Dropdown(id = "ts-clustering-metric", options = [{"label": "Euclidean", "value": "euclidean"}, {"label": "DBA", "value": "dtw"}, {"label": "Soft-DTW", "value": "softdtw"}],
                                        value = "euclidean")
                            ]
                        ),
                        html.Br(),
                        dbc.Button("Update", id = "ts-clustering-update", className = "btn btn-primary")
                    ]
                ),
                dbc.Col(width = 10,
                    children = [
                        dbc.Row(
                            dcc.Loading(dcc.Graph(id = "ts-clustering-plot"))
                        ),
                        dbc.Row(
                            children = [
                                dbc.Accordion(
                                    children = [
                                        dbc.AccordionItem(
                                            children = [
                                                dcc.Loading(dcc.Graph(id = "ts-clustering-random-forest-plot"))
                                            ],
                                            title = "CART"
                                        )
                                    ],
                                    start_collapsed = True
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

custom_variables = html.Div(children = 
    [
        html.Div(style = {'display': 'flex', 'alignItems': 'center', 'padding': '20px'},
            children = [
                html.Span("I would like to create a custom variable called ", style = {'margin-right': '10px'}, className = "text-info"),
                dcc.Input(id = "custom-vars-var-name", style = {"margin-right": "10px", 'width': '200px'}),
                html.Span("by", className = "text-info")
            ]
        ),
        html.Div(id = "custom-vars-fill-area", style = {'display': 'flex', 'alignItems': 'center', "margin-left": "100px"},
            children = [
                dcc.Dropdown(
                    id = "custom-vars-operation", 
                    options = [{"label": "Dividing", "value": "division"}, {"label": "Adding", "value": "addition"}, {"label": "Multiplying", "value": "multiplication"}, {"label": "Subtracting", "value": "subtraction"}],
                    placeholder = "Operation",
                    style = {"width": "200px", "margin-right": "10px"}
                ),
                html.Div(id = "custom-vars-output-dropdown-div",
                    children = [
                ])
            ]
        ),
        html.Div(style = {"padding": 20},
            children = [
                dbc.Button(id = "create-custom-variable-button", children = "Create", className = "btn btn-primary btn-lg"),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Success!"), close_button = True),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id = "close-centered",
                                className = "ms-auto",
                                n_clicks = 0,
                            )
                        )
                    ],
                    id = "custom-variable-created-modal",
                    is_open = False,
                )
            ]
        )
    ]
)

stress_connection = html.Div(id = "stress-connection", style = {"padding": 20},
    children = [
        dbc.Row(
            children = [
                dbc.Col(
                    children = [
                        html.Div("Outputs", className = "text-primary"),
                        dcc.Dropdown(id = "stress-connection-outputs",
                            options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in Options().outputs],
                            value = ["emissions_CO2eq_total_million_ton_CO2eq"],
                            multi = True
                        )
                    ]
                )
            ]
        ),
        dbc.Row(style = {"padding-top": 10},
            children = [
                dbc.Col(
                    children = [
                        html.Div("Inputs", className = "text-primary"),
                        dcc.Dropdown(id = "stress-connection-inputs",
                            options = [{'label': i, 'value': i} for i in options_obj.input_names],
                            value = ["WindGas", "BioCCS"],
                            multi = True
                        )
                    ]
                )
            ]
        ),
        dbc.Row(style = {"padding-top": 10},
            children = [
                dbc.Col(width = 2,
                    children = [
                        html.Div("Scenario", className = "text-primary"),
                        dcc.Dropdown(id = "stress-connection-scenario", options = [{"label": i, "value": i} for i in options_obj.scenario_display_names_rev.keys()], value = "2C_med")
                    ]
                ),
                dbc.Col(width = 1,
                    children = [
                        html.Div("Year", className = "text-primary"),
                        dcc.Dropdown(id = "stress-connection-year", options = [{"label": i, "value": i} for i in options_obj.years], value = 2050)
                    ]
                ),
                dbc.Col(width = 1,
                    children = [
                        html.Div("Region", className = "text-primary"),
                        dcc.Dropdown(id = "stress-connection-region", options = [{"label": i, "value": i} for i in options_obj.region_names], value = "GLB")
                    ]
                ),
                dbc.Col(
                    children = [
                        html.Div("Color", className = "text-primary"),
                        dcc.Dropdown(id = "stress-connection-color", options = [{'label': Readability().naming_dict_long_names_first[i], 'value': i} for i in options_obj.outputs], value = "primary_energy_use_Biofuel_FirstGen_EJ")
                    ]
                )
            ]
        ),
        dbc.Row(style = {"padding-top": 10},
            children = [
                dbc.Col(
                    children = [
                        dbc.Button("Apply", id = "stress-connection-button", className = "btn btn-primary")
                    ]
                )
            ]
        ),
        dbc.Row(style = {"padding-top": 10},
            children = [
                dbc.Col(
                    children = [
                        dcc.Loading(dcc.Graph(id = "stress-connection-graph"), type = "cube")
                    ]
                )
            ]
        ),
        dbc.Row(style = {"padding-top": 10},
            children = [
                dbc.Col(
                    children = [
                        dbc.Button("Update Table", id = "stress-connection-update-table-button", className = "btn btn-primary")
                    ]
                )
            ]
        ),
        dbc.Row(style = {"padding-top": 10},
            children = [
                dbc.Col(
                    children = [
                        dcc.Loading(dbc.Table(id = "stress-connection-table"), type = "cube")
                    ]
                )
            ]
        )
    ]
)

examples = html.Div(style = {},
    children = [
        html.Br(),
        dbc.Accordion(children = 
                      [
                          dbc.AccordionItem(
                              title = "Example Scenario Discovery Pipeline - Basic Usage",
                              children = 
                                            [
                                                html.P("Carbon Emissions in the 21st Century", className = "text-primary"),
                                                html.P("We can study what the EPPA model tells us will influence carbon emissions over the 21st century. First, we display the time series data for \
                                                       carbon emissions (see the Output Time Series tab) and for the desired regions/scenarios (for this example, data is displayed for the US, EU, and China under both reference and \
                                                       2C policies)."),
                                                html.Div(),
                                                html.Img(src = "assets\examples\carbonemissions.svg"),
                                                html.P("We can now run scenario discovery algorithms to determine the key drivers of carbon emissions in a given year according to the EPPA model.\
                                                       Using the Input-Output Mapping tab, and USA as the region, Ref as the scenario, and 2050 as the year, we obtain the following plots."),
                                                html.Img(src = "assets\examples\cart_usa_ref_2050.svg"),
                                                html.P("Compare these results with the same parameters, except with the 2C policy data."),
                                                html.Img(src = "assets\examples\cart_usa_2c_2050.svg"),
                                                html.P("These results indicate WindGas, the cost of wind energy with gas backup, is the most important feature to predict carbon emissions for the USA under \
                                                       policy in 2050. Using the Input Distributions tab, we can visualize this input, along with some other related inputs, and use the Input Focus plot \
                                                       to see the full histogram for the WindGas input."),
                                                html.Img(src = "assets\examples\inputs_windgas_focus.svg")
                                            ])
                      ])
    ]
)

# build layout
app.layout = html.Div(
    [
        navbar,
        html.Br(),
        html.P("Select a tab to display data or run scenario discovery algorithms. All figures are preserved when you switch between tabs.", className = "text-primary", style = {"padding": 20}),
        html.Div([
            dbc.Tabs(
                id = "tabs",
                children = [
                    # dbc.Tab(id = "examples-gallery", label = "Examples Gallery", children = [examples]),
                    dbc.Tab(id = "overview", label = "Overview", children = [overview]),
                    dbc.Tab(id = "output-timeseries", label = "Output Distributions", children = [output_timeseries]),
                    dbc.Tab(id = "input-dist", label = "Input Distributions", children = [input_dists]),
                    dbc.Tab(id = "input-output-mapping", label = "Input-Output Mapping", children = [input_output_mapping]),
                    dbc.Tab(id = "output-output-mapping", label = "Output-Output Mapping", children = [output_output_mapping]),
                    dbc.Tab(id = "choropleth-map", label = "Choropleth Mapping", children = [choropleth_map]),
                    dbc.Tab(id = "ts-clustering-tab", label = "Time Series Clustering", children = [time_series_clustering]),
                    dbc.Tab(id = "regional-heatmaps-tab", label = "Regional Heatmaps", children = [regional_heatmaps]),
                    dbc.Tab(id = "custom-variables-tab", label = "Custom Variables", children = [custom_variables]),
                    # dbc.Tab(id = "stress-connection-tab", label = "Connect to STRESS Platform", children = [stress_connection])
                ]
                )
            ]
            ),
        dcc.Store(id = "stored-custom-variables", storage_type = "memory")
    ]
)

output_dropdowns_to_update_ids = ["output-dropdown", "input-output-mapping-output", "choropleth-mapping-output", "ts-clustering-output", "output-output-mapping-output", "regional-heatmaps-output",
                                  "custom-io-mapping-dropdown-1", "custom-io-mapping-dropdown-2", "custom-io-mapping-dropdown-3",
                                  "custom-oo-mapping-dropdown-1", "custom-oo-mapping-dropdown-2", "custom-oo-mapping-dropdown-3"]
for dropdown in output_dropdowns_to_update_ids:
    @app.callback(
        Output(dropdown, "options", allow_duplicate = True),
        Output(dropdown, "value"),
        State(dropdown, "options"),
        Input("stored-custom-variables", "data"),
        Input("create-custom-variable-button", "n_clicks"),
        Input("custom-vars-operation", "value"),
        Input("overview-data-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_output_dropdowns(current_options, current_stored_data, n_clicks, operation, publication_output):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]
        if trigger_id == "overview-data-dropdown":
            if publication_output == "full":
                options = [{"label": j, "value": i} for i, j in Readability().naming_dict_long_names_first.items()]
            else:
                options = [{"label": j, "value": i} for i, j in Readability().publication_naming_dict_long_names_first.items()]
            if current_stored_data: 
                custom_variable_options = [{"label": i, "value": json.dumps(j)} for i, j in current_stored_data.items()]
                options = options + custom_variable_options
            return options, options[0]["value"]
        else:
            if n_clicks is None or not current_stored_data:
                raise PreventUpdate
            else:
                # Create a copy of current options to avoid modifying the original
                updated_options = current_options.copy()
                
                # Check for new custom variables and add them to options
                for custom_var_name, custom_var_data in current_stored_data.items():
                    all_current_labels = [x["label"] for x in updated_options]
                    if custom_var_name not in all_current_labels:
                        value_to_use = json.dumps(custom_var_data)
                        dict_to_append = {"label": custom_var_name, "value": value_to_use}
                        updated_options.append(dict_to_append)
                
                # Return updated options and keep the current value
                return updated_options, dash.no_update

# change all scenario menus depending on overview data dropdown
scenario_dropdowns_to_update_ids = ["scenario-dropdown", "ts-clustering-scenario", "regional-heatmaps-scenario", "output-output-mapping-scenario", "input-output-mapping-scenario", "choropleth-mapping-scenario"]
for dropdown in scenario_dropdowns_to_update_ids:
    @app.callback(
        Output(dropdown, "options"),
        Output(dropdown, "value"),
        Input("overview-data-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_scenario_dropdowns(publication_output):
        if publication_output == "full":
            options = [{"label":k, "value":v} for k, v in options_obj.scenario_display_names_rev.items()]
        else:
            options = [{"label": "Reference", "value": "Ref"}, {"label": "2C", "value": "2C"}]
        
        # For multi-select dropdowns, return a list with the first value
        # For single-select dropdowns, return just the first value
        if dropdown in ["scenario-dropdown", "regional-heatmaps-scenario"]:
            return options, [options[0]["value"]]
        else:
            return options, options[0]["value"]

# # adding slider for histogram when user selects histogram option
# @app.callback(
#     Output('slider-area', 'style'),
#     Input('chart-options', 'value'))
# def add_hist_slider(chart_type):
#     if chart_type == 'dist-by-year':
#         return {}
#     else:
#         return {"display": "none"}

# callback for output time series
@app.callback(
    Output('output-time-series-plot', 'figure'),
    [Input('output-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('scenario-dropdown', 'value'),
    #  Input('chart-options', 'value'),
    #  Input('year-slider', 'value'),
     Input('output-color-scheme', 'value'),
     Input("time-series-plot-apply-bound-changes", "n_clicks"),
     Input("time-series-plot-apply-styling-changes", "n_clicks")],
    [State('output-time-series-plot', 'figure'),
     State("time-series-plot-upper-bound", "value"),
     State("time-series-plot-lower-bound", "value"),
     State("time-series-plot-color-picker", "value"),
     State("time-series-plot-toggle-gridlines", "checked"),
     State("overview-data-dropdown", "value")]
)
# add back in commented out inputs when ready to re-incorporate
def update_timeseries_graph(output_name, selected_regions, selected_scenarios, color_scheme, n_clicks_bound_changes, 
                            n_clicks_styling_changes, existing_figure, upper_bound, lower_bound, plot_bgcolor, toggle_gridlines, overview_data_dropdown):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]
    
    if overview_data_dropdown == "full":
        db = db_full
    elif overview_data_dropdown == "publication":
        db = db_publication
    else:
        raise ValueError("Invalid overview data dropdown value")

    # if chart_type == "time-series":
    if not selected_regions or not selected_scenarios:
        raise PreventUpdate
    
    if not existing_figure or len(existing_figure.get('data')) == 0:
        # if no existing figure, create a new one
        region = selected_regions[0]
        scenario = selected_scenarios[0]
        new_trace_df = DataRetrieval(db, output_name, region, scenario).single_output_df_to_graph(lower_bound, upper_bound)
        traces_to_add = NewTimeSeries(output_name, region, scenario, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()

        fig = go.Figure(traces_to_add)
        fig.update_layout(
            height = 625,
            margin = dict(t = 40, b = 0, l = 10),
            title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name]),
            yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
            xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
            plot_bgcolor = plot_bgcolor,
            uirevision = output_name,
            datarevision = str(id(fig))
        )
        fig.update_xaxes(showgrid = toggle_gridlines)
        fig.update_yaxes(showgrid = toggle_gridlines)

        return fig.to_dict()

    else:
        existing_fig = go.Figure(existing_figure)
        # if the existing figure is a histogram, then we need to start from scratch
        # otherwise, the new time series is added to the histogram figure
        if existing_fig.data[0]["type"] == "histogram":
            fig = ModifyOutputTimeseries(output_name, selected_regions, selected_scenarios, go.Figure(), {"color": color_scheme}, db, lower_bound, upper_bound, True).create_new_figure()

            fig.update_layout(
                height = 625,
                margin = dict(t = 40, b = 0, l = 10),
                title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name]),
                yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
                xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
                plot_bgcolor = plot_bgcolor,
                uirevision = output_name,
                datarevision = str(id(fig))
            )
            fig.update_xaxes(showgrid = toggle_gridlines)
            fig.update_yaxes(showgrid = toggle_gridlines)

            return fig.to_dict()

        # the figure will only change if the regions/scenarios/output has changed
        # to make sure the figure is changed when styling options are changed, check if any of the styling options have changed
        if trigger_id == "output-color-scheme" or trigger_id == "time-series-plot-apply-styling-changes" or trigger_id == "time-series-plot-apply-bound-changes":
            change_fig = True
        else:
            change_fig = False
        
        # When output dropdown changes, we MUST start with a fresh figure
        # This is because UIDs won't match between different outputs (especially custom vs regular)
        # and ModifyOutputTimeseries.remove_traces() won't be able to remove old traces
        if trigger_id == "output-dropdown":
            existing_fig = go.Figure()
            change_fig = True
        
        fig = ModifyOutputTimeseries(output_name, selected_regions, selected_scenarios, existing_fig, {"color": color_scheme}, db, lower_bound, upper_bound, change_fig).create_new_figure()
        if db == db_full:
            if output_name not in readability_obj.naming_dict_long_names_first:
                just_name = json.loads(output_name)["name"]
            else:
                just_name = readability_obj.naming_dict_long_names_first[output_name]
        else:
            if output_name not in readability_obj.publication_naming_dict_long_names_first:
                just_name = json.loads(output_name)["name"]
            else:
                just_name = readability_obj.publication_naming_dict_long_names_first[output_name]
        title_text = "Time Series for " + just_name
        fig.update_layout(
            height = 625,
            margin = dict(t = 40, b = 0, l = 10),
            title_text = title_text,
            yaxis = dict(title = dict(text = just_name, font = dict(size = 16))),
            xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
            plot_bgcolor = plot_bgcolor,
            hovermode = 'closest',  # Ensure interactivity is preserved
            # uirevision forces Plotly.js to completely reset the figure when output changes
            # This prevents stale renders when switching from custom variables to regular outputs
            uirevision = output_name,
            # Add timestamp to force complete re-render (diagnostic)
            datarevision = str(id(fig))
        )
        fig.update_xaxes(showgrid = toggle_gridlines)
        fig.update_yaxes(showgrid = toggle_gridlines)

        # Convert to dict to ensure clean serialization to browser
        return fig.to_dict()

        # this code describes a histogram that was previously available, but was removed to keep things simple
        # current_trace_info = TraceInfo(existing_figure)
        # if current_trace_info.type[0] == "histogram": # means active figure is histogram, so need to generate scatter 
        #     for region, scenario in product(selected_regions, selected_scenarios):
        #         new_trace_df = DataRetrieval(db, output_name, region, scenario).single_output_df_to_graph(lower_bound, upper_bound)
        #         traces_to_add = NewTimeSeries(output_name, region, scenario, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()

        #     try:
        #         title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name])
        #     except KeyError:
        #         title_text = "Time Series for {}".format(json.loads(output_name)["name"])

        #     fig = go.Figure(traces_to_add)
        #     fig.update_layout(
        #         height = 625,
        #         margin = dict(t = 40, b = 0, l = 10),
        #         title_text = title_text,
        #         # yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
        #         xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
        #         plot_bgcolor = plot_bgcolor
        #     )
        #     fig.update_xaxes(showgrid = toggle_gridlines)
        #     fig.update_yaxes(showgrid = toggle_gridlines)
        #     return fig
        # else:
        #     combos_with_trace_name = list(product(selected_regions, selected_scenarios, ["lower", "median", "upper"]))
        #     current_traces = current_trace_info.traces
        #     custom_data_just_strings = [i[0] for i in current_trace_info.custom_data]
        #     existing_selections = set(custom_data_just_strings)
        #     all_selections = set(["{}|{}|{}|{}".format(output_name, reg, sce, trace_name) for reg, sce, trace_name in combos_with_trace_name])

        #     # changes to make
        #     if trigger_id == "output-color-scheme":
        #         # without this logic, the color of the figure will not update when the color scheme is changed
        #         # what this does is take all existing plots and changes color according to what the new color scheme dictates
        #         existing_figure_data = existing_figure["data"]
        #         for i in existing_figure_data:
        #             trace_name = i["customdata"][0]
        #             region = trace_name.split(' ')[-3]
        #             scenario = trace_name.split(' ')[-2]
        #             if color_scheme == "by-region":
        #                 color = Color().get_color_for_timeseries(color_scheme, region)
        #             elif color_scheme == "by-scenario":
        #                 color = Color().get_color_for_timeseries(color_scheme, scenario)
        #             elif color_scheme == "standard":
        #                 color = Color().get_color_for_timeseries(color_scheme, [region, scenario])

        #             i["line"]["color"] = color

        #     no_change = existing_selections.intersection(all_selections)
        #     to_delete = existing_selections.difference(all_selections)
        #     to_add = all_selections.difference(existing_selections)

        #     # removing traces - well, keeping ones that haven't been removed
        #     indices_to_delete = [custom_data_just_strings.index(i) for i in to_delete]
        #     indices_to_keep = [i for i in range(len(current_traces)) if i not in indices_to_delete]
        #     current_traces = [current_traces[i] for i in indices_to_keep]

        #     # adding traces
        #     new_traces = []
        #     decomposed_traces_to_add = set([i.split("|")[0] + "|" + i.split("|")[1] + "|" + i.split("|")[2] for i in to_add])
        #     for i in decomposed_traces_to_add:
        #         output, reg, sce = tuple(i.split("|"))
        #         new_trace_df = DataRetrieval(db, output_name, reg, sce).single_output_df_to_graph(lower_bound, upper_bound)
        #         traces_to_add = NewTimeSeries(output_name, reg, sce, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()
        #         new_traces += traces_to_add

        #     if output_name not in options_obj.outputs:
        #         title_text = "Time Series for " + json.loads(output_name)["name"]
        #     else:
        #         title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name])

        #     fig = go.Figure(data = current_traces + new_traces)
        #     fig.update_layout(
        #         height = 625,
        #         margin = dict(t = 40, b = 0, l = 10),
        #         title_text = title_text,
        #         # yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
        #         xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
        #         plot_bgcolor = plot_bgcolor
        #     )
        #     fig.update_xaxes(showgrid = toggle_gridlines)
        #     fig.update_yaxes(showgrid = toggle_gridlines)
        #     return fig

    # else:
    #     if not selected_regions or not selected_scenarios:
    #         raise PreventUpdate

    #     styling_options = {"color": color_scheme}
    #     fig = OutputHistograms(output_name, selected_regions, selected_scenarios, year, db, styling_options = styling_options).make_plot()

    #     if output_name not in options_obj.outputs:
    #         title_text = "Histograms for " + output_name.split("-")[-1]
    #     else:
    #         title_text = "Histograms for {}".format(readability_obj.naming_dict_long_names_first[output_name])
    #     fig.update_layout(title_text = title_text)
    #     fig.update_layout(
    #         height = 550,
    #         margin = dict(t = 70, b = 20, l = 10)
    #     )
        return fig

# callback for data download - doing this separately helps make everything more organized
@app.callback(
    Output("time-series-plot-download-csv", "data"),
    Input("time-series-plot-download-data-button", "n_clicks"),
    State('output-dropdown', 'value'),
    State('region-dropdown', 'value'),
    State('scenario-dropdown', 'value'),
    State("time-series-plot-upper-bound", "value"),
    State("time-series-plot-lower-bound", "value"),
    State("overview-data-dropdown", "value"),
    prevent_initial_call=True
)
def timeseries_data_download(n_clicks, output, regions, scenarios, upper_bound, lower_bound, overview_data_dropdown):
    if not output or not regions or not scenarios or not n_clicks:
        raise PreventUpdate
    
    # Select appropriate database
    if overview_data_dropdown == "full":
        db = db_full
    else:
        db = db_publication
    
    full_data_df = pd.DataFrame()
    for reg in regions:
        for sce in scenarios:
            df = DataRetrieval(db, output, reg, sce).single_output_df()
            df["Region"] = [reg] * len(df)
            df["Scenario"] = [sce] * len(df)
            full_data_df = pd.concat([full_data_df, df], axis = 0)
    
    # Create a safe filename (handle custom variables)
    if output.startswith('{'):
        filename = "eppa_dashboard_data_custom_variable.csv"
    else:
        filename = f"eppa_dashboard_data_{output}.csv"
    
    return dcc.send_data_frame(full_data_df.to_csv, filename, index=False)

# callback for high-res image download
@app.callback(
    Output("time-series-plot-download-image", "data"),
    Input("time-series-plot-download-image-button", "n_clicks"),
    State('output-time-series-plot', 'figure'),
    State('output-dropdown', 'value'),
    prevent_initial_call=True
)
def timeseries_plot_image_download(n_clicks, figure_data, output):
    if not n_clicks or not figure_data:
        raise PreventUpdate
    
    figure = go.Figure(figure_data)
    
    # Create a safe filename
    if output and output.startswith('{'):
        filename = "time_series_custom_variable.png"
    else:
        filename = f"time_series_{output}.png" if output else "time_series_plot.png"

    return dcc.send_bytes(figure.to_image(format="png", scale=3), filename)

# callback for SVG download
@app.callback(
    Output("time-series-plot-download-svg", "data"),
    Input("time-series-plot-download-svg-button", "n_clicks"),
    State('output-time-series-plot', 'figure'),
    State('output-dropdown', 'value'),
    prevent_initial_call=True
)
def timeseries_plot_svg_download(n_clicks, figure_data, output):
    if not n_clicks or not figure_data:
        raise PreventUpdate
    
    figure = go.Figure(figure_data)
    
    # Create a safe filename
    if output and output.startswith('{'):
        filename = "time_series_custom_variable.svg"
    else:
        filename = f"time_series_{output}.svg" if output else "time_series_plot.svg"

    return dcc.send_bytes(figure.to_image(format="svg"), filename)

# Citation download callbacks - triggered alongside main downloads
@app.callback(
    Output("time-series-plot-download-citation-csv", "data"),
    Input("time-series-plot-download-data-button", "n_clicks"),
    prevent_initial_call=True
)
def download_citation_with_csv(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return dcc.send_string(CITATION_TEXT, "suggested_citation.txt")

@app.callback(
    Output("time-series-plot-download-citation-image", "data"),
    Input("time-series-plot-download-image-button", "n_clicks"),
    prevent_initial_call=True
)
def download_citation_with_image(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return dcc.send_string(CITATION_TEXT, "suggested_citation.txt")

@app.callback(
    Output("time-series-plot-download-citation-svg", "data"),
    Input("time-series-plot-download-svg-button", "n_clicks"),
    prevent_initial_call=True
)
def download_citation_with_svg(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return dcc.send_string(CITATION_TEXT, "suggested_citation.txt")

# callback for inputs
@app.callback(
    Output("input-dist-graph", "figure"),
    Input("input-dist-options", "value"))
def update_input_dist(inputs):
    if not inputs:
        raise PreventUpdate
    figure = InputDistributionAlternate(inputs).make_plot()

    return figure

# callback for i/o mapping
@app.callback(
    Output("input-output-mapping-figure-container", "hidden"),
    Output("input-output-mapping-figure", "figure"),
    Output("input-output-mapping-output", "disabled"),
    Output("input-output-mapping-percentile", "disabled"),
    Output("input-output-mapping-setting", "disabled"),
    Output("input-output-mapping-run-count", "children"),
    State("input-output-mapping-output", "value"),
    State("input-output-mapping-region", "value"),
    State("input-output-mapping-scenario", "value"),
    State("input-output-mapping-year", "value"),    
    State("input-output-mapping-percentile", "value"),
    State("input-output-mapping-setting", "value"),
    State("input-output-mapping-n-estimators", "value"),
    State("input-output-mapping-max-depth", "value"),
    Input("input-output-mapping-mode", "value"),  # Input so mode change triggers callback
    State("slider-custom-io-mapping-1", "value"),
    State("slider-custom-io-mapping-2", "value"),
    State("slider-custom-io-mapping-3", "value"),
    State("custom-io-mapping-dropdown-1", "value"),
    State("custom-io-mapping-dropdown-2", "value"),
    State("custom-io-mapping-dropdown-3", "value"),
    Input("input-output-mapping-update-all-settings", "n_clicks"),
    State("overview-data-dropdown", "value"),
    prevent_initial_call = True
)
def update_io_mapping_figure(output, region, scenario, year, percentile, setting, n_estimators, max_depth, mode, slider_1, slider_2, slider_3, dropdown_1, dropdown_2, dropdown_3, update_all_settings, publication_output):
    if not region or not output or not year or not scenario:
        raise PreventUpdate
    
    if publication_output == "full":
        db = db_full
    else:
        db = db_publication

    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]
    gt = True if setting == "above" else False

    # If only the mode changed, just update UI visibility without running analysis
    if trigger_id == "input-output-mapping-mode":
        if mode == "standard":
            # Show standard UI, hide filtered UI
            return True, dash.no_update, False, False, False, ""
        else:
            # Show filtered UI, hide standard UI elements
            return False, dash.no_update, True, True, True, ""

    if mode == "standard":
        df = DataRetrieval(db, output, region, scenario, year).mapping_df()
        unstyled_figure = InputOutputMappingPlot(output, region, scenario, year, df, threshold = percentile, gt = gt, n_estimators = n_estimators, max_depth = max_depth)
        finished_figure = FinishedFigure(unstyled_figure).make_finished_figure()

        return True, finished_figure, False, False, False, ""

    if mode == "filtered":
        outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
        if not outputs_to_include:
            raise PreventUpdate
            
        df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()
        
        # Apply percentile constraints to filter runs
        constraint_df = df.copy()
        constraint_df["in_constraint_range"] = 1  # Initialize all rows as within constraint range
        
        # Iterate through each dropdown/slider pair to apply constraints
        sliders = [slider_1, slider_2, slider_3]
        for i, (dropdown, slider) in enumerate(zip([dropdown_1, dropdown_2, dropdown_3], sliders)):
            if dropdown and slider:  # Ensure dropdown has a selection
                col_name = readability_obj.naming_dict_long_names_first[dropdown]
                lower_bound, upper_bound = np.percentile(df[col_name], slider)
                constraint_df["in_constraint_range"] &= (
                    (constraint_df[col_name] >= lower_bound) & 
                    (constraint_df[col_name] <= upper_bound)
                ).astype(int)
        
        # Count runs that meet all constraints
        total_runs = len(constraint_df)
        selected_runs = constraint_df["in_constraint_range"].sum()
        run_count_text = f"Selected {selected_runs} of {total_runs} runs based on percentile constraints"
        
        # Generate the feature importance plot
        fig = FilteredInputOutputMappingPlot(
            constraint_df, region, scenario, year, 
            n_estimators=n_estimators, random_forest_depth=max_depth
        ).make_plot()

        return False, fig, True, True, True, run_count_text

# callback for i/o tree
@app.callback(
    Output("full-cart-tree", "figure"), 
    State("input-output-mapping-output", "value"),
    State("input-output-mapping-region", "value"),
    State("input-output-mapping-scenario", "value"),
    State("input-output-mapping-year", "value"),
    State("input-output-mapping-mode", "value"),
    State("full-cart-tree-depth-dropdown", "value"),
    State("slider-custom-io-mapping-1", "value"),
    State("slider-custom-io-mapping-2", "value"),
    State("slider-custom-io-mapping-3", "value"),
    State("custom-io-mapping-dropdown-1", "value"),
    State("custom-io-mapping-dropdown-2", "value"),
    State("custom-io-mapping-dropdown-3", "value"),
    State("overview-data-dropdown", "value"),
    State("input-output-mapping-percentile", "value"),
    State("input-output-mapping-setting", "value"),
    Input("input-output-mapping-update-all-settings", "n_clicks"),
    prevent_initial_call = True
)
def update_tree(output, region, scenario, year, mode, cart_depth, slider_1, slider_2, slider_3, dropdown_1, dropdown_2, dropdown_3, publication_output, percentile, setting, update_all_settings_n_clicks):
    if not cart_depth or not output or not region or not scenario or not year:
        raise PreventUpdate
    
    if publication_output == "full":
        db = db_full
    else:
        db = db_publication

    gt = True if setting == "above" else False

    if mode == "standard":
        df = DataRetrieval(db, output, region, scenario, year).mapping_df()
        _, y = InputOutputMapping(output, region, scenario, year, df, threshold = percentile, gt = gt, cart_depth = cart_depth).preprocess_for_classification()
        tree = InputOutputMapping(output, region, scenario, year, df, threshold = percentile, gt = gt, cart_depth = cart_depth).CART()
        fig = PlotTree(tree, y).make_plot(show = False)

        return fig
    
    if mode == "filtered":
        outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
        if not outputs_to_include:
            return go.Figure()
            
        df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()

        constraint_df = df.copy()
        constraint_df["in_constraint_range"] = 1  # Initialize all rows as within constraint range
        
        # Iterate through each dropdown/slider pair to apply constraints
        for dropdown, slider in zip([dropdown_1, dropdown_2, dropdown_3], [slider_1, slider_2, slider_3]):
            if dropdown and slider:  # Ensure dropdown has a selection
                lower_bound, upper_bound = np.percentile(df[readability_obj.naming_dict_long_names_first[dropdown]], slider)
                constraint_df["in_constraint_range"] &= ((constraint_df[readability_obj.naming_dict_long_names_first[dropdown]] >= lower_bound) & (constraint_df[readability_obj.naming_dict_long_names_first[dropdown]] <= upper_bound)).astype(int)

        filtered_mapping = FilteredInputOutputMappingPlot(constraint_df, region, scenario, year, cart_depth = cart_depth)
        tree = filtered_mapping.CART()
        fig = PlotTree(tree, filtered_mapping.y_discrete).make_plot(show = False)

        return fig

# callback for permutation importance
@app.callback(
    Output("input-output-mapping-permutation-importance", "figure"),
    State("input-output-mapping-output", "value"),
    State("input-output-mapping-region", "value"),
    State("input-output-mapping-scenario", "value"),
    State("input-output-mapping-year", "value"),
    State("input-output-mapping-n-estimators", "value"),
    State("input-output-mapping-max-depth", "value"),
    State("input-output-mapping-mode", "value"),
    State("slider-custom-io-mapping-1", "value"),
    State("slider-custom-io-mapping-2", "value"),
    State("slider-custom-io-mapping-3", "value"),
    State("custom-io-mapping-dropdown-1", "value"),
    State("custom-io-mapping-dropdown-2", "value"),
    State("custom-io-mapping-dropdown-3", "value"),
    Input("input-output-mapping-update-all-settings", "n_clicks"),
    State("overview-data-dropdown", "value"),
    prevent_initial_call = True
)
def update_permutation_importance(output, region, scenario, year, n_estimators, max_depth, mode, slider_1, slider_2, slider_3, dropdown_1, dropdown_2, dropdown_3, update_all_settings, publication_output):
    if not region or not output or not year or not scenario:
        raise PreventUpdate
    
    if publication_output == "full":
        db = db_full
    else:
        db = db_publication
    
    if mode == "standard":
        df = DataRetrieval(db, output, region, scenario, year).mapping_df()
        unstyled_figure = PermutationImportance(df, output, region, scenario, year, n_estimators = n_estimators, max_depth = max_depth)
        finished_figure = FinishedFigure(unstyled_figure).make_finished_figure()
        return finished_figure
    
    if mode == "filtered":
        outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
        if not outputs_to_include:
            return go.Figure()
            
        df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()
        
        # Apply percentile constraints to filter runs
        constraint_df = df.copy()
        constraint_df["in_constraint_range"] = 1
        
        for dropdown, slider in zip([dropdown_1, dropdown_2, dropdown_3], [slider_1, slider_2, slider_3]):
            if dropdown and slider:
                col_name = readability_obj.naming_dict_long_names_first[dropdown]
                lower_bound, upper_bound = np.percentile(df[col_name], slider)
                constraint_df["in_constraint_range"] &= (
                    (constraint_df[col_name] >= lower_bound) & 
                    (constraint_df[col_name] <= upper_bound)
                ).astype(int)
        
        # Use FilteredInputOutputMapping for permutation importance
        from analysis import FilteredInputOutputMapping
        filtered_mapping = FilteredInputOutputMapping(
            constraint_df, region, scenario, year, 
            n_estimators=n_estimators, random_forest_depth=max_depth
        )
        results = filtered_mapping.permutation_importance()
        
        # Create figure from results
        fig = go.Figure()
        if results:
            fig.add_trace(go.Bar(
                x=[k["variable"] for k in results], 
                y=[k["mean"] for k in results], 
                error_y=dict(type="data", array=[k["std"] for k in results])
            ))
            fig.update_layout(
                title="Permutation Importance (Filtered Mode)",
                xaxis_title="Feature",
                yaxis_title="Importance"
            )
        
        return fig

# callback for o/o mapping
@app.callback(
    Output("output-output-mapping-figure-container", "hidden"),
    Output("output-output-mapping-figure", "figure"),
    Output("output-output-mapping-output", "multi"),
    Output("output-output-mapping-output", "options"),
    Output("output-output-mapping-output", "disabled"),
    Output("output-output-mapping-run-count", "children"),
    Input("output-output-mapping-mode", "value"),  # Keep as Input so mode switching is immediate
    Input("output-output-mapping-update", "n_clicks"),  # Update button triggers the callback
    State("output-output-mapping-output", "value"),
    State("output-output-mapping-region", "value"),
    State("output-output-mapping-scenario", "value"),
    State("output-output-mapping-year", "value"),
    State("custom-oo-mapping-dropdown-1", "value"),
    State("custom-oo-mapping-dropdown-2", "value"),
    State("custom-oo-mapping-dropdown-3", "value"),
    State("slider-custom-oo-mapping-1", "value"),
    State("slider-custom-oo-mapping-2", "value"),
    State("slider-custom-oo-mapping-3", "value"),
    State("output-output-mapping-output", "options"),
    State("overview-data-dropdown", "value"),
    prevent_initial_call = True
)
def update_output_output_mapping(mode, update_clicks, output, region, scenario, year, dropdown_1, dropdown_2, dropdown_3, slider_1, slider_2, slider_3, options, publication_output):
    if not region or not output or not scenario or not year:
        raise PreventUpdate
    
    if publication_output == "full":
        db = db_full
    else:
        db = db_publication

    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]

    # If only the mode changed, just update UI visibility without running analysis
    if trigger_id == "output-output-mapping-mode":
        if mode == "standard":
            # Enable dropdown in standard mode
            return True, dash.no_update, False, dash.no_update, False, ""
        else:
            # Disable dropdown in filtered mode (all outputs are used)
            return False, dash.no_update, True, dash.no_update, True, ""

    if mode == "standard":
        df = DataRetrieval(db, output, region, scenario, year).mapping_df()
        fig = OutputOutputMappingPlot(db, output, region, scenario, year, df)
        finished_fig = FinishedFigure(fig).make_finished_figure()

        return True, finished_fig, False, options, False, ""

    if mode == "filtered":
        # Fetch the outputs selected for constraint filtering
        outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
        if not outputs_to_include:
            raise PreventUpdate
        
        df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()

        # Apply constraints based on percentile sliders
        constraint_df = df.copy()
        constraint_df["in_constraint_range"] = 1

        for dropdown, slider in zip([dropdown_1, dropdown_2, dropdown_3], [slider_1, slider_2, slider_3]):
            if dropdown:
                output_name = readability_obj.naming_dict_long_names_first[dropdown] if dropdown in Options().outputs else json.loads(dropdown)["name"]
                lower_bound, upper_bound = np.percentile(df[output_name], slider)
                constraint_df["in_constraint_range"] &= ((constraint_df[output_name] >= lower_bound) & (constraint_df[output_name] <= upper_bound)).astype(int)

        # Count runs that meet constraints
        runs_selected = constraint_df["in_constraint_range"].sum()
        total_runs = len(constraint_df)
        run_count_text = f"Selected {runs_selected} of {total_runs} runs based on percentile constraints."

        # FilteredOutputOutputMappingPlot now uses ALL outputs as inputs, 
        # with in_constraint_range as the target variable
        fig = FilteredOutputOutputMappingPlot(db, constraint_df, region, scenario, year).make_plot()

        if options[0]["value"] == "all":
            new_options = options
        else:
            new_options = [{"label": "All", "value": "all"}] + options
        
        # Disable dropdown in filtered mode (all outputs are used)
        return False, fig, True, new_options, True, run_count_text
    
    # this mode has been deprecated, but I'm leaving the code here for now in case we need it in the future
    # the purpose of this mode was to look at the upper and lower ranges of the output variables, but that's 
    # just a more specific case of the filtered mode
    # if mode == "high-low":
    #     df_upper = DataRetrieval(db, output, region, scenario, year).mapping_df()
    #     df_lower = df_upper.copy()

        
    #     fig = OutputOutputMappingPlot(db, output, region, scenario, year, df)
    #     finished_fig = FinishedFigure(fig).make_finished_figure()

    #     return go.Figure(), True, True, finished_fig, False, options

    #     return filter_fig, False, False, fig, True, options

# callback for regional heatmaps
@app.callback(Output("regional-heatmaps-figure", "figure"),
              Input("regional-heatmaps-apply-button", "n_clicks"),
              State("regional-heatmaps-output", "value"),
              State("regional-heatmaps-region", "value"),
              State("regional-heatmaps-scenario", "value"),
              State("overview-data-dropdown", "value"),
              prevent_initial_call = True)
def update_regional_heatmaps_figure(n_clicks, output, regions, scenarios, publication_output):
    """
    Generate regional heatmaps showing feature importance.
    All data fetching and processing logic is encapsulated in RegionalHeatmaps class.
    """
    if not regions or not output or not scenarios:
        raise PreventUpdate
    
    db = db_full if publication_output == "full" else db_publication
    
    # RegionalHeatmaps handles all data fetching and model training internally
    fig = RegionalHeatmaps(db, output, regions, scenarios).fig

    return fig

# callback for choropleth mapping
@app.callback(
    Output("choropleth-mapping-figure", "figure"),
    Input("choropleth-mapping-update", "n_clicks"),
    State("choropleth-mapping-output", "value"),
    State("choropleth-mapping-scenario", "value"),
    State("choropleth-mapping-year", "value"),
    State("overview-data-dropdown", "value"),
    prevent_initial_call = True
)
def update_choropleth_figure(n_clicks, output, scenario, year, publication_output):
    if not n_clicks or not scenario or not output or not year:
        raise PreventUpdate
    
    if publication_output == "full":
        db = db_full
    else:
        db = db_publication
    
    df = DataRetrieval(db, output, "GLB", scenario, year).choropleth_map_df(5, 95)
    unstyled_fig = ChoroplethMap(df, output, scenario, year, 5, 95)
    finished_fig = FinishedFigure(unstyled_fig).make_finished_figure()

    return finished_fig

# callback for ts clustering
@app.callback(
    Output("ts-clustering-plot", "figure"),
    Output('ts-clustering-random-forest-plot', 'figure'),
    # Output("ts-clustering-cart-tree-plot", "figure"),
    Input("ts-clustering-update", "n_clicks"),
    State("ts-clustering-output", "value"),
    State("ts-clustering-region", "value"),
    State("ts-clustering-scenario", "value"),
    State("ts-clustering-n-clusters", "value"),
    State("ts-clustering-metric", "value"),
    State("overview-data-dropdown", "value"),
    prevent_initial_call = True
)
def update_ts_clustering_figure(n_clicks, output, region, scenario, n_clusters, metric, publication_output):
    if not n_clicks or not region or not output or not scenario:
        raise PreventUpdate
    
    if publication_output == "full":
        db = db_full
    else:
        db = db_publication

    df = DataRetrieval(db, output, region, scenario).single_output_df()
    fig_obj = TimeSeriesClusteringPlot(df, output, region, scenario, n_clusters = n_clusters, metric = metric)

    cart_fig_obj = TimeSeriesClusteringPlotCART(df, output, region, scenario, n_clusters = n_clusters, metric = metric)
    finished_figure = FinishedFigure(fig_obj).make_finished_figure()
    finished_figure_cart = FinishedFigure(cart_fig_obj).make_finished_figure()

    return finished_figure, finished_figure_cart

# callback for dynamic display of custom variables tab
@app.callback(
    Output("custom-vars-output-dropdown-div", "children"),
    Output("custom-vars-operation", "value"),
    Output("custom-vars-var-name", "value"),
    Input("custom-vars-operation", "value"),
    Input("close-centered", "n_clicks"),
    State("output-dropdown", "options"),
    State("custom-vars-var-name", "value")
)
def dynamic_custom_variables_fill(operation_type, n_clicks, options, var_name):
    if not operation_type:
        raise PreventUpdate

    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]

    if trigger_id == "custom-vars-operation":
        if operation_type == "division":
                return [(
                    dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                        placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                    html.Span("by", style = {'margin-right': '10px'}),
                    dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", options = options,
                                placeholder = "Output 2",
                                style = {"width": "500px"})
                        ), operation_type, var_name]
        if operation_type == "addition":
            return [(dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                        placeholder = "Select Outputs to Add", style = {"margin-right": "10px", "width": "1000px"}, multi = True),
                        dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", style = {"display": "none"})), operation_type, var_name]
        if operation_type == "multiplication":
                return [(
                    dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                        placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                    dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", options = options,
                                placeholder = "Output 2",
                                style = {"width": "500px"})
                        ), operation_type, var_name]
        if operation_type == "subtraction":
                return [(
                    dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                        placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                    dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", options = options,
                                placeholder = "Output 2",
                                style = {"width": "500px"})
                        ), operation_type, var_name]

    if trigger_id == "close-centered":
        return [[], "", ""]

# callback for custom variables
@app.callback(
    Output("stored-custom-variables", "data"),
    State("stored-custom-variables", "data"),
    State("custom-vars-var-name", "value"),
    State("custom-vars-output-1-dropdown-div-1", "value"),
    State("custom-vars-output-2-dropdown-div-2", "value"),
    State("custom-vars-operation", "value"),
    Input("create-custom-variable-button", "n_clicks"),
    prevent_initial_call = True
)
def update_custom_variables(current_data, var_name, output_1, output_2, operation, n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    current_data = current_data or {}
    if operation == "division":
        current_data[var_name] = {"operation": operation, "output1": output_1, "output2": output_2, "name": var_name}
    if operation == "addition":
        current_data[var_name] = {"operation": operation, "outputs": output_1, "name": var_name}
    if operation == "multiplication":
        current_data[var_name] = {"operation": operation, "output1": output_1, "output2": output_2, "name": var_name}
    if operation == "subtraction":
        current_data[var_name] = {"operation": operation, "output1": output_1, "output2": output_2, "name": var_name}

    return current_data

# callback for modal pop-up on custom variables tab
@app.callback(
    Output("custom-variable-created-modal", "is_open"),
    Input("create-custom-variable-button", "n_clicks"), 
    Input("close-centered", "n_clicks"),
    State("custom-variable-created-modal", "is_open"),
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# callback for STRESS platform connection graph
@app.callback(
    Output("stress-connection-graph", "figure"),
    State("stress-connection-inputs", "value"),
    State("stress-connection-outputs", "value"),
    State("stress-connection-color", "value"),
    State("stress-connection-region", "value"),
    State("stress-connection-scenario", "value"),
    State("stress-connection-year", "value"),
    State("stress-connection-graph", "restyleData"),
    Input("stress-connection-button", "n_clicks")
)
def update_stress_connection_graph(inputs, outputs, color, region, scenario, year, constraint_range, n_clicks_apply):
    if not inputs or not outputs or not color:
        raise PreventUpdate

    fig, df = STRESSPlatformConnection(db, inputs, outputs, color, region, scenario, year).make_plot()

    return fig

# callback for STRESS platform connection table
@app.callback(
    Output("stress-connection-table", "children"),
    State("stress-connection-inputs", "value"),
    State("stress-connection-outputs", "value"),
    State("stress-connection-color", "value"),
    State("stress-connection-region", "value"),
    State("stress-connection-scenario", "value"),
    State("stress-connection-year", "value"),
    State("stress-connection-graph", "figure"),
    Input("stress-connection-update-table-button", "n_clicks")
)
def update_stress_connection_table(inputs, outputs, color, region, scenario, year, figure, n_clicks_table):
    if not inputs or not outputs or not color:
        raise PreventUpdate

    fig, df = STRESSPlatformConnection(db, inputs, outputs, color, region, scenario, year).make_plot()
    if figure:
        print(figure["data"][0]["dimensions"][0]["line"])

    return dbc.Table.from_dataframe(df, striped = True, bordered = True, hover = True, size = "sm")

if __name__ == '__main__':
    app.run(debug = True, host = "0.0.0.0")

    # discarded components

    '''
    Gradient text
    html.H2(style = {"background-image": "linear-gradient(to right, violet, lightblue)",
        "-webkit-background-clip": "text",
        "color": "transparent",
        "background-clip": "text",
        "-webkit-text-fill-color": "transparent"},
        children = ["MIT Joint Program on the Science and Policy of Global Change"]),
    html.H3(style = {"background-image": "linear-gradient(to right, lightblue, orange)",
        "-webkit-background-clip": "text",
        "color": "transparent",
        "background-clip": "text",
        "-webkit-text-fill-color": "transparent"},
        children = ["Data Visualization Dashboard"]
    )
    '''
     