# use venv when running this code
import dash
from dash.exceptions import PreventUpdate
from dash import html, dcc, callback_context
import dash_bootstrap_components as dbc
from sql_utils import SQLConnection, DataRetrieval
from styling import Options, Readability, Color, FinishedFigure
from dash.dependencies import Input, Output, State, MATCH
from figure import NewTimeSeries, InputDistribution, InputOutputMappingPlot, TraceInfo, OutputHistograms, ChoroplethMap, TimeSeriesClusteringPlot
import numpy as np
import plotly.graph_objects as go
from itertools import product
from pprint import pprint
from dash_iconify import DashIconify

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.PULSE, dbc.icons.BOOTSTRAP],
                suppress_callback_exceptions = True)

# initialize SQL database and other UI elements
db = SQLConnection("all_data_jan_2024")
readability_obj = Readability()
options_obj = Options()

# construct navigation bar
jp_logo = r"assets\images\JPSPGC.logo.color.png"
navbar = dbc.Navbar(
    class_name = "navbar navbar-expand-lg custom-navbar",
    color = "#3e8cda",
    dark = True,
    children = [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(style = {"margin-left": 20}, children = html.Img(src = jp_logo, height = "60px")),
                    dbc.Col(dbc.NavbarBrand("MIT EPPA Model - Data Visualization Dashboard", className = "ms-2")),
                ],
                align = "center",
                className = "g-0",
            ),
            href = "https://globalchange.mit.edu/",
            target = "_blank",
            style = {"textDecoration": "none"},
        ),
        dbc.NavbarToggler(id = "navbar-toggler", n_clicks = 0),
        dbc.Row(style = {"margin-left": 250},
            children = [
                dbc.Col(children = 
                    html.A(
                        "Dashboard Guide",
                        href = "https://www.notion.so/thekenster/MIT-JP-Data-Visualization-Dashboard-User-Guide-f696462c92bc4280a261ef67b1ab3bf3",
                        target = "_blank",
                        style = {"textDecoration": "none", "color": "white", "text-align": "right"},
                    ),
                    width = "auto"
                ),
                dbc.Col(
                    html.A(children = 
                        "Upcoming Publication",
                        href = "https://globalchange.mit.edu/publication/18092",
                        target = "_blank",
                        style = {"textDecoration": "none", "color": "white", "text-align": "right", "padding": 100}
                    ),
                    width = "auto"
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
                                    html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Output Visualization"),
                                    DashIconify(icon = "feather:info", width = 60, style = {"padding": 10, "color": "#9AC1F4"})
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
                                            dbc.Col(style = {},
                                                width = 3,
                                                children = [
                                                    dbc.Row(html.Div("View", className = "text-primary")),
                                                    dbc.Row(
                                                        children = [
                                                            dcc.Dropdown(id = "chart-options", options = [{"label": "Time Series", "value": "time-series"},
                                                                                    {"label": "Distribution by Year", "value": "dist-by-year"}],
                                                                        value = "time-series")
                                                        ]
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
                            dcc.Graph(id = "output-time-series-plot")
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
                            dbc.Checklist(id = "scenario-dropdown", style = {"padding": 10}, options = [{"label":k, "value":v} for k, v in options_obj.scenario_display_names_rev.items()],
                                        inline = True, value = ["2C_med"])
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
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Input Visualization"),
                                                DashIconify(icon = "feather:info", width = 60, style = {"padding": 10, "color": "#9AC1F4"})
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
                                                        ),
                                                        dbc.Col(width = 2,
                                                            children = [
                                                                dbc.Row(html.Div("Input to Highlight", className = "text-primary")),
                                                                dbc.Row(
                                                                    dcc.Dropdown(style = {"width": 200},
                                                                        id = "expanded-view-input-dist-options",
                                                                        options = [{'label': i, 'value': i} for i in Options().input_names],
                                                                        value = "wind"
                                                                        )
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
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Input/Output Mapping"),
                                                DashIconify(icon = "feather:info", width = 60, style = {"padding": 10, "color": "#9AC1F4"})
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
                                dcc.Dropdown(id = "input-output-mapping-output",
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
                                html.Div("Region", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "input-output-mapping-regions",
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
                                    value = 2050)
                                ]
                            ),
                        dbc.Col(width = 10,
                            children = [
                                dcc.Loading([dcc.Graph(id = "input-output-mapping-figure")]),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

output_output_mapping = html.Div()

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
                                                html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Choropleth Mapping"),
                                                DashIconify(icon = "feather:info", width = 60, style = {"padding": 10, "color": "#9AC1F4"})
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
                        dbc.Col(
                            children = [
                                html.Div("Scenario", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "choropleth-mapping-scenario",
                                    options = [{'label': i, 'value': i} for i in Options().scenarios],
                                    value = "Ref"
                                ),
                                html.Div("Year", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "choropleth-mapping-year",
                                    options = [{'label': i, 'value': i} for i in Options().years],
                                    value = 2050)
                                ]
                            ),
                        dbc.Col(
                            children = [
                                dbc.Row(
                                    children = [dcc.Loading([dcc.Graph(id = "choropleth-mapping-figure")])]
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
                                    html.H4(style = {"padding": 10, "color": "#9AC1F4"}, children = "Time Series Clustering"),
                                    DashIconify(icon = "feather:info", width = 60, style = {"padding": 10, "color": "#9AC1F4"})
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
                        )
                    ]
                ),
                dbc.Col(width = 10,
                    children = [
                        dbc.Row(
                            dcc.Loading(dcc.Graph(id = "ts-clustering-plot"))
                        )
                    ]
                )
            ]
        )
    ]
)

custom_variables = html.Div(children = 
    [
        html.Div(style = {'display': 'flex', 'align-items': 'center', 'padding': '20px'},
            children = [
                html.Span("I would like to create a custom variable called ", style = {'margin-right': '10px'}, className = "text-info"),
                dcc.Input(id = "custom-vars-var-name", style = {"margin-right": "10px", 'width': '200px'}),
                html.Span("by", className = "text-info")
            ]
        ),
        html.Div(id = "custom-vars-fill-area", style = {'display': 'flex', 'align-items': 'center', "margin-left": "100px"},
            children = [
                dcc.Dropdown(
                    id = "custom-vars-operation", 
                    options = [{"label": "Dividing", "value": "division"}],
                    placeholder = "Operation",
                    style = {"width": "200px", "margin-right": "10px"}
                ),
                dcc.Dropdown(id = "custom-vars-output-1-dropdown-div", 
                    options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                    placeholder = "Output 1", 
                    style = {"display": "none"}),
                dcc.Dropdown(id = "custom-vars-output-2-dropdown-div", 
                    options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                    placeholder = "Output 1", 
                    style = {"display": "none"})
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
                        ),
                    ],
                    id = "custom-variable-created-modal",
                    is_open = False,
                )
            ]
        )
    ]
)

# custom_variables = html.Div(style = {"padding": 20},
#     children = [
#         dbc.Row(className = "text-primary", children = [
#                 dbc.Col(width = 4, children = html.Div("I would like to create a variable called")),
#                 dbc.Col(width = 3, children = dcc.Input(id = "custom-vars-var-name")),
#                 dbc.Col(width = 1, children = html.Div("by"))
#             ]
#         ),
#         dbc.Row(align = "center", justify = "between", style = {"padding": 20},
#             children = [
#                 dbc.Col(width = {"size": 1, "offset": 1}, 
#                         children = [
#                             dcc.Dropdown()
#                         ]),
#                 dbc.Col(width = 4, 
#                     children = 
#                         dcc.Dropdown(id = "custom-vars-output-1-dropdown-div", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
#                                     placeholder = "Output 1")
#                 ),
#                 dbc.Col(width = 1, children = html.Div("by"), className = "text-info"),
#                 dbc.Col(width = 4, 
#                     children = 
#                         dcc.Dropdown(id = "custom-vars-output-2-dropdown-div", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
#                                     placeholder = "Output 2")
#                 )
#             ]
#         ),
#         dbc.Button(id = "create-custom-variable-button", children = "Create", className = "btn btn-primary btn-lg"),
#         dbc.Modal(
#             [
#                 dbc.ModalHeader(dbc.ModalTitle("Success!"), close_button = True),
#                 dbc.ModalFooter(
#                     dbc.Button(
#                         "Close",
#                         id = "close-centered",
#                         className = "ms-auto",
#                         n_clicks = 0,
#                     )
#                 ),
#             ],
#             id = "custom-variable-created-modal",
#             is_open = False,
#         ),

#     ]
# )

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

# footer = dbc.Navbar(style = {"border-radius": "30px 30px 0px 0px"},
#     class_name = "navbar navbar-expand-lg custom-navbar",
#     color = "#785EF0",
#     children = [
#         dbc.Row(
#             style = {"padding": 20, "margin-top": 15},
#             children = [
#                 html.P(style = {"color": "white"},
#                         children = 
#                         ["Created by Jennifer Morris and Kenny Cox | Github for this dashboard: "]
#                 )
#             ],
#             align = "center"
#         )
#     ]
# )

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
                    dbc.Tab(id = "output-timeseries", label = "Output Distributions", children = [output_timeseries]),
                    dbc.Tab(id = "input-dist", label = "Input Distributions", children = [input_dists]),
                    dbc.Tab(id = "input-output-mapping", label = "Input-Output Mapping", children = [input_output_mapping]),
                    dbc.Tab(id = "choropleth-map", label = "Choropleth Mapping", children = [choropleth_map]),
                    dbc.Tab(id = "ts-clustering-tab", label = "Time Series Clustering", children = [time_series_clustering]),
                    dbc.Tab(id = "custom-variables-tab", label = "Custom Variables", children = [custom_variables])
                ]
                )
            ]
            ),
        dcc.Store(id = "stored-custom-variables", storage_type = "session")
    ]
)

# adding slider for histogram when user selects histogram option
@app.callback(
    Output('slider-area', 'style'),
    Input('chart-options', 'value'))
def add_hist_slider(chart_type):
    if chart_type == 'dist-by-year':
        return {}
    else:
        return {"display": "none"}

# callback for output time series
@app.callback(
    Output('output-time-series-plot', 'figure'),
    [Input('output-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('scenario-dropdown', 'value'),
     Input('chart-options', 'value'),
     Input('year-slider', 'value'),
     Input('output-color-scheme', 'value')],
    [State('output-time-series-plot', 'figure')]
)
def update_timeseries_graph(output_name, selected_regions, selected_scenarios, chart_type, year, color_scheme, existing_figure):
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]
    if chart_type == "time-series":
        if not selected_regions or not selected_scenarios:
            raise PreventUpdate

        if not existing_figure or len(existing_figure.get('data')) == 0:
            region = selected_regions[0]
            scenario = selected_scenarios[0]
            new_trace_df = DataRetrieval(db, output_name, region, scenario).single_output_df_to_graph(5, 95)
            traces_to_add = NewTimeSeries(output_name, region, scenario, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()

            fig = go.Figure(traces_to_add)
            fig.update_layout(
                height = 625,
                margin = dict(t = 40, b = 0, l = 10),
                title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name]),
                yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
                xaxis = dict(title = dict(text = "Year", font = dict(size = 16)))
            )
            return fig

        current_trace_info = TraceInfo(existing_figure)
        if current_trace_info.type[0] == "histogram": # means active figure is histogram, so need to generate scatter 
            for region, scenario in product(selected_regions, selected_scenarios):
                new_trace_df = DataRetrieval(db, output_name, region, scenario).single_output_df_to_graph(5, 95)
                traces_to_add = NewTimeSeries(output_name, region, scenario, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()

            try:
                title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name])
            except KeyError:
                title_text = "Time Series for {}".format(output_name)

            fig = go.Figure(traces_to_add)
            fig.update_layout(
                height = 625,
                margin = dict(t = 40, b = 0, l = 10),
                title_text = title_text,
                # yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
                xaxis = dict(title = dict(text = "Year", font = dict(size = 16)))
            )
            return fig
        else:
            combos_with_trace_name = list(product(selected_regions, selected_scenarios, ["lower", "median", "upper"]))
            current_traces = current_trace_info.traces
            custom_data_just_strings = [i[0] for i in current_trace_info.custom_data]
            existing_selections = set(custom_data_just_strings)
            all_selections = set(["{}|{}|{}|{}".format(output_name, reg, sce, trace_name) for reg, sce, trace_name in combos_with_trace_name])

            # changes to make
            if trigger_id == "output-color-scheme":
                # without this logic, the color of the figure will not update when the color scheme is changed
                # what this does is take all existing plots and changes color according to what the new color scheme dictates
                existing_figure_data = existing_figure["data"]
                for i in existing_figure_data:
                    trace_name = i["customdata"][0]
                    region = trace_name.split(' ')[-3]
                    scenario = trace_name.split(' ')[-2]
                    if color_scheme == "by-region":
                        color = Color().get_color_for_timeseries(color_scheme, region)
                    elif color_scheme == "by-scenario":
                        color = Color().get_color_for_timeseries(color_scheme, scenario)
                    elif color_scheme == "standard":
                        color = Color().get_color_for_timeseries(color_scheme, [region, scenario])

                    i["line"]["color"] = color

            no_change = existing_selections.intersection(all_selections)
            to_delete = existing_selections.difference(all_selections)
            to_add = all_selections.difference(existing_selections)

            # removing traces - well, keeping ones that haven't been removed
            indices_to_delete = [custom_data_just_strings.index(i) for i in to_delete]
            indices_to_keep = [i for i in range(len(current_traces)) if i not in indices_to_delete]
            current_traces = [current_traces[i] for i in indices_to_keep]

            # adding traces
            new_traces = []
            decomposed_traces_to_add = set([i.split("|")[0] + "|" + i.split("|")[1] + "|" + i.split("|")[2] for i in to_add])
            for i in decomposed_traces_to_add:
                output, reg, sce = tuple(i.split("|"))
                new_trace_df = DataRetrieval(db, output_name, reg, sce).single_output_df_to_graph(5, 95)
                traces_to_add = NewTimeSeries(output_name, reg, sce, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()
                new_traces += traces_to_add

            if output not in options_obj.outputs:
                title_text = "Time Series for " + output_name.split("-")[-1]
            else:
                title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name])

            fig = go.Figure(data = current_traces + new_traces)
            fig.update_layout(
                height = 625,
                margin = dict(t = 40, b = 0, l = 10),
                title_text = title_text,
                # yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
                xaxis = dict(title = dict(text = "Year", font = dict(size = 16)))
            )
            return fig

    else:
        if not selected_regions or not selected_scenarios:
            raise PreventUpdate
        
        styling_options = {"color": color_scheme}
        fig = OutputHistograms(output_name, selected_regions, selected_scenarios, year, db, styling_options = styling_options).make_plot()
        fig.update_layout(
            height = 550,
            margin = dict(t = 50, b = 20, l = 10)
        )
        return fig

# callback for inputs
@app.callback(
    Output("input-dist-graph", "figure"),
    Input("input-dist-options", "value"),
    Input("expanded-view-input-dist-options", "value"))
def update_input_dist(inputs, focus_input):
    if not inputs:
        raise PreventUpdate
    figure = InputDistribution(inputs).make_plot(focus_input)

    return figure

# callback for i/o mapping
@app.callback(
    Output("input-output-mapping-figure", "figure"),
    Input("input-output-mapping-output", "value"),
    Input("input-output-mapping-regions", "value"),
    Input("input-output-mapping-scenario", "value"),
    Input("input-output-mapping-year", "value")
)
def update_figure(output, region, scenario, year):
    if not region or not output or not year:
        raise PreventUpdate
    
    df = DataRetrieval(db, output, region, scenario, year).input_output_mapping_df()
    unstyled_figure = InputOutputMappingPlot(output, region, scenario, year, df)
    finished_figure = FinishedFigure(unstyled_figure).make_finished_figure()

    return finished_figure

# callback for choropleth mapping
@app.callback(
    Output("choropleth-mapping-figure", "figure"),
    Input("choropleth-mapping-output", "value"),
    Input("choropleth-mapping-scenario", "value"),
    Input("choropleth-mapping-year", "value")
)
def update_figure(output, scenario, year):
    if not scenario or not output or not year:
        raise PreventUpdate
    
    df = DataRetrieval(db, output, "GLB", scenario, year).choropleth_map_df(5, 95)
    unstyled_fig = ChoroplethMap(df, output, scenario, year, 5, 95)
    finished_fig = FinishedFigure(unstyled_fig).make_finished_figure()

    return finished_fig

# callback for ts clustering
@app.callback(
    Output("ts-clustering-plot", "figure"),
    Input("ts-clustering-output", "value"),
    Input("ts-clustering-region", "value"),
    Input("ts-clustering-scenario", "value"),
    Input("ts-clustering-n-clusters", "value"),
    Input("ts-clustering-metric", "value")
)
def update_figure(output, region, scenario, n_clusters, metric):
    if not region or not output or not scenario:
        raise PreventUpdate
    
    df = DataRetrieval(db, output, region, scenario).single_output_df()
    fig = TimeSeriesClusteringPlot(df, output, region, scenario, n_clusters = n_clusters, metric = metric)
    finished_figure = FinishedFigure(fig).make_finished_figure()

    return finished_figure

# callback for dynamic display of custom variables tab
@app.callback(
    Output("custom-vars-fill-area", "children"),
    Output("custom-vars-var-name", "value"),
    Output("custom-vars-operation", "value"),
    Input("custom-vars-operation", "value"),
    Input("create-custom-variable-button", "n_clicks"),
    State("custom-vars-fill-area", "children"),
    State("custom-vars-var-name", "value"),
    State("custom-vars-operation", "value")
)
def dynamic_custom_variables_fill(operation_type, n_clicks, current_fill_area, text_box_value, current_operation):
    if not operation_type:
        raise PreventUpdate
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]
    
    if trigger_id == "custom-vars-operation":
        if operation_type == "division":
            if len(current_fill_area) == 4:
                return current_fill_area, text_box_value, current_operation
            else:
                return ([current_fill_area[0]] + [
                    dcc.Dropdown(id = "custom-vars-output-1-dropdown-div", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                        placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                        html.Span("by", style = {'margin-right': '10px'}),
                        dcc.Dropdown(id = "custom-vars-output-2-dropdown-div", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                                    placeholder = "Output 2",
                                    style = {"width": "500px"})
                            ], text_box_value, current_operation)
    else:
        return ([current_fill_area[0]], "", None)

# callback for custom variables
@app.callback(
    Output("stored-custom-variables", "data"),
    State("stored-custom-variables", "data"),
    State("custom-vars-var-name", "value"),
    State("custom-vars-output-1-dropdown-div", "value"),
    State("custom-vars-output-2-dropdown-div", "value"),
    Input("create-custom-variable-button", "n_clicks")
)
def update_custom_variables(current_data, var_name, output_1, output_2, n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    current_data = current_data or {}
    current_data[var_name] = {"operation": "division", "output1": output_1, "output2": output_2}

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

output_dropdowns_to_update_ids = ["output-dropdown", "input-output-mapping-output", "choropleth-mapping-output", "ts-clustering-output"]
for dropdown in output_dropdowns_to_update_ids:
    @app.callback(
        Output(dropdown, "options"),
        State(dropdown, "options"),
        Input("stored-custom-variables", "data"),
        Input("create-custom-variable-button", "n_clicks")
    )
    def update_output_dropdowns(current_options, current_stored_data, n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        for custom_var_name in current_stored_data.keys():
            all_current_outputs = [x["label"] for x in current_options]
            if custom_var_name not in all_current_outputs:
                value_to_use = current_stored_data[custom_var_name]["output1"] + "-" + current_stored_data[custom_var_name]["operation"] + "-" + current_stored_data[custom_var_name]["output2"] + "-" + custom_var_name
                dict_to_append = {"label": custom_var_name, "value": value_to_use}
        current_options.append(dict_to_append)

        return current_options

if __name__ == '__main__':
    app.run(debug = True, host = "localhost")

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