# use venv when running this code
import dash
from dash.exceptions import PreventUpdate
from dash import html, dcc, ctx
import dash_bootstrap_components as dbc
from sql_utils import SQLConnection, DataRetrieval
from styling import Options, Readability
from dash.dependencies import Input, Output, State, MATCH
from figure import NewTimeSeries, InputDistribution, OutputDistribution, InputOutputMappingPlot, TraceInfo, OutputHistograms, ChoroplethMap
import numpy as np
import dash_mantine_components as dmc
import plotly.io as pio

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.PULSE, dbc.icons.BOOTSTRAP])

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
        html.Br(),
        html.Div("Click the button below to display time series data for a new output.", className = "primary"),
        dbc.Button('Add New Figure', id = 'add-output-button', n_clicks = 0, color = "primary")
    ])

input_dists = html.Div(id = "tab-2-content", style = {"padding": 20},
    children = [
        html.Br(),
        html.Div("Click the button below to add a new input visualization plot.", className = "primary"),
        dbc.Button('Add New Input Distribution', id = 'add-input-dist-button', n_clicks = 0, color = "primary")
    ])

input_output_mapping = html.Div(id = "tab-4-content",
    children = [
        html.Div(style = {"padding": 20, "display": "flex", "flex-direction": "row", "gap": 10},
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
        dbc.Row(
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

# input_dists = html.Div(id = "tab-2-content",
#     children = [
#         html.Br(),
#         html.Div("Click the button below to add a new input visualization plot."),
#         dbc.Button('Add New Input Distribution', id = 'add-input-dist-button', n_clicks = 0, color = "primary")
#     ])

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
                        dbc.Col(
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
                                    options = [{'label': i, 'value': i} for i in Options().scenarios],
                                    value = "Ref"
                                ),
                                html.Div("Year", className = "text-primary"),
                                dcc.Dropdown(
                                    id = "input-output-mapping-year",
                                    options = [{'label': i, 'value': i} for i in Options().years],
                                    value = 2050)
                                ]
                            ),
                        dbc.Col(
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
                                dcc.Loading([dcc.Graph(id = "choropleth-mapping-figure")]),
                                ]
                            )
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

connect = html.Div(style = {},
    children = [
        html.Br(),
        html.Div(style = {},
                 children = [
                     dcc.Graph(id = "scenario-connection-graph")
                ]
            ),
        dbc.Row(children = [
            dbc.Col(children = [
                html.P("Add Input", className = "text-primary"),
                dcc.Dropdown(id = "scenario-connection-inputs", options = [{'label': i, 'value': i} for i in Options().input_names],
                             value = ["wind"]
                )
            ]),
            dbc.Col(children = [
                html.P("Add Output", className = "text-primary"),
                dbc.Row(children = [
                    dbc.Col(children = [
                        html.P("Name"),
                        dcc.Dropdown(id = "scenario-connection-output", options = [{'label': Readability().readability_dict_forward[i], 'value': i} for i in Options().outputs])
                    ]),
                    dbc.Col(children = [
                        html.P("Region"),
                        dcc.Dropdown(id = "scenario-connection-region", options = [{"label": i, "value": i} for i in Options().region_names])
                    ])
                ])
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
                dbc.Tab(id = "connection", label = "Connect", children = connect)
            ]
            )
        ]
        )
    ]
)

# graphs and callback for output time series visualization
@app.callback(Output("tab-1-content", "children"),
              Input("add-output-button", "n_clicks"),
              State("tab-1-content", "children"))
def add_new_graph(n_clicks, children):
    if not n_clicks:
        raise PreventUpdate
    new_element = html.Div(style = {"padding": 20, "display": "flex", "flex-direction": "row"},
            children = [
                html.Div(
                    children = [
                        html.Div("Region", className = "text-primary"),
                        dbc.Checklist(
                            id = {"type": "checklist-region", "index": n_clicks},
                            options = [{'label': i, 'value': i} for i in Options().region_names],
                            value = ["GLB"]
                        )
                    ]
                ),
            html.Div(style = {"flex": 1, "margin-left": 20},
                children = [
                    html.Div("Output", className = "text-primary"),
                    dcc.Dropdown(id = {"type": "output-dropdown", "index": n_clicks},
                    options = [{'label': Readability().readability_dict_forward[i], 'value': i} for i in Options().outputs],
                    value = "total_emissions_CO2_million_ton_CO2"
                ),
                    dcc.Graph(id = {"type": "chart", "index": n_clicks}),
                    dbc.Row(
                        children = 
                        [
                            dbc.Col(
                                children = [
                                    html.Div("Scenario", className = "text-primary"),
                                    dbc.Checklist(
                                        id = {"type": "checklist-scenario", "index": n_clicks},
                                        options = [{'label': i, 'value': i} for i in Options().scenarios],
                                        value = ["Ref"],
                                        inline = True
                                    ),
                                    html.Div("Show Uncertainty", className = "text-primary"),
                                    dbc.Checklist(
                                        id = {"type": "toggle-uncertainty", "index": n_clicks},
                                        options = [{"label": "Uncertainty On", "value": True}],
                                        value = [True],
                                        inline = True
                                    )
                                ]
                            ),
                            dbc.Col(
                                children = [
                                    html.Div(
                                        children = [
                                            html.Div("Year", className = "text-primary"),
                                            dcc.Slider(
                                                min = min(Options().years),
                                                max = max(Options().years),
                                                step = int((max(Options().years) - min(Options().years))/(len(Options().years) - 1)),
                                                id = {"type": "output-dist-hist-year", "index": n_clicks},
                                                marks = {year: str(year) for year in Options().years[::2]},
                                                value = 2050)
                                        ]
                                    )
                                ]
                            )
                        ]
                        ),
                        dbc.Accordion(start_collapsed = True,
                                      children = [
                                          dbc.AccordionItem(title = "Advanced Options",
                                                            children = [
                                                                html.P("Set Uncertainty Range - Upper and Lower Percentiles", className = "text-primary"),
                                                                html.P("Upper Bound"),
                                                                dcc.Slider(51, 99, 1, id = {"type": "uncertainty-range-slider-upper", "index": n_clicks}, value = 95,
                                                                           marks = {label: str(label) for label in range(50, 100, 5)}, tooltip = dict(always_visible = True)),
                                                                html.P("Lower Bound"),
                                                                dcc.Slider(1, 49, 1, id = {"type": "uncertainty-range-slider-lower", "index": n_clicks}, value = 5, 
                                                                           marks = {label: str(label) for label in range(0, 50, 5)}, tooltip = dict(always_visible = True)),
                                                                dbc.Button("Set Bounds", id = {"type": "regenerate-plot-button", "index": n_clicks}, class_name = "Primary")
                                                            ]
                                                        ),
                                            dbc.AccordionItem(title = "Styling Options",
                                                              children = [
                                                                html.P("Adjust Background Color"),
                                                                dmc.ColorPicker(id = {"type": "timeseries-color-picker", "index": n_clicks},
                                                                                format = "hex",
                                                                                value = "#e5ecf5"),
                                                                html.Br(),
                                                                dmc.Switch(id = {"type": "toggle-gridlines", "index": n_clicks}, checked = True),
                                                                dmc.Switch(id = {"type": "smooth-histograms", "index": n_clicks}, checked = False),
                                                                dbc.Button(id = {"type": "apply-styling-button", "index": n_clicks},
                                                                        children = [
                                                                            html.Div("Apply")
                                                                        ])
                                                                ]),
                                            dbc.AccordionItem(title = "Save Options",
                                                              children = [
                                                                dbc.Row(children = 
                                                                [
                                                                dbc.Col(children = [
                                                                    html.P("Save Figure Data as CSV", className = "text-primary"),
                                                                    dbc.Button("Download CSV", id = {"type": "timeseries-data-download-button", "index": n_clicks}, className = "primary"),
                                                                    dcc.Download(id = {"type": "timeseries-data-download", "index": n_clicks})
                                                                ]
                                                                ),
                                                                dbc.Col(children = [
                                                                    html.P("Save Figure as High-Res PNG", className = "text-primary"),
                                                                    dbc.Button("Download PNG", id = {"type": "timeseries-figure-download-button", "index": n_clicks}, className = "primary"),
                                                                    dcc.Download(id = {"type": "timeseries-figure-download", "index": n_clicks})
                                                                ]
                                                                ),
                                                                dbc.Col(children = [
                                                                    html.P("Save Figure as SVG", className = "text-primary"),
                                                                    dbc.Button("Download SVG", id = {"type": "timeseries-svg-download-button", "index": n_clicks}, className = "primary"),
                                                                    dcc.Download(id = {"type": "timeseries-svg-download", "index": n_clicks})
                                                                ]
                                                                )
                                                                ]                                     
                                                                ),
                                                                dbc.Row(children = [
                                                                    html.P("Note: image and SVG exports may take a few moments.")
                                                                ])
                                                              ]
                                                              )
                                      ])
                    ]
                )
            ]
        )
    children.append(new_element)
    return children
@app.callback(
    Output({"type": "chart", "index": MATCH}, "figure"),
    Output({"type": "timeseries-data-download", "index": MATCH}, "data"),
    Output({"type": "timeseries-figure-download", "index": MATCH}, "data"),
    Output({"type": "timeseries-svg-download", "index": MATCH}, "data"),
    Input({"type": "output-dropdown", "index": MATCH}, "value"),
    Input({"type": "checklist-region", "index": MATCH}, "value"),
    Input({"type": "checklist-scenario", "index": MATCH}, "value"),
    Input({"type": "toggle-uncertainty", "index": MATCH}, "value"),
    Input({"type": "output-dist-hist-year", "index": MATCH}, "value"),
    Input({"type": "regenerate-plot-button", "index": MATCH}, "n_clicks"),
    Input({"type": "apply-styling-button", "index": MATCH}, "n_clicks"),
    Input({"type": "timeseries-data-download-button", "index": MATCH}, "n_clicks"),
    Input({"type": "timeseries-figure-download-button", "index": MATCH}, "n_clicks"),
    Input({"type": "timeseries-svg-download-button", "index": MATCH}, "n_clicks"),
    State({"type": "uncertainty-range-slider-upper", "index": MATCH}, "value"),
    State({"type": "uncertainty-range-slider-lower", "index": MATCH}, "value"),
    State({"type": "timeseries-color-picker", "index": MATCH}, "value"),
    State({"type": "toggle-gridlines", "index": MATCH}, "checked"),
    State({"type": "smooth-histograms", "index": MATCH}, "checked")
    )
def update_graph(output, regions, scenarios, show_uncertainty, year, 
                num_uncertainty_clicks, num_styling_clicks, num_download_clicks, num_png_clicks, num_svg_clicks,
                upper_bound, lower_bound, bk_color,
                gridlines, smooth_histograms):
    if not output or not regions or not scenarios:
        raise PreventUpdate
    df = db.output_df(output, regions, scenarios)
    figure = TimeSeries(output, year, df).make_plot({"bk_color": bk_color, "gridlines": gridlines, "smooth_histograms": smooth_histograms}, 
                                                    use_styling = True, show_uncertainty = show_uncertainty, upper = upper_bound, lower = lower_bound)
    
    # check which input has been triggered and only download if the download button was pressed
    try:
        button_clicked = ctx.triggered_id["type"]
    except TypeError:
        button_clicked = None

    if button_clicked == "timeseries-data-download-button":
        return figure, dcc.send_data_frame(df.to_csv, "data.csv"), dash.no_update, dash.no_update
    if button_clicked == "timeseries-figure-download-button":
        img_bytes = pio.to_image(figure, format = "png", scale = 3, width = 2000, height = 1200)
        return figure, dash.no_update, dcc.send_bytes(img_bytes, filename="figure.png"), dash.no_update
    if button_clicked == "timeseries-svg-download-button":
        svg_bytes = pio.to_image(figure, format = "svg")
        return figure, dash.no_update, dash.no_update, dcc.send_bytes(svg_bytes, filename="figure.svg")
    else:
        return figure, dash.no_update, dash.no_update, dash.no_update
    ###############################################

# graphs and callback for input viz
@app.callback(Output("tab-2-content", "children"),
              Input("add-input-dist-button", "n_clicks"),
              State("tab-2-content", "children"))
def add_input_distribution(n_clicks, children):
    if not n_clicks:
        raise PreventUpdate
    new_element = html.Div(
            children = [
                html.Br(),
                dbc.Row(style = {"padding": 20},
                        children = [
                            dbc.Col(
                                    dcc.Dropdown(
                                        id = {"type": "checklist-input", "index": n_clicks},
                                        options = [{'label': i, 'value': i} for i in Options().input_names],
                                        value = ["wind"],
                                        multi = True
                                    ),
                                ),
                                dbc.Col(
                                    children = [
                                        dcc.Dropdown(style = {"width": 100},
                                            id = {"type": "checklist-input-histogram", "index": n_clicks},
                                            options = [{'label': i, 'value': i} for i in Options().input_names],
                                            value = "wind"
                                        )
                                    ]
                                )
                                ]
                            ),
                html.Div(
                        children = [
                            dcc.Graph(id = {"type": "input-dist-graph", "index": n_clicks})
                        ]
                )
            ]
        )
    children.append(new_element)
    return children

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
def update_graph(output_name, selected_regions, selected_scenarios, chart_type, year, color_scheme, existing_figure):
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
                height = 650,
                margin = dict(t = 20, b = 20, l = 10)
            )
            return fig

        current_trace_info = TraceInfo(existing_figure)
        if current_trace_info.type[0] == "histogram": # means active figure is histogram, so need to generate scatter 
            for region, scenario in product(selected_regions, selected_scenarios):
                new_trace_df = DataRetrieval(db, output_name, region, scenario).single_output_df_to_graph(5, 95)
                traces_to_add = NewTimeSeries(output_name, region, scenario, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()

            fig = go.Figure(traces_to_add)
            fig.update_layout(
                height = 650,
                margin = dict(t = 20, b = 20, l = 10)
            )
            return fig
        else:
            combos = list(product(selected_regions, selected_scenarios))
            combos_with_trace_name = list(product(selected_regions, selected_scenarios, ["lower", "median", "upper"]))
            current_traces = current_trace_info.traces
            custom_data_just_strings = [i[0] for i in current_trace_info.custom_data]
            existing_selections = set(custom_data_just_strings)
            all_selections = set(["{} {} {} {}".format(output_name, reg, sce, trace_name) for reg, sce, trace_name in combos_with_trace_name])

            # changes to make
            no_change = existing_selections.intersection(all_selections)
            to_delete = existing_selections.difference(all_selections)
            to_add = all_selections.difference(existing_selections)

            # removing traces - well, keeping ones that haven't been removed
            indices_to_delete = [custom_data_just_strings.index(i) for i in to_delete]
            indices_to_keep = [i for i in range(len(current_traces)) if i not in indices_to_delete]
            current_traces = [current_traces[i] for i in indices_to_keep]

            # adding traces
            new_traces = []
            decomposed_traces_to_add = set([i.split(" ")[0] + " " + i.split(" ")[1] + " " + i.split(" ")[2] for i in to_add])
            for i in decomposed_traces_to_add:
                output, reg, sce = tuple(i.split(" "))
                new_trace_df = DataRetrieval(db, output_name, reg, sce).single_output_df_to_graph(5, 95)
                traces_to_add = NewTimeSeries(output_name, reg, sce, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()
                new_traces += traces_to_add

            fig = go.Figure(data = current_traces + new_traces)
            fig.update_layout(
                height = 650,
                margin = dict(t = 20, b = 20, l = 10)
            )
            return fig

    else:
        if not selected_regions or not selected_scenarios:
            raise PreventUpdate
        
        fig = OutputHistograms(output_name, selected_regions, selected_scenarios, year, db).make_plot()
        fig.update_layout(
            height = 600,
            margin = dict(t = 30, b = 20, l = 10)
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
    fig = InputOutputMappingPlot(output, region, df).make_plot()

    return fig
###############################################

# graphs and callbacks for o/o mapping
###############################################

# graphs and callbacks for 
###############################################

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
