"""Dash layout fragments for the MIT EPPA visualization dashboard."""
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from styling import Options, Readability
from eppa_viz.webapp.connections import options_obj, readability_obj

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
                        href = "https://saber-wrench-cdd.notion.site/MIT-CS3-Data-Visualization-Dashboard-Guide-2de867c045ed80909bb1ff1ed60ea2e1",
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



def build_main_layout(navbar):
    return html.Div(
        [
            navbar,
            html.Br(),
            html.P(
                "Select a tab to display data or run scenario discovery algorithms. "
                "All figures are preserved when you switch between tabs.",
                className="text-primary",
                style={"padding": 20},
            ),
            html.Div(
                [
                    dbc.Tabs(
                        id="tabs",
                        children=[
                            dbc.Tab(id="overview", label="Overview", children=[overview]),
                            dbc.Tab(id="output-timeseries", label="Output Distributions", children=[output_timeseries]),
                            dbc.Tab(id="input-dist", label="Input Distributions", children=[input_dists]),
                            dbc.Tab(id="input-output-mapping", label="Input-Output Mapping", children=[input_output_mapping]),
                            dbc.Tab(id="output-output-mapping", label="Output-Output Mapping", children=[output_output_mapping]),
                            dbc.Tab(id="choropleth-map", label="Choropleth Mapping", children=[choropleth_map]),
                            dbc.Tab(id="ts-clustering-tab", label="Time Series Clustering", children=[time_series_clustering]),
                            dbc.Tab(id="regional-heatmaps-tab", label="Regional Heatmaps", children=[regional_heatmaps]),
                            dbc.Tab(id="custom-variables-tab", label="Custom Variables", children=[custom_variables]),
                        ],
                    )
                ]
            ),
            dcc.Store(id="stored-custom-variables", storage_type="memory"),
        ]
    )
