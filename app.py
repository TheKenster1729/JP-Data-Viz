# use venv when running this code
import dash
from dash.exceptions import PreventUpdate
from dash import html, dcc
import dash_bootstrap_components as dbc
from sql_utils import SQLConnection
from styling import Options, Readability
from dash.dependencies import Input, Output, State, MATCH
from figure import TimeSeries, InputDistribution, OutputDistribution, InputOutputMappingPlot
import numpy as np

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.PULSE])

# initialize SQL database
db = SQLConnection("all_data_jan_2024")

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

input_dists = html.Div(id = "tab-2-content",
    children = [
        html.Br(),
        html.Div("Click the button below to add a new input visualization plot."),
        dbc.Button('Add New Input Distribution', id = 'add-input-dist-button', n_clicks = 0, color = "primary")
    ])

input_output_mapping = html.Div(id = "tab-4-content",
    children = [
        html.Div(style = {"padding": 20, "display": "flex", "flex-direction": "row", "gap": 10},
            children = [
                html.Div(style = {"display": "flex", "flex-direction": "column", "gap": 5, "justify-content": "center", "width": 100},
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
                html.Div(style = {"display": "flex", "flex-direction": "column", "flex": 1, "gap": 5},
                    children = [
                                html.Div("Output", className = "text-primary"),
                                dcc.Dropdown(id = "input-output-mapping-output",
                                options = [{'label': Readability().readability_dict_forward[i], 'value': i} for i in Options().outputs],
                                value = "total_emissions_CO2_million_ton_CO2"
                                ),
                                dcc.Loading([dcc.Graph(id = "input-output-mapping-figure")]),
                ]),
        ])
    ])

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
                dbc.Tab(id = "examples-gallery", label = "Examples Gallery", children = [examples]),
                dbc.Tab(id = "output-timeseries", label = "Output Distributions", children = [output_timeseries]),
                dbc.Tab(id = "input-dist", label = "Input Distributions", children = [input_dists]),
                dbc.Tab(id = "input-output-mapping", label = "Input-Output Mapping", children = [input_output_mapping])
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
                                                            ])
                                      ])
                    ]
                )
            ]
        )
    children.append(new_element)
    return children

@app.callback(
    Output({"type": "chart", "index": MATCH}, "figure"),
    Input({"type": "output-dropdown", "index": MATCH}, "value"),
    Input({"type": "checklist-region", "index": MATCH}, "value"),
    Input({"type": "checklist-scenario", "index": MATCH}, "value"),
    Input({"type": "toggle-uncertainty", "index": MATCH}, "value"),
    Input({"type": "output-dist-hist-year", "index": MATCH}, "value"),
    Input({"type": "regenerate-plot-button", "index": MATCH}, "n_clicks"),
    State({"type": "uncertainty-range-slider-upper", "index": MATCH}, "value"),
    State({"type": "uncertainty-range-slider-lower", "index": MATCH}, "value")
    )
def update_graph(output, regions, scenarios, show_uncertainty, year, num_uncertainty_clicks, upper_bound, lower_bound):
    if not output or not regions or not scenarios:
        raise PreventUpdate
    df = db.output_df(output, regions, scenarios)
    figure = TimeSeries(output, year, df).make_plot(show_uncertainty = show_uncertainty, upper = upper_bound, lower = lower_bound)
    
    return figure
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

@app.callback(
    Output({"type": "input-dist-graph", "index": MATCH}, "figure"),
    Input({"type": "checklist-input", "index": MATCH}, "value"),
    Input({"type": "checklist-input-histogram", "index": MATCH}, "value"))
def update_input_dist(inputs, focus_input):
    if not inputs:
        raise PreventUpdate
    figure = InputDistribution(inputs).make_plot(focus_input)

    return figure
###############################################

#graphs and callback for tab 3
'''
@app.callback(Output("tab-3-content", "children"),
              Input("add-output-dist-button", "n_clicks"),
              State("tab-3-content", "children"))
def add_output_distribution(n_clicks, children):
    if not n_clicks:
        raise PreventUpdate
    new_element = html.Div(style = {"padding": 20, "display": "flex", "flex-direction": "row"},
            children = [
                html.Div(
                    children = [
                        dbc.Checklist(
                            id = {"type": "checklist-region-output-dist", "index": n_clicks},
                            options = [{'label': i, 'value': i} for i in Options().region_names],
                            value = ["GLB"]
                        )
                    ]
                ),
            html.Div(style = {"padding": 20, "flex": 1},
                children = [
                    dcc.Dropdown(id = {"type": "output-dropdown-output-dist", "index": n_clicks},
                    options = [{'label': Readability().readability_dict_forward[i], 'value': i} for i in Options().outputs],
                    value = "total_emissions_CO2_million_ton_CO2"
                ),
                    dcc.Graph(id = {"type": "output-dist-graph", "index": n_clicks}),
                    dbc.Checklist(
                        id = {"type": "checklist-scenario-output-dist", "index": n_clicks},
                        options = [{'label': i, 'value': i} for i in Options().scenarios],
                        value = ["Ref"],
                        inline = True
                )
            ])
        ])
    children.append(new_element)
    return children

@app.callback(
    Output({"type": "output-dist-graph", "index": MATCH}, "figure"),
    [Input({"type": "output-dropdown-output-dist", "index": MATCH}, "value"),
    Input({"type": "checklist-region-output-dist", "index": MATCH}, "value"),
    Input({"type": "checklist-scenario-output-dist", "index": MATCH}, "value")])
def update_graph(output, regions, scenarios):
    if not output:
        raise PreventUpdate
    figure = OutputDistribution(output).create_output_distribution(regions, scenarios)
    return figure
###############################################
'''

# graphs and callbacks for i/o mapping
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
    
    df = db.input_output_mapping_df(output, region, scenario, year)
    fig = InputOutputMappingPlot(output, df).make_plot()

    return fig
###############################################

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
