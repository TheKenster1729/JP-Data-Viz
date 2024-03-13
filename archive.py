# old output distributions

# # graphs and callback for output time series visualization
# @app.callback(Output("tab-1-content", "children"),
#               Input("add-output-button", "n_clicks"),
#               State("tab-1-content", "children"))
# def add_new_graph(n_clicks, children):
#     if not n_clicks:
#         raise PreventUpdate
#     new_element = html.Div(style = {"padding": 20, "display": "flex", "flex-direction": "row"},
#             children = [
#                 html.Div(
#                     children = [
#                         html.Div("Region", className = "text-primary"),
#                         dbc.Checklist(
#                             id = {"type": "checklist-region", "index": n_clicks},
#                             options = [{'label': i, 'value': i} for i in Options().region_names],
#                             value = ["GLB"]
#                         )
#                     ]
#                 ),
#             html.Div(style = {"flex": 1, "margin-left": 20},
#                 children = [
#                     html.Div("Output", className = "text-primary"),
#                     dcc.Dropdown(id = {"type": "output-dropdown", "index": n_clicks},
#                     options = [{'label': Readability().readability_dict_forward[i], 'value': i} for i in Options().outputs],
#                     value = "total_emissions_CO2_million_ton_CO2"
#                 ),
#                     dcc.Graph(id = {"type": "chart", "index": n_clicks}),
#                     dbc.Row(
#                         children = 
#                         [
#                             dbc.Col(
#                                 children = [
#                                     html.Div("Scenario", className = "text-primary"),
#                                     dbc.Checklist(
#                                         id = {"type": "checklist-scenario", "index": n_clicks},
#                                         options = [{'label': i, 'value': i} for i in Options().scenarios],
#                                         value = ["Ref"],
#                                         inline = True
#                                     ),
#                                     html.Div("Show Uncertainty", className = "text-primary"),
#                                     dbc.Checklist(
#                                         id = {"type": "toggle-uncertainty", "index": n_clicks},
#                                         options = [{"label": "Uncertainty On", "value": True}],
#                                         value = [True],
#                                         inline = True
#                                     )
#                                 ]
#                             ),
#                             dbc.Col(
#                                 children = [
#                                     html.Div(
#                                         children = [
#                                             html.Div("Year", className = "text-primary"),
#                                             dcc.Slider(
#                                                 min = min(Options().years),
#                                                 max = max(Options().years),
#                                                 step = int((max(Options().years) - min(Options().years))/(len(Options().years) - 1)),
#                                                 id = {"type": "output-dist-hist-year", "index": n_clicks},
#                                                 marks = {year: str(year) for year in Options().years[::2]},
#                                                 value = 2050)
#                                         ]
#                                     )
#                                 ]
#                             )
#                         ]
#                         ),
#                         dbc.Accordion(start_collapsed = True,
#                                       children = [
#                                           dbc.AccordionItem(title = "Advanced Options",
#                                                             children = [
#                                                                 html.P("Set Uncertainty Range - Upper and Lower Percentiles", className = "text-primary"),
#                                                                 html.P("Upper Bound"),
#                                                                 dcc.Slider(51, 99, 1, id = {"type": "uncertainty-range-slider-upper", "index": n_clicks}, value = 95,
#                                                                            marks = {label: str(label) for label in range(50, 100, 5)}, tooltip = dict(always_visible = True)),
#                                                                 html.P("Lower Bound"),
#                                                                 dcc.Slider(1, 49, 1, id = {"type": "uncertainty-range-slider-lower", "index": n_clicks}, value = 5, 
#                                                                            marks = {label: str(label) for label in range(0, 50, 5)}, tooltip = dict(always_visible = True)),
#                                                                 dbc.Button("Set Bounds", id = {"type": "regenerate-plot-button", "index": n_clicks}, class_name = "Primary")
#                                                             ])
#                                       ])
#                     ]
#                 )
#             ]
#         )
#     children.append(new_element)
#     return children

# #graphs and callback for tab 3
# '''
# @app.callback(Output("tab-3-content", "children"),
#               Input("add-output-dist-button", "n_clicks"),
#               State("tab-3-content", "children"))
# def add_output_distribution(n_clicks, children):
#     if not n_clicks:
#         raise PreventUpdate
#     new_element = html.Div(style = {"padding": 20, "display": "flex", "flex-direction": "row"},
#             children = [
#                 html.Div(
#                     children = [
#                         dbc.Checklist(
#                             id = {"type": "checklist-region-output-dist", "index": n_clicks},
#                             options = [{'label': i, 'value': i} for i in Options().region_names],
#                             value = ["GLB"]
#                         )
#                     ]
#                 ),
#             html.Div(style = {"padding": 20, "flex": 1},
#                 children = [
#                     dcc.Dropdown(id = {"type": "output-dropdown-output-dist", "index": n_clicks},
#                     options = [{'label': Readability().readability_dict_forward[i], 'value': i} for i in Options().outputs],
#                     value = "total_emissions_CO2_million_ton_CO2"
#                 ),
#                     dcc.Graph(id = {"type": "output-dist-graph", "index": n_clicks}),
#                     dbc.Checklist(
#                         id = {"type": "checklist-scenario-output-dist", "index": n_clicks},
#                         options = [{'label': i, 'value': i} for i in Options().scenarios],
#                         value = ["Ref"],
#                         inline = True
#                 )
#             ])
#         ])
#     children.append(new_element)
#     return children

# @app.callback(
#     Output({"type": "output-dist-graph", "index": MATCH}, "figure"),
#     [Input({"type": "output-dropdown-output-dist", "index": MATCH}, "value"),
#     Input({"type": "checklist-region-output-dist", "index": MATCH}, "value"),
#     Input({"type": "checklist-scenario-output-dist", "index": MATCH}, "value")])
# def update_graph(output, regions, scenarios):
#     if not output:
#         raise PreventUpdate
#     figure = OutputDistribution(output).create_output_distribution(regions, scenarios)
#     return figure

# old input viz
# ###############################################

# # graphs and callback for input viz
# @app.callback(Output("tab-2-content", "children"),
#               Input("add-input-dist-button", "n_clicks"),
#               State("tab-2-content", "children"))
# def add_input_distribution(n_clicks, children):
#     if not n_clicks:
#         raise PreventUpdate
#     new_element = html.Div(
#             children = [
#                 html.Br(),
#                 dbc.Row(style = {"padding": 20},
#                         children = [
#                             dbc.Col(
#                                     dcc.Dropdown(
#                                         id = {"type": "checklist-input", "index": n_clicks},
#                                         options = [{'label': i, 'value': i} for i in Options().input_names],
#                                         value = ["wind"],
#                                         multi = True
#                                     ),
#                                 ),
#                                 dbc.Col(
#                                     children = [
#                                         dcc.Dropdown(style = {"width": 100},
#                                             id = {"type": "checklist-input-histogram", "index": n_clicks},
#                                             options = [{'label': i, 'value': i} for i in Options().input_names],
#                                             value = "wind"
#                                         )
#                                     ]
#                                 )
#                                 ]
#                             ),
#                 html.Div(
#                         children = [
#                             dcc.Graph(id = {"type": "input-dist-graph", "index": n_clicks})
#                         ]
#                 )
#             ]
#         )
#     children.append(new_element)
#     return children

# addition block for custom variables
# dbc.Row(align = "center", justify = "between", style = {"padding": 20},
#     children = [
#         dbc.Col(width = {"size": 1, "offset": 1}, children = html.Div("Adding"), className = "text-info"),
#         dbc.Col(width = 4, 
#             children = 
#                 dcc.Dropdown(id = "custom-vars-output-1-dropdown-add", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
#                             placeholder = "Output 1")
#         ),
#         dbc.Col(width = 1, children = html.Div("and"), className = "text-info"),
#         dbc.Col(width = 4, 
#             children = 
#                 dcc.Dropdown(id = "custom-vars-output-2-dropdown-add", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
#                             placeholder = "Output 2")
#         )
#     ]
# ),