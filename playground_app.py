import dash
from dash import html, dcc, Input, State, Output, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from styling import Readability

readability_obj = Readability()
# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions = True)

app.layout = custom_variables = html.Div(children = 
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
                )
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
        ),
        dcc.Store(id = "stored-custom-variables", storage_type = "session")
    ]
)

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
            print(current_fill_area)
            if len(current_fill_area) == 4:
                return current_fill_area, text_box_value, current_operation
            else:
                return (current_fill_area + [
                    dcc.Dropdown(id = "custom-vars-output-1-dropdown-div", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                        placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                        html.Span("by", style = {'margin-right': '10px'}),
                        dcc.Dropdown(id = "custom-vars-output-2-dropdown-div", options = [{"label": k, "value": v} for k, v in readability_obj.naming_dict_display_names_first.items()],
                                    placeholder = "Output 2",
                                    style = {"width": "500px"})
                            ], text_box_value, current_operation)
    else:
        return ([current_fill_area[0]], "", None)

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

if __name__ == '__main__':
    app.run_server(debug=True, host = "localhost")
