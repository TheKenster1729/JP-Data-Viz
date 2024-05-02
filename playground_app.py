import dash
from dash import html, dcc, Input, Output, State, callback_context
import uuid  # For generating unique IDs

app = dash.Dash(__name__)

# Layout with a button to add a new dropdown and a placeholder for dynamically added dropdowns
app.layout = html.Div([
    html.Button("Add Dropdown", id="add-button", n_clicks=0),
    html.Div(id="dropdown-container", children=[]),
    html.Button("Update Options", id="update-options", n_clicks=0),
    html.Div(id="output-container")
])

@app.callback(
    Output("dropdown-container", "children"),
    Input("add-button", "n_clicks"),
    State("dropdown-container", "children"),
)
def add_dropdown(n_clicks, children):
    if n_clicks > 0:
        new_dropdown_id = str(uuid.uuid4())
        new_dropdown = dcc.Dropdown(
            id={'type': 'dynamic-dropdown', 'index': new_dropdown_id},
            options=[{'label': 'Initial Option', 'value': 'initial'}],
            value='initial'
        )
        children.append(html.Div([new_dropdown]))
    return children

@app.callback(
    Output({'type': 'dynamic-dropdown', 'index': dash.ALL}, 'options'),
    Input("update-options", "n_clicks"),
    prevent_initial_call=True
)
def update_dropdown_options(n_clicks):
    new_options = [
        {'label': 'Option 1', 'value': 'option1'},
        {'label': 'Option 2', 'value': 'option2'}
    ]
    return [new_options] * n_clicks  # Return a list of new options for each dropdown

if __name__ == '__main__':
    app.run_server(debug=True)
