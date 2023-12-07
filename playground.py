import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from sql_utils import SQLConnection
from figure import InputOutputMappingPlot

# Initialize the Dash app
app = dash.Dash(__name__)

df = SQLConnection("jp_data").input_output_mapping_df("elec_prod_Renewables_TWh", "USA", "2C", 2050)
fig = InputOutputMappingPlot("total_emissions_CO2_million_ton_CO2", df).make_plot()

# Layout of the Dash app
app.layout = html.Div([
    dcc.Graph(
        id='parallel-coordinates-plot',
        figure=fig
    ),
    html.Button('Add Annotations', id='annotation-button', n_clicks=0),
])

# Callback to add annotations
@app.callback(
    Output('parallel-coordinates-plot', 'figure'),
    [Input('annotation-button', 'n_clicks')],
    [State('parallel-coordinates-plot', 'restyleData')]
)
def update_annotations(n_clicks, data):
    if n_clicks > 0:
        selected_runs = data
        print(selected_runs)
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
