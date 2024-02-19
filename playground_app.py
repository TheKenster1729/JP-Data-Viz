import pandas as pd
import numpy as np
from sql_utils import SQLConnection
from itertools import product
import plotly.graph_objects as go
from styling import Options

class TraceInfo:
    def __init__(self, figure):
        self.fig = figure
        self.traces = self.__traces()
        self.number = self.__len__()
        self.names = self.__trace_names()
        self.colors = self.__trace_colors()

    def __getitem__(self, index):
        return self.traces[index]

    def __len__(self):
        return len(self.traces)

    def __traces(self):
        return self.fig.get('data', [])

    def __trace_colors(self):
        return [trace.get('marker', {}).get('color') for trace in self.traces]

    def __trace_names(self):
        return [trace.get('name') for trace in self.traces]

def number_to_ordinal(n):
    """
    Convert an integer from 1 to 100 into its English ordinal representation.
    
    Args:
    n (int): Integer from 1 to 100
    
    Returns:
    str: The ordinal representation of n
    """
    if 11 <= n <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return str(n) + suffix

def get_data(output, region, scenario, lower_bound, upper_bound):
    df = SQLConnection("all_data_jan_2024").single_output_df(output, region, scenario)
    df_to_graph = df.groupby(["Year"])["Value"].agg([
        lambda x: np.percentile(x, lower_bound),
        np.median,
        lambda x: np.percentile(x, upper_bound)
    ])
    df_to_graph.columns = ['Lower Bound', 'Median', 'Upper Bound']

    return df_to_graph

def add_new_trace(output, region, scenario, lower_bound, upper_bound):
    df = get_data(output, region, scenario, lower_bound, upper_bound)
    return go.Scatter(
            x = df.index,
            y = df["Median"],
            name = "{} {}".format(region, scenario)
        )

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# Initialize the Dash app.
app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='output-dropdown',
        options=[{'label': r, 'value': r} for r in Options().outputs],
        value='consumption_billion_USD2007'  # Default value
    ),
    dcc.Dropdown(
        id='region-dropdown',
        options=[{'label': r, 'value': r} for r in ['USA', 'GLB', 'EUR']],
        value=['GLB'],  # Default value
        multi=True
    ),
    dcc.Dropdown(
        id='scenario-dropdown',
        options=[{'label': s, 'value': s} for s in ['Above2C_med', '15C_med', 'Ref']],
        value=['Ref'],  # Default value
        multi=True
    ),
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('output-dropdown', 'value'),
     Input('region-dropdown', 'value'),
     Input('scenario-dropdown', 'value')],
    [State('scatter-plot', 'figure')]
)
def update_graph(output_name, selected_regions, selected_scenarios, existing_figure):
    if not existing_figure or len(existing_figure.get('data')) == 0:
        df = get_data(output_name, "GLB", "Ref", 5, 95)
        trace = go.Scatter(
            x = df.index,
            y = df["Median"],
            name = "GLB Ref"
        )

        return go.Figure(data = trace)
    else:
        combos = list(product(selected_regions, selected_scenarios))
        current_trace_info = TraceInfo(existing_figure)
        current_traces = current_trace_info.traces
        existing_selections = set(current_trace_info.names)
        new_selections = set([reg + " " + sce for reg, sce in combos])

        # changes to make
        no_change = existing_selections.intersection(new_selections)
        to_delete = existing_selections.difference(new_selections)
        to_add = new_selections.difference(existing_selections)

        # removing traces
        indices_to_delete = [current_trace_info.names.index(i) for i in to_delete]
        for i in sorted(indices_to_delete, reverse = True):
            current_traces.pop(i)

        # adding traces
        new_traces = []
        for i in list(to_add):
            l = i.split(" ")
            reg, sce = l[0], l[1]
            new_trace = add_new_trace(output_name, reg, sce, 5, 95)
            new_traces.append(new_trace)
        
        return go.Figure(data = current_trace_info.traces + new_traces)

if __name__ == '__main__':
    app.run_server(debug=True)
