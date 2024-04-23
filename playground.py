import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Output, Input, State
from dash import callback_context

app = dash.Dash(__name__)

values1 = np.random.randint(100, 400, 100)
values2 = np.random.randint(100, 400, 100)
values3 = np.random.randint(100, 400, 100)
values4 = np.random.randint(100, 400, 100)

app.layout = html.Div([
    dcc.Graph(
        id='parallel-coordinates-plot',
        figure={
            'data': [
                go.Parcoords(
                    line=dict(color='blue'),
                    dimensions=list([
                        dict(range=[100,400],
                             constraintrange=[100,200],
                             label='Dimension 1', values=values1),
                        dict(range=[100,400],
                             label='Dimension 2', values=values2),
                        dict(range=[100,400],
                             label='Dimension 3', values=values3),
                        dict(range=[100,400],
                             label='Dimension 4', values=values4)
                    ])
                )
            ],
            'layout': go.Layout(
                title='Parallel Coordinates Plot',
                plot_bgcolor = 'white'
            )
        }
    ),
    html.Div([
        dcc.RangeSlider(
            id='my-slider',
            min=100,
            max=400,
            step=1,
            value=[250, 300],
            marks={i: '{}'.format(i) for i in range(100, 401, 50)}
        ),
        html.Div(id='slider-output-container')
    ])
])

@app.callback(
    Output('parallel-coordinates-plot', 'figure'),
    Input('my-slider', 'value'),
    Input('parallel-coordinates-plot', 'figure')
)
def constraintrange(value, figure):
    figure['data'][0]['dimensions'][1]['constraintrange'] = value
    print(callback_context.triggered[0]["prop_id"])
    print(figure['data'][0]['dimensions'][2].get('constraintrange'))
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
