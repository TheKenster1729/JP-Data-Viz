# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.graph_objects as go
# import numpy as np

# app = dash.Dash(__name__)

# # Sample data
# np.random.seed(42)
# num_vars = 4
# num_samples = 100
# data = np.random.rand(num_samples, num_vars)

# # Create parallel coordinates plot
# fig = go.Figure(data=go.Parcoords(
#     line=dict(color=np.random.randint(low=0, high=num_samples, size=num_samples),
#               colorscale='Viridis',
#               showscale=True),
#     dimensions=[{'label': f'Variable {i+1}', 'values': data[:, i]} for i in range(num_vars)]
# ))

# # Overlaying transparent scatter for interaction
# for i in range(num_samples):
#     fig.add_trace(
#         go.Scatter(
#             x=list(range(num_vars)), y=data[i, :], mode='lines+markers',
#             marker=dict(color='rgba(0,0,0,0)'), line=dict(width=3, color='rgba(0,0,0,0)'),
#             hoverinfo='text', text=[f"Var {j+1}: {data[i, j]:.2f}" for j in range(num_vars)],
#             showlegend=False
#         )
#     )

# app.layout = html.Div([
#     dcc.Graph(id='parallel-plot', figure=fig),
#     html.Div(id='hover-data', style={'padding': '10px', 'fontSize': '16px'})
# ])

# @app.callback(
#     Output('hover-data', 'children'),
#     [Input('parallel-plot', 'hoverData')]
# )
# def display_hover_data(hoverData):
#     if hoverData is not None:
#         print(hoverData)
#         return 'Hovering over:\n' + '\n'.join([f"{d['curveNumber']}: {d['text']}" for d in hoverData['points']])
#     return "Hover over the data!"

# if __name__ == '__main__':
#     app.run_server(debug=True)
import os
import itertools

directory = os.listdir("/Users/kcox1729/Downloads/0_New Ensembles2")
directory.remove(".DS_Store")
folder_pairs = list(itertools.combinations(directory, 2))

for f1, f2 in folder_pairs:
    files1 = os.listdir(os.path.join("/Users/kcox1729/Downloads/0_New Ensembles2", f1))
    files2 = os.listdir(os.path.join("/Users/kcox1729/Downloads/0_New Ensembles2", f2))

    files1_for_comparison = set([i.replace(f1.replace(".", ""), "") for i in files1])

    files2_for_comparison = set([i.replace(f2.replace(".", ""), "") for i in files2])
    
    print(f1, f2)
    print(files1_for_comparison.symmetric_difference(files2_for_comparison))
    print("\n")
