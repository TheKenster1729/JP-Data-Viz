import dash
import dash_html_components as html

app = dash.Dash(__name__)

# Suppose your SVG content is stored in a variable named `svg_content`
svg_content = """
<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />
</svg>
"""

app.layout = html.Div([
    html.Div(svg_content)
])

if __name__ == '__main__':
    app.run_server(debug=True)
