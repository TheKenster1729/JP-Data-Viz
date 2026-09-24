"""Create and configure the Dash application."""
import dash
import dash_bootstrap_components as dbc

from eppa_viz.webapp.callbacks import register_callbacks
from eppa_viz.webapp.layouts import navbar, build_main_layout


def create_app():
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.PULSE, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.layout = build_main_layout(navbar)
    register_callbacks(app)
    return app
