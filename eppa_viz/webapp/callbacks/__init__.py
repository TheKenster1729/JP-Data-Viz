from eppa_viz.webapp.callbacks.dropdowns import register_dropdown_callbacks
from eppa_viz.webapp.callbacks.plots import register_plot_callbacks


def register_callbacks(app):
    register_dropdown_callbacks(app)
    register_plot_callbacks(app)
