"""Shared base for dashboard figure builders."""

# Canonical keys for FinishedFigure.display_names_for_figure_type.
OUTPUT_TIMESERIES = "output-timeseries"


class DashboardFigure:
    """Plotly figure wrapper with metadata FinishedFigure expects."""

    def __init__(self, figure_type: str) -> None:
        self.figure_type = figure_type
        if not hasattr(self, "fig"):
            self.fig = None
            self.figure = None

    def set_fig(self, fig):
        """Assign the plotly figure and keep .figure as an alias for callers."""
        self.fig = fig
        self.figure = fig
        return fig

    def finish(self):
        """Apply the shared styling pipeline and return the styled figure."""
        from eppa_viz.figures.pipeline import apply_finished_style
        return apply_finished_style(self)
