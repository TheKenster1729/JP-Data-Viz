"""Mandatory styling entry point for dashboard figures."""

from styling import FinishedFigure


def apply_finished_style(fig_object):
    """Run FinishedFigure and return fig_object.fig."""
    FinishedFigure(fig_object).style_figure()
    return fig_object.fig
