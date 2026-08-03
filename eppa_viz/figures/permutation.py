"""Dashboard figure builders (plotly)."""

import json
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors

from analysis import (
    FilteredInputOutputMapping,
    FilteredOutputOutputMapping,
    InputOutputMapping,
    OutputOutputMapping,
    TimeSeriesClustering,
)
from eppa_viz.figures.base import DashboardFigure, OUTPUT_TIMESERIES
from eppa_viz.figures.utils import sanitize_uid, TraceInfo
from sql_utils import DataRetrieval, SQLConnection
from styling import Color, Options, Readability



def _scenario_subplot_title(scenario):
    opts = Options()
    if scenario in opts.publication_scenario_display_names:
        return opts.publication_scenario_display_names[scenario]
    return opts.scenario_display_names[scenario]

class PermutationImportance(DashboardFigure):
    def __init__(self, df, output, region, scenario, year, n_estimators = 100, max_depth = 4):
        super().__init__("permutation-importance")
        self.df = df
        self.output = output
        self.region = region
        self.scenario = scenario
        self.year = year
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.fig = self.make_plot()

    def make_plot(self, show = False):
        results = InputOutputMapping(self.output, self.region, self.scenario, self.year, self.df, n_estimators = self.n_estimators, max_depth = self.max_depth).permutation_importance()
        fig = go.Figure()
        fig.add_trace(go.Bar(x = [k["variable"] for k in results], y = [k["mean"] for k in results], error_y = dict(type = "data", array = [k["std"] for k in results])))
        fig.update_layout(showlegend = False)

        if show:
            fig.show()

        return fig

