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

class STRESSPlatformConnection(DashboardFigure):
    def __init__(self, db_obj, inputs, outputs, color, region, scenario, year):
        super().__init__("stress-platform-connection")
        self.db_obj = db_obj
        self.inputs = inputs
        self.outputs = outputs
        self.color = color
        self.region = region
        self.scenario = scenario
        self.year = year

    def make_plot(self, show = False):
        fig = go.Figure()

        # construct dataframe for parallel coordinates plot
        inputs_df = pd.read_csv(f"Cleaned Data/InputsMasterTFP.csv")
        parcoords_df = DataRetrieval(self.db_obj, self.color, self.region, self.scenario, self.year).mapping_df().drop(columns = "Year").rename(columns = {"Value": self.color})
        for output in self.outputs:
            df = DataRetrieval(self.db_obj, output, self.region, self.scenario, self.year).mapping_df().drop(columns = "Year")
            parcoords_df = parcoords_df.merge(df.rename(columns = {"Value": output}), on = "Run #")
        parcoords_df = parcoords_df.merge(inputs_df[["Run #"] + self.inputs], on = "Run #")

        # construct plot
        dimensions = []
        # color_scale = [(0.00, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[1]),  (1.00, Color().parallel_coords_colors[1])]
        for col in parcoords_df.columns[2:]:
            dimensions.append(dict(label = col, values = parcoords_df[col]))
        fig = go.Figure(
            go.Parcoords(line = dict(color = parcoords_df[self.color], colorscale = "viridis", showscale = True),
                                    dimensions = dimensions, labelside = "bottom"
            )
        )
        fig.update_layout(width = 1200, height = 800)
        if show:
            fig.show()

        return fig, parcoords_df

