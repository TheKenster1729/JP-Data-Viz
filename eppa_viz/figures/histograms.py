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

class OutputHistograms(DashboardFigure):
    def __init__(self, output, regions, scenarios, year, db_obj, styling_options = None):
        super().__init__("output-histograms")
        self.output = output
        self.db_obj = db_obj
        self.regions = regions
        self.scenarios = scenarios
        self.year = year
        self.styling_options = styling_options

    def get_data(self, region, scenario):
        df = DataRetrieval(self.db_obj, self.output, region, scenario).single_output_df()
        return df.query("Year==@self.year")

    def get_color(self, region, scenario):
        if self.styling_options["color"] == "by-region":
            color = Color().region_colors[region]
        elif self.styling_options["color"] == "by-scenario":
            color = Color().scenario_colors[scenario]
        elif self.styling_options["color"] == "standard":
            base_shade = Color().region_colors[region]
            if scenario in ("2C", "Ref"):
                amount_to_lighten = 15 if scenario == "2C" else 0
            else:
                amount_to_lighten = Options().scenarios.index(scenario)
            color = Color().lighten_hex(base_shade, brightness_offset=amount_to_lighten * 8)
        return color

    def make_plot(self, show = False):
        # Create an empty figure
        fig = make_subplots(rows = len(self.regions), cols = len(self.scenarios), subplot_titles = [_scenario_subplot_title(scenario) for scenario in self.scenarios])

        # Loop through each combination of region and scenario
        for i, region in enumerate(self.regions):
            for j, scenario in enumerate(self.scenarios):
                # Fetch the data for this combination
                df = self.get_data(region, scenario)
                
                # Add a histogram to the figure for this combination
                trace_to_add = go.Histogram(x=df["Value"],
                                            name=f"{region} - {_scenario_subplot_title(scenario)}",
                                            marker_color = self.get_color(region, scenario),
                                            opacity = 0.75,
                                            )
                fig.add_trace(trace_to_add,
                            row = i + 1,
                                            col = j + 1)
                    
                if j == 0:
                    fig.update_yaxes(title_text = region, row = i + 1, col = j + 1)
        
        if show:
            fig.show()

        return fig
   
