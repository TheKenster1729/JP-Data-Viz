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

class ChoroplethMap(DashboardFigure):
    def __init__(self, df, output, scenario, year, lower_bound, upper_bound) -> None:
        super().__init__("choropleth-map")
        self.df = df
        self.output = output
        self.scenario = scenario
        self.region = "GLB"
        self.year = year
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.fig = self.make_plot()

    def number_to_ordinal(self, n):
        """
        Convert an integer from 1 to 100 into its English ordinal representation.
        
        Args:
        n (int): Integer from 1 to 100
        
        Returns:
        str: The ordinal representation of n
        """
        if 11 <= n <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return str(n) + suffix

    def make_plot(self, show = False):
        # Retrieve the data for the specified parameters
        lower_bound_column_name = '{} Percentile'.format(self.number_to_ordinal(self.lower_bound))
        upper_bound_column_name = '{} Percentile'.format(self.number_to_ordinal(self.upper_bound))

        global_min = self.df[[upper_bound_column_name, "Median", upper_bound_column_name]].min().min()
        global_max = self.df[[upper_bound_column_name, "Median", upper_bound_column_name]].max().max()

        # Load the spatial data
        gdf = gpd.read_file(r"assets/Eppa countries/eppa6_regions_simplified.shp").rename(columns = {"EPPA6_Regi": "Region"})

        # Merge the data with the spatial data
        merged_gdf = gdf.merge(self.df, on = "Region")
        geojson = json.loads(merged_gdf.to_json())
        for feature in geojson['features']:
                feature['id'] = feature['properties']['Region']

        merged_gdf["text"] = "Lower Bound: " + merged_gdf[lower_bound_column_name].apply(lambda x: str(int(x))) + "<br>" + "Upper Bound: " + merged_gdf[upper_bound_column_name].apply(lambda x: str(int(x)))
        # Create a choropleth map
        fig = go.Figure(go.Choropleth(
            geojson = geojson,
            locations = merged_gdf["Region"],
            z = merged_gdf["Median"],
            featureidkey = "properties.Region",
            colorscale = "bupu",
            marker_line_color = 'black',
            marker_line_width = 0.5,
            hovertext = merged_gdf["text"]
        ))
        # lower = go.Figure(go.Choropleth(
        #     geojson = geojson,
        #     locations = merged_gdf["Region"],
        #     z = merged_gdf[lower_bound_column_name],
        #     featureidkey = "properties.Region",
        #     colorscale = "Viridis",
        #     marker_line_color = 'black',
        #     marker_line_width = 0.5,
        #     zmin = global_min,
        #     zmax = global_max,
        #     showscale = False
        # ))
        # mid = go.Figure(go.Choropleth(
        #     geojson = geojson,
        #     locations = merged_gdf["Region"],
        #     z = merged_gdf['Median'],
        #     featureidkey = "properties.Region",
        #     colorscale = "Viridis",
        #     marker_line_color = 'black',
        #     marker_line_width = 0.5,
        #     zmin = global_min,
        #     zmax = global_max,
        #     showscale = False
        # ))
        # upper = go.Figure(go.Choropleth(
        #     geojson = geojson,
        #     locations = merged_gdf["Region"],
        #     z = merged_gdf[upper_bound_column_name],
        #     featureidkey = "properties.Region",
        #     colorscale = "Viridis",
        #     marker_line_color = 'black',
        #     marker_line_width = 0.5,
        #     zmin = global_min,
        #     zmax = global_max,            
        #     showscale = True,
        #     colorbar_title = Readability().naming_dict_long_names_first[self.output],
        #     colorbar = dict(orientation = 'h')
        # ))

        if show:
            fig.show()

        return fig

