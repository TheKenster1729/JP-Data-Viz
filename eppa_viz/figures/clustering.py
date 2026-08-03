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

class TimeSeriesClusteringPlot(TimeSeriesClustering, DashboardFigure):
    def __init__(self, df, output, region, scenario, n_clusters = 3, metric = "euclidean", year = None, num_to_plot = 5, cart_depth = 4, n_estimators = 100, max_depth = 4):
        super().__init__(df, output, region, scenario, n_clusters, metric = metric, num_to_plot = num_to_plot, cart_depth = cart_depth, n_estimators = n_estimators, max_depth = max_depth)
        DashboardFigure.__init__(self, "ts-clustering")
        self.colors = ["#648fff", "#491d8b", "#FFB000", "#a2191f", "#00539a", "#0e6027", "#565151"]
        self.fig = self.make_plot()
        self.year = year # this may never become relevant, it is just included to prevent errors during the styling process
        self.output = output
        self.region = region
        self.scenario = scenario

    def single_trace(self, data, cluster, color, showlegend = False):
        opacity = 1 if showlegend else 0.3
        line_thickness = 4 if showlegend else 0.5
        mode = "lines+markers" if showlegend else "lines"
        trace = go.Scatter(
            x = Options().years,
            y = data,
            legendgroup = cluster,
            mode = mode,
            line = dict(color = color, width = line_thickness),
            showlegend = showlegend,
            name = cluster,
            opacity = opacity
        )

        return trace

    def make_plot(self, show = False):
        clusters = self.generate_clusters()
        fig = go.Figure()
        cluster_labels = ["Cluster {}".format(str(i)) for i in range(1, self.n_clusters + 1)]

        assert len(clusters.labels_) == len(self.df_for_clustering)
        for i in range(len(self.df_for_clustering)):
            inidvidual_time_series = self.df_for_clustering.iloc[i].values
            cluster_number = clusters.labels_[i]
            cluster_label = cluster_labels[cluster_number]
            color = self.colors[cluster_number]

            trace = self.single_trace(inidvidual_time_series, cluster_label, color)

            fig.add_trace(trace)

        cluster_centers = clusters.cluster_centers_
        for i, yi in enumerate(cluster_centers):
            color = self.colors[i]
            trace = self.single_trace(yi.ravel(), cluster_labels[i], color, showlegend = True)
            fig.add_trace(trace)

        if show:
            fig.show()

        return fig

class TimeSeriesClusteringPlotCART(TimeSeriesClustering, DashboardFigure):
    def __init__(self, df, output, region, scenario, n_clusters = 3, metric = "euclidean", year = None, num_to_plot = 5, cart_depth = 4, n_estimators = 100, max_depth = 4):
        super().__init__(df, output, region, scenario, n_clusters, metric = metric, num_to_plot = num_to_plot, cart_depth = cart_depth, n_estimators = n_estimators, max_depth = max_depth)
        DashboardFigure.__init__(self, "ts-clustering-cart")
        self.colors = ["#648fff", "#491d8b", "#FFB000", "#a2191f", "#00539a", "#0e6027", "#565151"]
        self.fig = self.make_plot()
        self.year = year # this may never become relevant, it is just included to prevent errors during the styling process

    def make_plot(self, show = False):
        feature_importances, sorted_labeled_importances, top_n = self.cluster_mapping()
        fig = make_subplots(cols = 2, specs = [[{"type": "xy"}, {"type": "domain"}]], column_widths = [0.4, 0.6], 
                            subplot_titles = ("Feature Importances, Top 5 Features", "Parallel Axis Plot, Top 5 Features"))

        parcoords_df = self.inputs[top_n].copy()
        y_discrete = self.y
        parcoords_df[self.output] = self.y + np.ones(len(self.y)) # to index the clusters as 1, 2, etc instead of 0, 1, etc
        parcoords_df["y_discrete"] = y_discrete
        
        dimensions = []
        colors_to_use = self.colors
        color_scale = []
        for i in range(self.n_clusters):
            color_scale.append((i/self.n_clusters, colors_to_use[i]))
            color_scale.append(((i+1)/self.n_clusters, colors_to_use[i]))
        for col in parcoords_df.columns[:-1]:
            dimensions.append(dict(label = col, values = parcoords_df[col]))
        fig.add_trace(go.Bar(x = top_n, y = sorted_labeled_importances[top_n]), row = 1, col = 1)
        fig.add_trace(go.Parcoords(line = dict(color = parcoords_df["y_discrete"], colorscale = color_scale, showscale = True,
                                    colorbar=dict(
                                    title='Group',
                                    tickvals=[(self.n_clusters-1)/(2*self.n_clusters)*(2*i+1) for i in range(self.n_clusters)],  # Positions at which ticks should be displayed
                                    ticktext=['Cluster {}'.format(i) for i in range(1, self.n_clusters + 1)],  # Text displayed at the ticks
                                    len=0.25,  # Makes the colorbar shorter
                                    y=0.5,   # Position the colorbar in the middle of the plot
                                    yanchor='middle')),
                                    dimensions = dimensions, labelside = "bottom"), 
                                    row = 1, col = 2)
        if show:
            fig.show()

        fig.update_layout(width = 1000, height = 750)

        return fig

