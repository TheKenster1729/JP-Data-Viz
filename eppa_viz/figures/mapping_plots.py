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

class InputOutputMappingPlot(InputOutputMapping, DashboardFigure):
    def __init__(self, output, region, scenario, year, df, threshold = 70, gt = True, num_to_plot = 5, n_estimators = 100, max_depth = 4):
        super().__init__(output, region, scenario, year, df, threshold = threshold, gt = gt, num_to_plot = num_to_plot, n_estimators = n_estimators, max_depth = max_depth)
        DashboardFigure.__init__(self, "input-output-mapping-main")

        self.fig = self.make_plot()

    def make_plot(self, show = False, save = False):
        feature_importances, sorted_labeled_importances, top_n = self.random_forest()
        fig = make_subplots(cols = 2, specs = [[{"type": "xy"}, {"type": "domain"}]], column_widths = [0.4, 0.6], 
                            subplot_titles = ("Feature Importances, Top 5 Features", "Parallel Axis Plot, Top 5 Features"))

        parcoords_df = self.inputs[top_n].copy()
        _, y_discrete = self.preprocess_for_classification()
        output_display_name = Readability().display_name_for_output(self.output)
        parcoords_df[output_display_name] = self.y_continuous.values
        parcoords_df["y_discrete"] = y_discrete

        first_sign = ">" if self.gt else "<"
        second_sign = "<" if self.gt else ">"
        ending = Readability().ordinal(self.threshold)

        dimensions = []
        color_scale = [(0.00, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[1]),  (1.00, Color().parallel_coords_colors[1])]
        for col in parcoords_df.columns[:-1]:
            dimensions.append(dict(label = col, values = parcoords_df[col]))
        fig.add_trace(go.Bar(x = top_n, y = sorted_labeled_importances[top_n]), row = 1, col = 1)
        fig.add_trace(go.Parcoords(line = dict(color = parcoords_df["y_discrete"], colorscale = color_scale, showscale = True,
                                    colorbar=dict(
                                    title='Group',
                                    tickvals=[0.25, 0.75],  # Positions at which ticks should be displayed
                                    ticktext=[f'{second_sign}{ending} Percentile', f'{first_sign}{ending} Percentile'],  # Text displayed at the ticks
                                    len=0.25,  # Makes the colorbar shorter
                                    y=0.5,   # Position the colorbar in the middle of the plot
                                    yanchor='middle')),
                                    dimensions = dimensions, labelside = "bottom"), 
                                    row = 1, col = 2)
        if show:
            fig.show()

        if save:
            fig.write_image(save + ".png", scale = 2)

        return fig

class OutputOutputMappingPlot(OutputOutputMapping, DashboardFigure):
    def __init__(self, db_obj, output, region, scenario, year, df, threshold = 70, gt = True, num_to_plot = 5):
        super().__init__(db_obj, output, region, scenario, year, df, threshold = threshold, gt = gt, num_to_plot = num_to_plot)
        DashboardFigure.__init__(self, "output-output-mapping-main")

        self.fig = self.make_plot()

    def make_plot(self, show = False, save = False):
        result = self.random_forest()
        if type(result) is str:
            return result
        
        feature_importances, sorted_labeled_importances, top_n = result[0], result[1], result[2]
        fig = make_subplots(cols = 2, specs = [[{"type": "xy"}, {"type": "domain"}]], column_widths = [0.4, 0.6], 
                            subplot_titles = ("Feature Importances, Top 5 Features", "Parallel Axis Plot, Top 5 Features"))

        parcoords_df = self.main_df[top_n].copy()
        y_discrete = self.preprocess_for_classification()
        output_display_name = Readability().display_name_for_output(self.output)
        parcoords_df[output_display_name] = self.y_continuous.values
        parcoords_df["y_discrete"] = y_discrete

        dimensions = []
        color_scale = [(0.00, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[1]),  (1.00, Color().parallel_coords_colors[1])]
        for col in parcoords_df.columns[:-1]:
            dimensions.append(dict(label = col, values = parcoords_df[col]))
        fig.add_trace(go.Bar(x = top_n, y = sorted_labeled_importances[top_n]), row = 1, col = 1)
        fig.add_trace(go.Parcoords(line = dict(color = parcoords_df["y_discrete"], colorscale = color_scale),
                                      dimensions = dimensions, labelside = "bottom"), row = 1, col = 2)
        if show:
            fig.show()

        if save:
            fig.write_image(save + ".png", scale = 2)

        return fig

class FilteredInputOutputMappingPlot(FilteredInputOutputMapping, DashboardFigure):
    def __init__(self, constraint_df, region, scenario, year, num_to_plot = 5, cart_depth = 4, n_estimators = 100, random_forest_depth = 4):
        super().__init__(constraint_df, region, scenario, year, num_to_plot = num_to_plot, cart_depth = cart_depth, n_estimators = n_estimators, random_forest_depth = random_forest_depth)
        DashboardFigure.__init__(self, "filtered-input-output-mapping-main")

        self.fig = self.make_plot()

    def make_plot(self, show = False, save = False):
        sortefeature_importances, sorted_labeled_importances, top_n = self.random_forest()
        fig = make_subplots(cols = 2, specs = [[{"type": "xy"}, {"type": "domain"}]], column_widths = [0.4, 0.6], 
                            subplot_titles = ("Feature Importances, Top 5 Features", "Parallel Axis Plot, Top 5 Features"))

        parcoords_df = self.X[top_n].copy()
        parcoords_df["y_discrete"] = self.y_discrete.values

        dimensions = []
        color_scale = [(0.00, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[1]),  (1.00, Color().parallel_coords_colors[1])]
        for col in parcoords_df.columns[:-1]:
            dimensions.append(dict(label = col, values = parcoords_df[col]))
        fig.add_trace(go.Bar(x = top_n, y = sorted_labeled_importances[top_n]), row = 1, col = 1)
        fig.add_trace(go.Parcoords(line = dict(color = parcoords_df["y_discrete"], colorscale = color_scale),
                                      dimensions = dimensions, labelside = "bottom", labelangle = 30), row = 1, col = 2)
        fig.update_annotations(yshift = 20)
        if show:
            fig.show()

        if save:
            fig.write_image(save + ".png", scale = 2)

        return fig

class FilteredOutputOutputMappingPlot(FilteredOutputOutputMapping, DashboardFigure):
    def __init__(self, db_obj, constraint_df, region, scenario, year, num_to_plot = 5):
        super().__init__(db_obj, constraint_df, region, scenario, year, num_to_plot = num_to_plot)
        DashboardFigure.__init__(self, "filtered-output-output-mapping-main")

        self.fig = self.make_plot()

    def make_plot(self, show = False, save = False):
        sorted_labeled_importances, top_n = self.run_analysis()
        
        if sorted_labeled_importances is None:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for analysis", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = make_subplots(cols = 2, specs = [[{"type": "xy"}, {"type": "domain"}]], column_widths = [0.4, 0.6], 
                            subplot_titles = ("Feature Importances, Top 5 Features", "Parallel Axis Plot, Top 5 Features"))

        parcoords_df = self.df_to_use[top_n].copy()
        parcoords_df["y_discrete"] = self.df_to_use["in_constraint_range"]

        dimensions = []
        color_scale = [(0.00, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[1]),  (1.00, Color().parallel_coords_colors[1])]
        for col in parcoords_df.columns[:-1]:
            dimensions.append(dict(label = col, values = parcoords_df[col]))
        fig.add_trace(go.Bar(x = top_n, y = sorted_labeled_importances[top_n]), row = 1, col = 1)
        fig.add_trace(go.Parcoords(line = dict(color = parcoords_df["y_discrete"], colorscale = color_scale),
                                      dimensions = dimensions, labelside = "bottom", labelangle = 30), row = 1, col = 2)
        fig.update_annotations(yshift = 20)
        fig.update_layout(width = 1600, height = 800)
        if show:
            fig.show()

        if save:
            fig.write_image(save + ".png", scale = 2)

        return fig

