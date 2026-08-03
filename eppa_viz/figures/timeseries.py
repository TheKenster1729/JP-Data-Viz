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

class OldTimeSeries:
    def __init__(self, output, region, scenario, year, df, styling_options = None):
        self.output = output
        self.df = df
        self.lower = df[df.columns[0]]
        self.median = df.Median
        self.upper = df[df.columns[2]]
        self.region = region
        self.scenario = scenario
        self.year = year
        self.data_for_histogram = self.df.query("Year==@self.year")
        self.styling_options = styling_options
    
    def lower_bound_trace(self, group, color = "rgb(255,0,0)", marker = "dash"):

        trace = go.Scatter(
            x = self.df.index,
            y = self.lower,
            line = dict(color = color, dash = marker),
            legendgroup = group,
            showlegend = False,
            hoverinfo = "skip",
            name = "{} {}".format(self.region, self.scenario),
            customdata = ["{} {} {} lower".format(self.output, self.region, self.scenario)]
        )

        return trace
    
    def median_trace(self, group, color = "rgb(255,0,0)", marker = "dash"):

        trace = go.Scatter(
            x = self.df.index,
            y = self.median,
            # name = "{} {}".format(self.region, Options().scenario_display_names[self.scenario]),
            line = dict(color = color, dash = marker),
            legendgroup = group,
            name = "{} {}".format(self.region, self.scenario),
            customdata = ["{} {} {} median".format(self.output, self.region, self.scenario)]
        )

        return trace

    def upper_bound_trace(self, group, color = "rgb(255,0,0)", marker = "dash"):
        fillcolor = Color().convert_to_fill(color)

        trace = go.Scatter(
            x = self.df.index,
            y = self.upper,
            fill = "tonexty",
            fillcolor = fillcolor,
            line = dict(color = color, dash = marker),
            legendgroup = group,
            showlegend = False,
            hoverinfo = "skip",
            name = "{} {}".format(self.region, self.scenario),
            customdata = ["{} {} {} upper".format(self.output, self.region, self.scenario)]
        )

        return trace

    def return_traces(self, show = False, show_uncertainty = True):
        group = "{} {}".format(self.region, self.scenario)
        lower = self.lower_bound_trace(group, color = Color().region_colors[self.region], marker = Color().scenario_markers[self.scenario])
        median = self.median_trace(group, color = Color().region_colors[self.region], marker = Color().scenario_markers[self.scenario])
        upper = self.upper_bound_trace(group, color = Color().region_colors[self.region], marker = Color().scenario_markers[self.scenario])

        # fig.update_layout(
        #     yaxis_title = '{}'.format(self.output),
        #     hovermode = "x",
        #     title = '{}'.format(Readability().readability_dict_forward[self.output])
        # )

        # if show:
        #     fig.show()

        return [lower, upper, median]
    
    def make_histograms(self, show = False):
        hist = make_subplots(rows = len(self.scenarios), cols = len(self.regions))
        for i, region in enumerate(self.regions):
            for j, scenario in enumerate(self.scenarios):
                df_to_plot = self.data_for_histogram.query("Region==@region & Scenario==@scenario")
                hist.add_trace(go.Histogram(x = df_to_plot["Value"], marker_color = Color().region_colors[region], name = region + " " + scenario, legendgroup = region,
                                            marker = dict(pattern = dict(shape = Color().histogram_patterns[scenario])), hoverinfo = "name"), 
                               row = j + 1, col = i + 1)
        hist.update_layout(title_text = "Distributions for {}, {}".format(Readability().readability_dict_forward[self.output], self.year))

        if show:
            hist.show()

        return hist
    
    def make_plot(self, show = False, show_uncertainty = True, upper = 95, lower = 5):
        timeseries = self.make_timeseries_plot(upper, lower, show_uncertainty = show_uncertainty)
        hist = self.make_histograms()

        num_rows = len(self.regions)
        num_columns = len(self.scenarios) * 2

        # Create a new subplot figure with enough rows and columns for all subplots
        plot = make_subplots(rows = num_rows, cols = num_columns,
                specs = [[{"colspan": int(num_columns/2), "rowspan": num_rows}, *(None for i in range(int(num_columns/2) - 1))] + [{} for i in range(int(num_columns/2))]] + [[*(None for i in range(int(num_columns/2)))] + [{} for i in range(int(num_columns/2))] for j in range(num_rows - 1)],
                )
        # Add the timeseries plot to the first subplot
        for i in range(len(timeseries.data)):
            plot.add_trace(timeseries.data[i], row = 1, col = 1)

        # Add each histogram subplot to the new subplot figure
        for i, trace in enumerate(hist.data):
            row_and_col = [(row, col) for row in range(1, num_rows + 1) for col in range(int(num_columns/2) + 1, int(num_columns + 1))]          
            row = row_and_col[i][0]
            col = row_and_col[i][1]
            plot.add_trace(trace, row = row, col = col)

        plot.update_layout(title = "{} Timeseries and Distributions for {}".format(Readability().readability_dict_forward[self.output], self.year),
                        height = 700,
                        margin = dict(l = 0, r = 0)
                        )
        plot.update_xaxes(title = "Year", row = 1, col = 1)

        if len(self.scenarios) % 2 == 1:
            # odd number of histogram columns, easier case
            plot.update_xaxes(title = "{}".format(Readability().readability_dict_forward[self.output]), row = num_rows, col = int(num_columns/2) + len(self.regions) % 2)
        else:
            # even number of histograms, so must take average
            x_position = (num_columns / 2 + len(self.scenarios)/2) / num_columns
            y_position = -0.05

            # add annotation for the x-axis label
            plot.add_annotation(dict(
                x = x_position, y = y_position, showarrow = False,
                text = "{}".format(Readability().readability_dict_forward[self.output]), xref = "paper", yref = "paper",
                xanchor = "center", yanchor = "top",
                font = dict(size = 14)
            ))

        if show:
            plot.show()

        return plot

class NewTimeSeries(DashboardFigure):
    def __init__(self, output, region, scenario, year, df, styling_options = {"color": "by-scenario"}):
        super().__init__(OUTPUT_TIMESERIES)
        self.output = output
        self.df = df
        self.lower = df[df.columns[0]]
        self.median = df.Median
        self.upper = df[df.columns[2]]
        self.region = region
        self.scenario = scenario
        self.year = year
        self.data_for_histogram = self.df.query("Year==@self.year")
        self.styling_options = styling_options
        if self.scenario == "2C" or self.scenario == "Ref":
            self.scenario_display_name = {"2C": "2C", "Ref": "Ref"}[self.scenario]
        else:
            self.scenario_display_name = Options().scenario_display_names[self.scenario]

        self.return_figure()

    def get_color(self):
        if self.styling_options["color"] == "by-region":
            color = Color().region_colors[self.region]
        elif self.styling_options["color"] == "by-scenario":
            color = Color().scenario_colors[self.scenario]
        elif self.styling_options["color"] == "standard":
            base_shade = Color().region_colors[self.region]
            if self.scenario == "2C" or self.scenario == "Ref":
                amount_to_lighten = 15 if self.scenario == "2C" else 0
            else:
                amount_to_lighten = Options().scenarios.index(self.scenario)
            color = Color().lighten_hex(base_shade, brightness_offset = amount_to_lighten*8)
        return color

    def lower_bound_trace(self, group, marker = "dash"):
        color = self.get_color()
        trace = go.Scatter(
            x = self.df.index,
            y = self.lower,
            line = dict(color = color),
            legendgroup = group,
            showlegend = False,
            hoverinfo = "skip",
            customdata = ["{} {} {} lower".format(self.output, self.region, self.scenario)],
            uid = sanitize_uid(self.output, self.region, self.scenario),
        )

        return trace

    def median_trace(self, group, marker = "dash"):
        color = self.get_color()
        trace = go.Scatter(
            x = self.df.index,
            y = self.median,
            line = dict(color = color),
            legendgroup = group,
            name = "{} {}".format(self.region, self.scenario_display_name),
            customdata = ["{} {} {} median".format(self.output, self.region, self.scenario)],
            uid = sanitize_uid(self.output, self.region, self.scenario),
        )

        return trace

    def upper_bound_trace(self, group, marker = "dash"):
        color = self.get_color()
        fillcolor = Color().convert_to_fill(color)

        trace = go.Scatter(
            x = self.df.index,
            y = self.upper,
            fill = "tonexty",
            fillcolor = fillcolor,
            line = dict(color = color),
            legendgroup = group,
            showlegend = False,
            hoverinfo = "skip",
            customdata = ["{} {} {} upper".format(self.output, self.region, self.scenario)],
            uid = sanitize_uid(self.output, self.region, self.scenario),
        )

        return trace

    def return_traces(self):
        group = "{} {}".format(self.region, self.scenario)
        lower = self.lower_bound_trace(group)
        median = self.median_trace(group)
        upper = self.upper_bound_trace(group)

        return [lower, upper, median]

    def return_figure(self):
        self.set_fig(go.Figure(data=self.return_traces()))

    def make_histograms(self, show = False):
        hist = make_subplots(rows = len(self.scenarios), cols = len(self.regions))
        for i, region in enumerate(self.regions):
            for j, scenario in enumerate(self.scenarios):
                df_to_plot = self.data_for_histogram.query("Region==@region & Scenario==@scenario")
                hist.add_trace(go.Histogram(x = df_to_plot["Value"], marker_color = Color().region_colors[region], name = region + " " + scenario, legendgroup = region,
                                            marker = dict(pattern = dict(shape = Color().histogram_patterns[scenario])), hoverinfo = "name"), 
                               row = j + 1, col = i + 1)
        if self.output in Options().outputs:
            title_text = "Distributions for {}, {}".format(Readability().readability_dict_forward[self.output], self.year)
        else:
            title_text = "Distributions for {}".format(json.loads(self.output)["name"])
        hist.update_layout(title_text = title_text)

        if show:
            hist.show()

        return hist

    def make_plot(self, show = False, show_uncertainty = True, upper = 95, lower = 5):
        traces = self.return_traces(show = show, show_uncertainty = show_uncertainty)
        fig = go.Figure(traces)

class ModifyOutputTimeseries(DashboardFigure):
    def __init__(self, output, regions, scenarios, existing_figure, styling_params, database, lower_bound = 5, upper_bound = 95, change_fig = False):
        super().__init__(OUTPUT_TIMESERIES)
        self.output = output
        self.regions = regions
        self.scenarios = scenarios
        self.existing_figure = existing_figure
        self.existing_figure_uids_list = [trace.uid for trace in self.existing_figure.data]
        self.existing_figure_uids_set = set(self.existing_figure_uids_list)
        self.styling_params = styling_params
        self.database = database
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.change_fig = change_fig

    def get_combinations(self):
        if not isinstance(self.regions, list):
            self.regions = [self.regions]
        if not isinstance(self.scenarios, list):
            self.scenarios = [self.scenarios]

        return product([self.output], self.regions, self.scenarios)

    def remove_traces(self):
        # Create a NEW figure object (not a reference) to ensure Dash detects the change
        combinations = self.get_combinations()
        new_uids = set([sanitize_uid(output, region, scenario) for output, region, scenario in combinations])
        uids_to_remove = self.existing_figure_uids_set - new_uids
        traces_to_keep = [trace for trace in self.existing_figure.data if trace.uid not in uids_to_remove]
        
        # Create a completely new figure with the traces we want to keep
        new_figure = go.Figure(data=traces_to_keep)
        # Copy over the layout from the existing figure
        new_figure.update_layout(self.existing_figure.layout)

        return new_figure

    def trace_already_exists(self, uid, figure):
        return uid in [trace.uid for trace in figure.data]

    def get_df(self, output, region, scenario):
        df = DataRetrieval(self.database, output, region, scenario).single_output_df_to_graph(self.lower_bound, self.upper_bound)
        return df

    def create_new_figure(self):
        combinations = self.get_combinations()
        combinations_list = [i for i in combinations]
        
        if self.change_fig:
            # When change_fig is True, create a completely fresh figure (no reference to existing)
            # This ensures Dash always detects it as a new figure
            new_figure = go.Figure()
            for combo in combinations_list:
                df = self.get_df(combo[0], combo[1], combo[2])
                traces_to_add = NewTimeSeries(combo[0], combo[1], combo[2], 2050, df, styling_options = self.styling_params).return_traces()
                new_figure.add_traces(traces_to_add)
        else:
            # Only when NOT changing, try to preserve existing traces
            new_figure = self.remove_traces()
            for combo in combinations_list:
                uid = sanitize_uid(combo[0], combo[1], combo[2])
                if not self.trace_already_exists(uid, new_figure):
                    df = self.get_df(combo[0], combo[1], combo[2])
                    traces_to_add = NewTimeSeries(combo[0], combo[1], combo[2], 2050, df, styling_options = self.styling_params).return_traces()
                    new_figure.add_traces(traces_to_add)

        return new_figure

