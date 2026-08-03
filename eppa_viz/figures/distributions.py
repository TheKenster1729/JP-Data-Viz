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

class InputDistribution:
    def __init__(self, inputs):
        self.inputs = inputs
        self.input_df_with_run = pd.read_csv(r"Cleaned Data/InputsMasterTFP.csv").rename(columns = {"Unnamed: 0": "Run #"})
        self.input_df_no_run = self.input_df_with_run.drop(columns = "Run #")
        if len(self.inputs) > 1:
            self.colors = n_colors("rgb(173, 216, 230)", "rgb(128, 0, 128)", len(self.inputs), colortype = "rgb")
        else:
            self.colors = ["lightblue"]

    def create_input_distribution(self):
        fig = go.Figure()
        for i, inp in enumerate(self.inputs):
            fig.add_trace(go.Violin(x = self.input_df_no_run[inp], name = inp, line = dict(color = self.colors[i])))

        fig.update_traces(orientation = 'h', side = 'positive', width = 3, points = False)

        return fig

    def create_input_histogram(self, input_name):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x = self.input_df_no_run[input_name], name = input_name, showlegend = False))

        return fig

    def make_plot(self, input_name, show = False):
        violin = self.create_input_distribution()
        hist = self.create_input_histogram(input_name)

        plot = make_subplots(rows = 1, cols = 2)
        for i in range(len(self.inputs)):
            plot.add_trace(violin.data[i], row = 1, col = 1)
        plot.add_trace(hist.data[0], row = 1, col = 2)
        plot.update_layout(title = "Inputs Visualization",
                        width = 1400,
                        height = 700)
        plot.update_xaxes(title = "Input Comparison", row = 1, col = 1)
        plot.update_xaxes(title = "Input Focus - {}".format(input_name), row = 1, col = 2)

        # if focus input is on violin plot, change color accordingly
        try:
            index = self.inputs.index(input_name)
            color = plot.data[index].line.color
            plot.update_traces(overwrite = True, marker = dict(color = color))
        except ValueError:
            pass

        if show:
            plot.show()

        return plot

class InputDistributionAlternate:
    def __init__(self, inputs):
        self.inputs = inputs
        self.input_df_with_run = pd.read_csv(r"Cleaned Data/InputsMasterTFP.csv").rename(columns = {"Unnamed: 0": "Run #"})
        self.input_df_no_run = self.input_df_with_run.drop(columns = "Run #")
        if len(self.inputs) > 1:
            self.colors = n_colors("rgb(173, 216, 230)", "rgb(128, 0, 128)", len(self.inputs), colortype = "rgb")
        else:
            self.colors = ["lightblue"]

    def create_input_distribution(self):
        fig = go.Figure()
        for i, inp in enumerate(self.inputs):
            fig.add_trace(go.Violin(x = self.input_df_no_run[inp], name = inp, line = dict(color = self.colors[i])))

        fig.update_traces(orientation = 'h', side = 'positive', width = 3, points = False)

        return fig

    def create_input_histogram(self, input_name):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x = self.input_df_no_run[input_name], name = input_name, showlegend = False))

        return fig

    def make_plot(self, show = False):
        # Create a figure with subplots for each input
        num_inputs = len(self.inputs)
        cols = 4  # Define the number of columns for the subplot grid
        rows = -(-num_inputs // cols)  # Calculate rows needed, round up division
        subplot_titles = [f"{input}" for input in self.inputs]
        
        plot = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        
        # Iterate over inputs to create a histogram for each
        for i, input in enumerate(self.inputs, start=1):
            hist_data = self.input_df_no_run[input]
            row = (i - 1) // cols + 1
            col = (i - 1) % cols + 1

            plot.add_trace(
                go.Histogram(x=hist_data, name=input, showlegend=False, marker_color = self.colors[i-1]),
                row=row, col=col
            )
        
        # Update layout to adjust for the number of subplots
        plot.update_layout(
            title_text="Input Distributions",
            height=300 * rows,  # Adjust height based on the number of rows
            width=1000
        )

        if show:
            plot.show()

        return plot
    
class OutputDistribution:
    def __init__(self, output):
        self.output = output

    def create_output_distribution_all_years(self, regions, scenarios):
        query = "SELECT * FROM fulldataset WHERE OutputName='{}'".format(self.output)
        df = SQLToDataframe("test_db", "fulldataset").read_query(query)
        fig = go.Figure()
        for region in regions:
            for scenario in scenarios:
                df_region_scenario = df[(df['Region'] == region) & (df['Scenario'] == scenario)]
                fig.add_trace(go.Violin(
                    x = df_region_scenario['Year'],
                    y = df_region_scenario['Value'],
                    name = f'{region} {scenario}',
                    box_visible = True,
                    meanline_visible = True,
                ))

        fig.update_layout(
            xaxis_title = 'Year',
            yaxis_title = 'Value',
            title = 'Violin plot by Region and Scenario',
            hovermode = 'x',
            violinmode = "group",
        )

        return fig

    def create_output_distribution_one_year(self, regions, scenarios, year):
        fig = go.Figure()
        for scenario in scenarios:
            df = SQLConnection("jp_data").read_data_from_sql_table(self.output, selected_region = regions, selected_year = year, selected_scenario = scenario)
            print(df)

        return fig

    def create_full_figure(regions, scenarios, year):
        pass

