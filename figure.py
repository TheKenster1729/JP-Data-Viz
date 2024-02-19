import plotly.graph_objects as go
from analysis import InputOutputMapping
from styling import Color, Options, Readability
import pandas as pd
from plotly.subplots import make_subplots
from sql_utils import SQLConnection, DataRetrieval
from processing import LoadData
import numpy as np
import plotly.express as px
from itertools import product
from plotly.colors import n_colors

class TraceInfo:
    def __init__(self, figure):
        self.fig = figure
        self.traces = self.__traces()
        self.number = self.__len__()
        self.names = self.__trace_names()
        self.colors = self.__trace_colors()
        self.uid_names = self.__uid_names()
        self.custom_data = self.__custom_data()
        self.type = self.__type()

    def __getitem__(self, index):
        return self.traces[index]

    def __len__(self):
        return len(self.traces)

    def __traces(self):
        return self.fig.get('data', [])

    def __trace_colors(self):
        return [trace.get('marker', {}).get('color') for trace in self.traces]

    def __trace_names(self):
        return [trace.get('name') for trace in self.traces]
    
    def __uid_names(self):
        return [trace.get('uid') for trace in self.traces]
    
    def __custom_data(self):
        return [trace.get("customdata") for trace in self.traces]
    
    def __type(self):
        return [trace.get("type") for trace in self.traces]

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
    
class NewTimeSeries:
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
            name = "{} {}".format(self.region, Options().scenario_display_names[self.scenario]),
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

class OutputHistograms:
    def __init__(self, output, regions, scenarios, year, db_obj, styling_options = None):
        self.output = output
        self.db_obj = db_obj
        self.regions = regions
        self.scenarios = scenarios
        self.year = year
        self.styling_options = styling_options

    def get_data(self, region, scenario):
        df = DataRetrieval(self.db_obj, self.output, region, scenario).single_output_df()
        return df.query("Year==@self.year")

    def make_plot(self, show = False):
        # Create an empty figure
        fig = make_subplots(rows = len(self.regions), cols = len(self.scenarios))

        # Loop through each combination of region and scenario
        for i, region in enumerate(self.regions):
            for j, scenario in enumerate(self.scenarios):
                # Fetch the data for this combination
                df = self.get_data(region, scenario)
                
                # Add a histogram to the figure for this combination
                fig.add_trace(go.Histogram(x = df['Value'],
                                        name = f"{region} - {Options().scenario_display_names[scenario]}",
                                        opacity = 0.75),
                                        row = i + 1,
                                        col = j + 1)

        # Update the layout of the figure
        fig.update_layout(
            title_text = f"Histograms for {Readability().naming_dict_long_names_first[self.output]}, {self.year}",  # Title
            xaxis_title_text = 'Value',  # X-axis label
            yaxis_title_text = 'Count'  # Y-axis label
        )
        
        if show:
            fig.show()

        return fig
   

class InputDistribution:
    def __init__(self, inputs):
        self.inputs = inputs
        self.input_df_with_run = pd.read_csv(r"Cleaned Data/InputsMaster.csv").rename(columns = {"Unnamed: 0": "Run #"})
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
                        plot_bgcolor = "#060606",
                        paper_bgcolor = "#060606",
                        font_color = "white",
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

class InputOutputMappingPlot(InputOutputMapping):
    def __init__(self, output, df):
        super().__init__(output, df)

    def make_plot(self, num_to_plot = 5, show = False):
        feature_importances, sorted_labeled_importances, top_n = self.random_forest(num_to_plot = num_to_plot)
        fig = make_subplots(cols = 2, specs = [[{"type": "xy"}, {"type": "domain"}]], column_widths = [0.4, 0.6])

        _, y_discrete = self.preprocess_for_classification()
        y_discrete_series = pd.Series(y_discrete.ravel(), name = "y_discrete")
        parcoords_df = pd.concat([self.inputs[top_n], self.y_continuous, y_discrete_series], axis = 1)

        dimensions = []
        color_scale = [(0.00, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[0]), (0.5, Color().parallel_coords_colors[1]),  (1.00, Color().parallel_coords_colors[1])]
        for col in parcoords_df.columns[:-1]:
            dimensions.append(dict(label = col, values = parcoords_df[col]))
        fig.add_trace(go.Bar(x = top_n, y = sorted_labeled_importances[top_n]), row = 1, col = 1)
        fig.add_trace(go.Parcoords(line = dict(color = parcoords_df["y_discrete"], colorscale = color_scale),
                                      dimensions = dimensions, labelside = "bottom"), row = 1, col = 2)
        fig.update_layout(margin = dict(l = 50, r = 50, t = 50),
                          title = "Feature Importances and Parallel Plot",
                          width = 1200,
                          height = 600)

        
        if show:
            fig.show()

        return fig

class ParallelCoords:
    pass

if __name__ == "__main__":
    # timeseries
    db_obj = SQLConnection("all_data_jan_2024")
    df = DataRetrieval(db_obj, "consumption_billion_usd2007", "GLB", "Ref").single_output_df()
    # lower = TimeSeries("consumption_billion_usd2007", "GLB", "Ref", 2050, df).lower_bound_trace(None)
    # upper = TimeSeries("consumption_billion_usd2007", "GLB", "Ref", 2050, df).upper_bound_trace(None)
    # median = TimeSeries("consumption_billion_usd2007", "GLB", "Ref", 2050, df).median_trace(None)
    # fig = go.Figure()
    # fig.add_trace(lower)
    # fig.add_trace(upper)
    # fig.add_trace(median)
    # fig.show()
    OutputHistograms("consumption_billion_USD2007", ["GLB", "USA", "EUR"], ["Ref", "Above2C_med"], 2050, db_obj).make_plot(show = True)
    # input dist
    # fig = InputDistribution(["WindGas", "wind", "BioCCS", "gas", "oil", "coal"]).make_plot("WindGas")
    # fig.show()
    # fig.write_image("assets\examples\inputs_windgas_focus.svg")

    # input-output mapping
    # df = SQLConnection("jp_data").input_output_mapping_df("elec_prod_Renewables_TWh", "USA", "2C", 2050)
    # fig = InputOutputMappingPlot("total_emissions_CO2_million_ton_CO2", df).make_plot()
    # fig.write_image("assets\examples\cart_usa_2c_2050.svg")
