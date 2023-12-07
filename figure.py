import plotly.graph_objects as go
from analysis import InputOutputMapping
from styling import Color, Options, Readability
import pandas as pd
from plotly.subplots import make_subplots
from sql_utils import SQLConnection
from processing import LoadData
import numpy as np
import plotly.express as px
from itertools import product
from plotly.colors import n_colors

class TimeSeries:
    def __init__(self, output, year, df):
        self.output = output
        self.df = df
        self.regions = self.df.Region.unique()
        self.scenarios = self.df.Scenario.unique()
        self.year = year
        self.data_for_histogram = self.df.query("Year==@self.year")

    def lower_bound_trace(self, data, group, bound, color = "black", marker = "dash"):
        lower = data.pivot_table(values = "Value", index = "Year", aggfunc = lambda x: np.percentile(x, bound))

        trace = go.Scatter(
            x = lower.index,
            y = lower.Value,
            line = dict(color = color, dash = marker),
            legendgroup = group,
            showlegend = False,
            hoverinfo = "skip"
        )

        return trace
    
    def median_trace(self, data, group, region, scenario, color = "black", marker = "dash"):
        median = data.pivot_table(values = "Value", index = "Year", aggfunc = np.median)

        trace = go.Scatter(
            x = median.index,
            y = median.Value,
            name = "{} {}".format(region, scenario),
            line = dict(color = color, dash = marker),
            legendgroup = group,
        )

        return trace

    def upper_bound_trace(self, data, group, bound, color = "black", marker = "dash"):
        upper = data.pivot_table(values = "Value", index = "Year", aggfunc = lambda x: np.percentile(x, bound))
        fillcolor = Color().convert_to_fill(color)

        trace = go.Scatter(
            x = upper.index,
            y = upper.Value,
            fill = "tonexty",
            fillcolor = fillcolor,
            line = dict(color = color, dash = marker),
            legendgroup = group,
            showlegend = False,
            hoverinfo = "skip"
        )

        return trace

    def make_timeseries_plot(self, upper_bound, lower_bound, show = False, show_uncertainty = True):
        fig = go.Figure()
        for region in self.regions:
            color = Color().region_colors[region]
            for scenario in self.scenarios:
                group = "{} {}".format(region, scenario)
                marker = Color().scenario_markers[scenario]
                data = self.df[self.df["Region"].isin([region]) & self.df["Scenario"].isin([scenario])]
                median = self.median_trace(data, group, region, scenario, color = color, marker = marker)
                if show_uncertainty:
                    lower = self.lower_bound_trace(data, group, lower_bound, color = color, marker = marker)
                    upper = self.upper_bound_trace(data, group, upper_bound, color = color, marker = marker)

                    fig.add_trace(lower)
                    fig.add_trace(upper)
                
                fig.add_trace(median)

        fig.update_layout(
            yaxis_title = '{}'.format(self.output),
            hovermode = "x",
            title = '{}'.format(Readability().readability_dict_forward[self.output])
        )

        if show:
            fig.show()

        return fig
    
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
    # df = SQLConnection("jp_data").output_df("total_emissions_CO2_million_ton_CO2", ["USA", "EUR", "CHN"], ["Ref", "2C"])
    # fig = TimeSeries("total_emissions_CO2_million_ton_CO2", 2050, df).make_plot()
    # fig.write_image("assets\examples\carbonemissions.svg")

    # input dist
    fig = InputDistribution(["WindGas", "wind", "BioCCS", "gas", "oil", "coal"]).make_plot("WindGas")
    fig.show()
    # fig.write_image("assets\examples\inputs_windgas_focus.svg")

    # input-output mapping
    # df = SQLConnection("jp_data").input_output_mapping_df("elec_prod_Renewables_TWh", "USA", "2C", 2050)
    # fig = InputOutputMappingPlot("total_emissions_CO2_million_ton_CO2", df).make_plot()
    # fig.write_image("assets\examples\cart_usa_2c_2050.svg")
