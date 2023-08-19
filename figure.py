import plotly.graph_objects as go
import plotly.express as px
from processing import LoadData
import numpy as np
from styling import Color
import pandas as pd
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

class Bar:
    pass

class TimeSeries:
    def __init__(self, type, output, regions, scenarios, years) -> None:
        self.allowed_types = ["type1", "type2"] # type 1: for a given output, select region and select multiple scenarios
        # type 2: for a given output, select a scenario and select multiple regions
        if type in self.allowed_types:
            self.type = type
        else:
            raise ValueError("Only type1 and type2 are allowed arguments for type")
        self.output = output
        self.regions = regions
        self.scenarios = scenarios
        self.years = years

    def create_type1_timeseries(self):
        dataframe = LoadData(output_name = self.output, region = self.regions, scenario = self.scenarios, year = self.years).csv_to_dataframe()

        tenth_df = dataframe.pivot_table("Value", index = ["Region"], columns = ["Year"], aggfunc = lambda x: np.percentile(x, 10))
        median_df = dataframe.pivot_table("Value", index = ["Region"], columns = ["Year"], aggfunc = np.median)
        ninetieth_df = dataframe.pivot_table("Value", index = ["Region"], columns = ["Year"], aggfunc = lambda x: np.percentile(x, 90))

        fig = go.Figure()
        for i, region in enumerate([self.regions]):
            color = Color().timeseries_colors[i]
            fig.add_scatter(x = tenth_df.columns, y = tenth_df.loc[region], mode = "lines", line_color = color)
            fig.add_scatter(x = median_df.columns, y = median_df.loc[region], mode = "lines", fill = "tonexty", line_color = color)
            fig.add_scatter(x = ninetieth_df.columns, y = ninetieth_df.loc[region], mode = "lines", fill = "tonexty", line_color = color)
        fig.layout.title = "10th/90th Percentiles and Median for {}".format(self.output)
        fig.layout.xaxis.title = "Year"
        fig.layout.yaxis.title = "{}".format(self.output)

        return fig
    # for a given output, select region and select multiple scenarios
    # for a given output, select a scenario and select multiple regions


class InputDistributions:
    def __init__(self, inputs):
        test_inputs = np.random.normal(loc = 0, scale = 1, size = (400, 50))
        column_names = [f"Column {i}" for i in range(1, 51)]
        testing_df = pd.DataFrame(test_inputs, columns = column_names)
        self.testing_df = testing_df
        self.inputs = inputs

    def create_input_distribution(self):
        data = self.testing_df[self.inputs]
        fig = make_subplots(rows = 1, cols = len(self.inputs))
        for i in range(len(self.inputs)):
            fig.add_trace(go.Violin(x = data[self.inputs[i]], name = self.inputs[0]), row = 1, col = i + 1)

        fig.update_traces(orientation = 'h', side = 'positive', width = 3, points = False)

        return fig

if __name__ == "__main__":

    fig = InputDistributions(["Column 1", "Column 2", "Column 3"]).create_input_distribution()
    fig.show()