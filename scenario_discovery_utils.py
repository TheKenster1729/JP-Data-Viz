import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from textwrap import wrap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import product
from seaborn import heatmap
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn import tree
import graphviz
import os
import plotly
import plotly.express as px
import string
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import host_subplot
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from colour import Color
import openpyxl

class GlobalProperties:
    def __init__(self):
        self.supported_input_scenarios = ['GLB_GRP_NORM', 'GLB_RAW', 'USA', 'CHN', 'EUR', 'CHN_ENDOG_RENEW', 'CHN_ENDOG_EMISSIONS']
        self.supported_output_scenarios = ['REF_GLB_RENEW_SHARE', 'REF_USA_RENEW_SHARE', 'REF_CHN_RENEW_SHARE', 'REF_EUR_RENEW_SHARE',
            '2C_GLB_RENEW_SHARE', '2C_USA_RENEW_SHARE', '2C_CHN_RENEW_SHARE', '2C_EUR_RENEW_SHARE', 'REF_GLB_RENEW', 'REF_GLB_TOT', '2C_GLB_RENEW', '2C_GLB_TOT',
            '2C_CHN_ENDOG_RENEW', 'REF_CHN_EMISSIONS']
        self.colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']

class Preparation(GlobalProperties):
    def __init__(self, output_case):
        super().__init__()
        self.output_case = output_case
        self.input_df = pd.read_excel("Full Data for Paper3+USAcons2050.xlsx", "samples", header = 2, usecols = "A:AZ, BC:BF")
        if self.output_case == 'pes':
            self.output_df = pd.read_excel("Full Data for Paper3+USAcons2050.xlsx", "A1.5C_pes_USA_cons", usecols = "I:X")
            na_rows = self.output_df.notna().all(axis = 1)
            self.input_df = self.input_df[na_rows]
            self.output_df.dropna(inplace = True)
        elif self.output_case == 'opt':
            self.output_df = pd.read_excel("Full Data for Paper3+USAcons2050.xlsx", "A1.5C_opt_USA_cons", usecols = "I:X")

    def get_X(self):
        return self.input_df[self.input_df.columns[1:]]

    def get_y(self):
        return self.output_df

    def get_y_by_year(self, year):
        return self.output_df[str(year)]

# class Preparation(GlobalProperties):
#     def __init__(self, input_case, output_case):
#         super().__init__()
#         """Creates an object that stores the input and output dataframes that contain simulation results. These objects have a number of
#         methods that are helpful for scenario discovery analysis. Be careful not to mix incompatible input/output cases together (e.g.
#         using U.S. input data to analyze share of renewables in China).

#         Args:
#             input_case (str): the input data to be used (supported: GLB_GRP_NORM (global, grouped, and normed); GLB_RAW
#             (global, full number of variables, no normalization); USA (full number of variables with GDP + Pop specific to US);
#             CHN (full number of variables with GDP + Pop specific to China); EUR (full number of variables with GDP + Pop
#             specific to EU); CHN_ENDOG_RENEW (output-output mapping as shown in paper section 4.1); CHN_ENDOG_EMISSIONS (output-output
#             mapping as shown in paper section 4.2))

#             output_case (str): the output metric (supported: REF_GLB_RENEW_SHARE (global share of renewables under
#             the reference scenario); REF_USA_RENEW_SHARE (US share of renewables under the reference scenario); REF_CHN_RENEW_SHARE
#             (China share of renewables under the reference scenario); REF_EUR_RENEW_SHARE (EU share of renewables under the reference
#             scenario); 2C_GLB_RENEW_SHARE (global share of renewables under the policy scenario); 2C_USA_RENEW_SHARE (US share of
#             renewables under the policy scenario); 2C_CHN_RENEW_SHARE (China share of renewables under the policy scenario);
#             2C_EUR_RENEW_SHARE (EU share of renewables under the policy scenario); REF_GLB_RENEW (global renewable energy
#             production in Twh); REF_GLB_TOT (total global energy production in Twh); 2C_GLB_RENEW (global renewable energy production
#             in Twh under policy); 2C_GLB_TOT (total global energy production in Twh under policy); 2C_CHN_ENDOG_RENEW (output-output 
#             mapping as shown in paper section 4.1); REF_CHN_EMISSIONS (output-output mapping as shown in paper section 4.1); CHN_ENDOG_EMISSIONS 
#             (output-output mapping as shown in paper section 4.2))

#         Raises:
#             ValueError: if an invalid input case is passed
#             ValueError: if an invalid output case is passed
#         """
#         self.input_case = input_case
#         self.output_case = output_case
#         self.hyperparams = None
#         # "translates" between the abbreviations used in the code and the long forms used in the paper
#         self.readability_dict = {'GLB_RAW': 'Global', 'REF': 'Share of Renewables Under Reference', 'POL': 'Share of Renewables Under Policy',
#                                 'CHN': 'China', 'USA': 'USA', 'EUR': 'Europe'}
#         self.ref_or_pol = 'REF' if 'REF' in self.output_case else 'POL'

#         self.natural_to_code_conversions_dict_inputs = {'GLB_GRP_NORM': ['samples-norm+groupedav', 'A:J'], 'GLB_RAW': ['samples', 'A:BB'], 'USA': ['samples', 'A:AZ, BC:BF'],
#             'CHN': ['samples', 'A:AZ, BG:BJ'], 'EUR': ['samples', 'A:AZ, BK: BN'], 'CHN_ENDOG_RENEW': ['2C_CHN_renew_outputs_inputs', 'A:G'],
#             'CHN_ENDOG_EMISSIONS': ['REF_CHN_emissions_inputs', 'A:H']}
#         if self.input_case in self.supported_input_scenarios:
#             sheetname = self.natural_to_code_conversions_dict_inputs[self.input_case][0]
#             columns = self.natural_to_code_conversions_dict_inputs[self.input_case][1]
#             if self.input_case == "GLB_GRP_NORM":
#                 self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400, engine = 'openpyxl')
#             else:
#                 self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400, header = 2, engine = 'openpyxl')
            
#             if self.input_case == 'CHN_ENDOG_RENEW':
#                 # these scenarios already have the crashed runs removed
#                 pass
#             else:
#                 if '2C' in output_case:
#                     # indicates a policy scenario in which runs crashed
#                     crashed_run_numbers = [3, 14, 74, 116, 130, 337]
#                     self.input_df = self.input_df.drop(index = [i - 1 for i in crashed_run_numbers])
#         else:
#             raise ValueError('This input scenario is not supported. Supported scenarios are {}'.format(self.supported_input_scenarios))

#         self.natural_to_code_conversions_dict_outputs = {'REF_GLB_RENEW_SHARE': 'ref_GLB_renew_share', 'REF_USA_RENEW_SHARE': 'ref_USA_renew_share',
#             'REF_CHN_RENEW_SHARE': 'ref_CHN_renew_share', 'REF_EUR_RENEW_SHARE': 'ref_EUR_renew_share', '2C_GLB_RENEW_SHARE': '2C_GLB_renew_share',
#             '2C_USA_RENEW_SHARE': '2C_USA_renew_share', '2C_CHN_RENEW_SHARE': '2C_CHN_renew_share', '2C_EUR_RENEW_SHARE': '2C_EUR_renew_share',
#             'REF_GLB_RENEW': 'ref_GLB_renew', 'REF_GLB_TOT': 'ref_GLB_total_elec', '2C_GLB_RENEW': '2C_GLB_renew', '2C_GLB_TOT': '2C_GLB_total_elec',
#             '2C_CHN_ENDOG_RENEW': '2C_CHN_renew_outputs_output', 'REF_CHN_EMISSIONS': 'REF_CHN_emissions_output'}
#         if self.output_case in self.supported_output_scenarios:
#             sheetname = self.natural_to_code_conversions_dict_outputs[output_case]
#             self.output_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = 'D:X', nrows = 400, engine = 'openpyxl')
#             if self.output_case == '2C_CHN_ENDOG_RENEW' or self.output_case == 'REF_CHN_EMISSIONS':
#                 self.output_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = 'A:B', nrows = 400, engine = 'openpyxl')
#                 print('Note: some methods are not supported for this output scenario because it contains data from only one year of the simulation (2050).')
#             else:
#                 if '2C' in output_case:
#                     # indicates a policy scenario in which runs crashed
#                     crashed_run_numbers = [3, 14, 74, 116, 130, 337]
#                     self.output_df = self.output_df.drop(index = [i - 1 for i in crashed_run_numbers])
#         else:
#             raise ValueError('This output scenario is not supported. Supported scenarios are {}'.format(self.supported_output_scenarios))

#     def get_X(self, runs = False):
#         """Get the exogenous dataset (does not include run numbers).

#         Returns:
#             DataFrame: Input variables and their values
#         """
#         if runs:
#             return self.input_df
#         else:
#             return self.input_df[self.input_df.columns[1:]]

#     def get_y(self, runs = False):
#         """Get the endogenous dataset (does not include run numbers).

#         Returns:
#             DataFrame: Output timeseries
#         """
#         if runs:
#             return self.output_df
#         else:
#             return self.output_df[self.output_df.columns[1:]]

#     def get_y_by_year(self, year):
#         """Get the series for an individual year.

#         Args:
#             year (int): A year included in the dataset (options:
#             2007, and 2010-2100 in 5-year increments)

#         Returns:
#             Series: A pandas Series object with the data from the given year
#         """
#         return self.output_df[str(year)]

class Analysis(GlobalProperties):
    def __init__(self, output_case):
        super().__init__()
        self.sd_obj = Preparation(output_case)

    def rfcMostImportantInputs(self, percentile, gt, num_to_plot):
        X = self.sd_obj.get_X()
        y = self.sd_obj.get_y_by_year(2050)

        perc_val = np.percentile(y, percentile)

        if gt:
            y_discrete = np.where(y > perc_val, 1, 0)
        else:
            y_discrete = np.where(y < perc_val, 1, 0)

        clf = RandomForestClassifier(n_estimators = 200, min_samples_split = 8,
                        max_depth = 6, random_state = 0)
        model_fit = clf.fit(X, y_discrete)

        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in model_fit.estimators_], columns = X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n

class Visualization(GlobalProperties):
    def __init__(self, plot_type, sd_obj, display = True):
        super().__init__()
        self.plot_type = plot_type
        self.display = display
        self.sd_obj = sd_obj
        self.plot_functions_map = {"Distribution": self.distribution, "Parallel Plot": self.parallelPlot, "Bar": self.barForRFCMostImportant}

    def makePlot(self, *args):
        return self.plot_functions_map[self.plot_type](*args)

    def distribution(self, inputs_to_visualize = None):
        # visualize inputs as a histogram
        if not inputs_to_visualize:
            inputs_to_visualize = self.sd_obj.get_X().columns
        input_fig = make_subplots(rows = 1, cols = len(inputs_to_visualize))
        for i, input in enumerate(inputs_to_visualize):
            trace = go.Histogram(x = self.sd_obj.get_X()[input], name = input)
            input_fig.add_trace(trace, row = 1, col = i + 1)
        input_fig.update_layout(
            title = "Distribution for Inputs " + ', '.join(inputs_to_visualize) + ' for ' + self.sd_obj.output_case
        )

        # visualize outputs as time series
        output_results = self.sd_obj.get_y()
        output_plot_df = pd.DataFrame(index = output_results.columns, columns = ["95th", "Median", "5th"])
        output_plot_df["95th"] = output_results.apply(lambda x: np.percentile(x, 95))
        output_plot_df["5th"] = output_results.apply(lambda x: np.percentile(x, 5))
        output_plot_df["Median"] = output_results.apply(lambda x: np.percentile(x, 50))

        output_fig = go.Figure([
            go.Scatter(
                name = 'Median',
                x = output_plot_df.index,
                y = output_plot_df['Median'],
                mode = 'lines',
                line = dict(color = self.colors[0]),
            ),
            go.Scatter(
                name = '95th Percentile',
                x = output_plot_df.index,
                y = output_plot_df['95th'],
                mode = 'lines',
                marker = dict(color = self.colors[1]),
                line = dict(width = 0),
                showlegend = False
            ),
            go.Scatter(
                name = '5th Percentile',
                x = output_plot_df.index,
                y = output_plot_df['5th'],
                marker = dict(color = self.colors[1]),
                line = dict(width = 0),
                mode = 'lines',
                # inelegant method to get a more transparent color for fill
                fillcolor = "rgba" + str((*Color(self.colors[1]).rgb, 0.2)),
                fill = 'tonexty',
                showlegend = False
            )
        ])
        output_fig.update_layout(
            yaxis_title = self.sd_obj.output_case,
            title = self.plot_type + " for " + self.sd_obj.output_case,
            hovermode = "x"
        )

        return input_fig, output_fig

    def parallelPlot(self, inputs_to_visualize, year_for_output, percentile, condition):
        # a few lines of code to grab the data type from the output name
        # it turns out that the word after the last underscore of the spreadsheet name is a unique identifier of the case:
        # e.g., if this word is 'RENEW', then the output case is total renewable production
        unique_identifier = self.sd_obj.output_case.split('_')[-1]
        unique_identifier_to_natural_lang_conversion_dict = {"SHARE": "Share", "RENEW": "Renewable Energy Production (TWh)", "TOT": "Total Elecricity Production (TWh)"}
        case_type = unique_identifier_to_natural_lang_conversion_dict[unique_identifier]

        y = self.sd_obj.get_y_by_year(str(year_for_output)).to_numpy()
        y_col_name = case_type + ' ' + str(year_for_output)
        inputs = self.sd_obj.get_X()[inputs_to_visualize]

        # establish target based on condition (i.e., greater than or less than percentile)
        percentile_val = np.percentile(y, percentile)
        if condition == "gt":
            target = np.where(y >= percentile_val, 1, 0).ravel()
            target_name = ["< Percentile", ">= Percentile"]
        elif condition == "lt":
            target = np.where(y <= percentile_val, 1, 0).ravel()
            target_name = ["> Percentile", "<= Percentile"]
        else:
            raise ValueError("Invalid condition. Supported conditions are 'gt' (greater than percentile) and 'lt' (less than percentile).") 

        # pd.concat is annoying, so using a dirtier way of creating the dataframe for the parallel plot
        parallel_plot_df = inputs.copy()
        parallel_plot_df[y_col_name] = y
        parallel_plot_df["Target"] = target
        parallel_plot_df = parallel_plot_df.sort_values(by = "Target")
        
        parallel_plot = px.parallel_coordinates(parallel_plot_df, color = parallel_plot_df.columns[-1], color_continuous_scale = 
                                                [(0, self.colors[0]), (0.5, self.colors[0]), (0.5, self.colors[3]), (1, self.colors[3])], dimensions = parallel_plot_df.columns[:-1])
        parallel_plot.update_layout(coloraxis_colorbar = dict(
            title = "Target",
            tickvals = [0.25, 0.75],
            ticktext = target_name,
            lenmode = 'pixels', len = 100, yanchor = 'middle'), 
            title = "Parallel Plot for " + ', '.join(inputs_to_visualize) + ' ' + y_col_name)

        parallel_plot.show()

    def barForRFCMostImportant(self, output_case, percentile, gt = True, num_to_plot = 4, savefig = False):
        fig = make_subplots(rows = 2, cols = 2)
        cases = [("pes", 90, True, 1, 1, "Pes >90th Perc"), ("pes", 10, False, 1, 2, "Pes <10th Perc"), ("opt", 90, True, 2, 1, "Opt >90th Perc"), ("opt", 10, False, 2, 2, "Opt <10th Perc")]
        for case in cases:
            feature_importances, sorted_labeled_importances, top_n = Analysis(case[0]).rfcMostImportantInputs(case[1], case[2], num_to_plot)
            fig.add_trace(
                go.Bar(x = top_n,
                       y = sorted_labeled_importances[top_n].values,
                       name = case[5]), row = case[3], col = case[4]
            )
        if savefig:
            fig.write_image("pes_opt_fig.png", "png", scale = 2)

class Display(GlobalProperties):
    def __init__(self, display = True):
        super().__init__()
        if display:
            with st.sidebar:
                st.header("Input and Ouput Data Info")
                st.subheader("Inputs")
                st.markdown("**GLB_RAW**: All of the indepedent (exogenous) variables, sampled independently from distributions specific to each variable\
                        and without any pre-processing steps (e.g. normalization) applied.")

    def displayCorrectOutputs(self, selected_input):
        region = selected_input.split('_')[0]
        output_options = [x for x in self.supported_output_scenarios if region in x]

        return output_options

    def displayDataMenu(self):
        st.write("Please choose the input and output data you'd like to analyze.")
        input_data = st.selectbox("Please select the input data (see sidebar for descriptions).",
                    (x for x in self.supported_input_scenarios))
        
        output_data = st.selectbox("Please select the output data (see sidebar for descriptions).",
                self.displayCorrectOutputs(input_data))

        return input_data, output_data

    def displayPlotsMenu(self):
        st.write("Please choose the type of plot you wish to generate.")
        plot = st.selectbox("Plots", ["Parallel Plot", "Top Features Plot", "Distribution"])

        return plot

    def makeSelectedPlot(self, plot_type, input_data, output_data):
        sd_obj = Preparation(input_data, output_data)
        if plot_type == "Distribution":
            with st.container():
                st.write("Select which inputs you'd like to visualize (max 5). Output data will be presented automatically.")
                inputs_to_visualize = st.multiselect("Inputs", sd_obj.get_X().columns, max_selections = 6)

                confirm = st.button("Confirm")
            
                if confirm:
                    input_fig, output_fig = Visualization(plot_type, sd_obj).makePlot(inputs_to_visualize)
                    st.plotly_chart(input_fig, use_container_width = True)
                    st.plotly_chart(output_fig, use_container_width = True)

        if plot_type == "Parallel Plot":
            with st.container():
                st.write("Select which inputs you'd like to visualize on the parallel plot (max 5).")
                inputs_to_visualize = st.multiselect("Inputs", sd_obj.get_X().columns, max_selections = 6)
                output_to_visualize = st.radio("Select which output you'd like to visualize on the parallel plot.", 
                                               ["Renewable Share", "Total Renewable Electricity Production", "Total Electricity Production"])


    def run(self):
        with st.container():
            input_data, output_data = self.displayDataMenu()
            selected_plot = self.displayPlotsMenu()

            self.makeSelectedPlot(selected_plot, input_data, output_data)

# suggested workflow
# 1. pass in region, policy to work with
    # issue here: comparison across different scenario types
        # can there be an option to grid the layout and choose which plot can go in which grid?

if __name__ == "__main__":
    pd.options.plotting.backend = "plotly"
    # st.set_page_config(layout = 'wide')

    # elements = Display().run()
    # Visualization("Parallel Plot", Preparation("GLB_RAW", "2C_GLB_TOT")).makePlot(["VINT", "wind", "Bio"], 2050, 70, "gt")
    Visualization("Bar", None).barForRFCMostImportant("pes", 90)
