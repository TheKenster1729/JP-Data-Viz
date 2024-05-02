import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sql_utils import SQLConnection, DataRetrieval
from styling import Readability, Options
from tslearn.clustering import TimeSeriesKMeans
from sklearn.inspection import permutation_importance

class InputOutputMapping:
    def __init__(self, output, region, scenario, year, df, threshold = 70, gt = True, num_to_plot = 5, cart_depth = 4, n_estimators = 100, max_depth = 4):
        self.output = output
        self.df = df
        self.scenario = scenario
        self.year = year
        self.y_continuous = self.df["Value"]
        self.threshold = threshold
        self.gt = gt
        self.inputs = pd.read_csv(r"Cleaned Data/InputsMaster.csv")
        self.num_to_plot = num_to_plot
        self.cart_depth = cart_depth
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # need to remove input pop/gdp not relevant to this region
        self.region = region
        columns_to_remove = []
        for column in self.inputs.columns:
            signifiers = [" GDP", "Non-{} GDP".format(self.region), " Pop", "Non-{} Pop".format(self.region)]
            if any(signifier in column for signifier in signifiers) and self.region not in column:
                columns_to_remove.append(column)
        self.inputs = self.inputs.drop(columns = columns_to_remove)

        # some scenarios have runs that didn't solve in all cases, so remove those as well
        runs_to_drop_dict = {"percapita_consumption_loss_percent":
                             {"About15C_pes": [82, 98, 283, 305, 338, 373],
                             "15C_med": [184, 221, 314, 374, 383]}
                             }
        drop_runs = runs_to_drop_dict.get(self.output)
        if drop_runs:
            runs_to_drop_for_scenario = drop_runs.get(self.scenario)
            if runs_to_drop_for_scenario:
                self.inputs = self.inputs.drop(self.inputs[self.inputs["Run #"].isin(runs_to_drop_for_scenario)].index)

    def preprocess_for_classification(self):

        try:
            assert len(self.inputs) == len(self.y_continuous)
        except AssertionError:
            # will happen when runs have been removed, e.g. because of creating a custom variable
            # that produced a division by 0 error
            # infer missing run numbers and remove them
            existing_output_run_numbers_set = set(self.df["Run #"].values)
            existing_input_run_numbers_set = set(self.inputs["Run #"].values)
            inputs_to_keep = existing_input_run_numbers_set.intersection(existing_output_run_numbers_set)
            self.y_continuous = self.df[self.df["Run #"].isin(inputs_to_keep)]["Value"]
            self.inputs = self.inputs[self.inputs["Run #"].isin(inputs_to_keep)]

        X = self.inputs[self.inputs.columns[1:]]

        percentile = np.percentile(self.y_continuous, self.threshold)
        if self.gt:
            y_discrete = np.where(self.y_continuous.to_numpy() > percentile, 1, 0)
        else:
            y_discrete = np.where(self.y_continuous.to_numpy() < percentile, 1, 0)


        return X, y_discrete

    def CART(self):
        X, y = self.preprocess_for_classification()
        fit_model = DecisionTreeClassifier(max_depth = self.cart_depth)
        fit_model.fit(X, y)

        return fit_model

    def random_forest(self):
        X, y = self.preprocess_for_classification()
        fit_model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth).fit(X, y)

        # get the average feature importances
        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in fit_model.estimators_], columns = X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:self.num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n
    
    def permutation_importance(self):
        X, y = self.preprocess_for_classification()

        fit_model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth).fit(X, y)
        permutation_importance_results = permutation_importance(fit_model, X, y, n_repeats = 10)

        important = []
        for i in permutation_importance_results.importances_mean.argsort()[::-1]:
            if permutation_importance_results.importances_mean[i] - 3*permutation_importance_results.importances_std[i] > 0:
                important.append({"variable": X.columns[i], "mean": permutation_importance_results.importances_mean[i], "std": permutation_importance_results.importances_std[i]})

        return important

class OutputOutputMapping:
    def __init__(self, db_obj, output, region, scenario, year, df, threshold = 70, gt = True, num_to_plot = 5, other_outputs = []):
        self.db_obj = db_obj
        self.output = output
        self.region = region
        self.scenario = scenario
        self.year = year
        self.df = df
        self.y_continuous = self.df["Value"]
        self.threshold = threshold
        self.gt = gt
        self.num_to_plot = num_to_plot
        self.other_outputs = other_outputs

    def preprocess_for_classification(self):
        # not all runs have solved in all regions and all scenarios
        # we need criteria that outputs must meet in order to be considered by the model
        # we also need to make sure run numbers are consistent

        # ensure the selected output is >350 data points
        # and that the majority of these are not 0
        if not len(self.df) > 350 :
            return "insufficient length (< 350)"
        
        if not len(self.df.query("Value==0")) < 175:
            return "too many (> half) zero values"
        
        # get the initial set of run numbers to use as a first filter for subsequent outputs
        target_output_run_numbers = self.df["Run #"]

        # create the X data frame for CART for this input/region/scenario/year combination
        main_df = pd.DataFrame(data = {"Run #": target_output_run_numbers})
        list_to_use = list(Options().outputs) + self.other_outputs
        list_to_use.remove(self.output)
        for output in list_to_use:
            df = DataRetrieval(self.db_obj, output, self.region, self.scenario, 2050).mapping_df()[["Run #", "Value"]]

            # only proceed if this new output includes at least all the solved runs of the target output
            target_runs_set = set(target_output_run_numbers.values)
            new_df_runs_set = set(df["Run #"].values)
            if target_runs_set.issubset(new_df_runs_set):
                # no need to verify length, we know it is at least 350 if target runs are a subset of new runs
                filtered_df = df[df["Run #"].isin(target_output_run_numbers)]["Value"].rename(output)
                main_df = pd.concat([main_df, filtered_df], axis = 1)
        
        percentile = np.percentile(self.y_continuous, self.threshold)
        if self.gt:
            y_discrete = np.where(self.y_continuous.to_numpy() > percentile, 1, 0)
        else:
            y_discrete = np.where(self.y_continuous.to_numpy() < percentile, 1, 0)
        
        self.main_df = main_df.drop(columns = "Run #")

        return y_discrete

    def random_forest(self):
        y = self.preprocess_for_classification()
        if type(y) is str:
            return y
        
        fit_model = RandomForestClassifier(n_estimators = 100).fit(self.main_df, y)

        # get the average feature importances
        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in fit_model.estimators_], columns = self.main_df.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:self.num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n

class FilteredInputOutputMapping:
    def __init__(self, constraint_df, region, scenario, year, num_to_plot = 5, cart_depth = 4, n_estimators = 100, random_forest_depth = 4):
        self.constraint_df = constraint_df
        self.region = region
        self.scenario = scenario
        self.year = year
        self.inputs = pd.read_csv(r"Cleaned Data/InputsMaster.csv")
        self.num_to_plot = num_to_plot
        self.cart_depth = cart_depth
        self.n_estimators = n_estimators
        self.random_forest_depth = random_forest_depth

        # need to remove input pop/gdp not relevant to this region
        columns_to_remove = []
        for column in self.inputs.columns:
            signifiers = [" GDP", "Non-{} GDP".format(self.region), " Pop", "Non-{} Pop".format(self.region)]
            if any(signifier in column for signifier in signifiers) and self.region not in column:
                columns_to_remove.append(column)
        self.inputs = self.inputs.drop(columns = columns_to_remove)

    def preprocess_for_classification(self):

        # try:
        #     assert len(self.inputs) == len(self.y_continuous)
        # except AssertionError:
        #     # will happen when runs have been removed, e.g. because of creating a custom variable
        #     # that produced a division by 0 error
        #     # infer missing run numbers and remove them
        #     existing_output_run_numbers_set = set(self.df["Run #"].values)
        #     existing_input_run_numbers_set = set(self.inputs["Run #"].values)
        #     inputs_to_keep = existing_input_run_numbers_set.intersection(existing_output_run_numbers_set)
        #     self.y_continuous = self.df[self.df["Run #"].isin(inputs_to_keep)]["Value"]
        #     self.inputs = self.inputs[self.inputs["Run #"].isin(inputs_to_keep)]

        self.X = self.inputs[self.inputs.columns[1:]]
        self.y_discrete = self.constraint_df["in_constraint_range"]

        assert len(self.X) == len(self.y_discrete) # need to write edge cases for this

    def CART(self):
        self.preprocess_for_classification()
        fit_model = DecisionTreeClassifier(max_depth = self.cart_depth)
        fit_model.fit(self.X, self.y_discrete)

        return fit_model

    def random_forest(self):
        self.preprocess_for_classification()
        fit_model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.random_forest_depth).fit(self.X, self.y_discrete)

        # get the average feature importances
        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in fit_model.estimators_], columns = self.X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:self.num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n

    def permutation_importance(self):
        X, y = self.preprocess_for_classification()

        fit_model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth).fit(X, y)
        permutation_importance_results = permutation_importance(fit_model, X, y, n_repeats = 10)

        important = []
        for i in permutation_importance_results.importances_mean.argsort()[::-1]:
            if permutation_importance_results.importances_mean[i] - 3*permutation_importance_results.importances_std[i] > 0:
                important.append({"variable": X.columns[i], "mean": permutation_importance_results.importances_mean[i], "std": permutation_importance_results.importances_std[i]})

        return important

class FilteredOutputOutputMapping:
    def __init__(self, db_obj, outputs_to_use, run_numbers, in_constraint_range, region, scenario, year, num_to_plot = 5):
        self.db_obj = db_obj
        self.outputs_to_use = outputs_to_use
        self.run_numbers = run_numbers
        self.in_constraint_range = in_constraint_range
        self.region = region
        self.scenario = scenario
        self.year = year
        self.num_to_plot = num_to_plot

    def create_dataframe(self):
        self.df_to_use = pd.DataFrame()
        # TODO: there could be less run numbers in the output,
        # need to handle this edge case
        for output in self.outputs_to_use:
            df = DataRetrieval(self.db_obj, output, self.region, self.scenario, self.year).mapping_df()
            temp_df = pd.DataFrame()
            temp_df[Readability().naming_dict_long_names_first[output]] = df["Value"]
            self.df_to_use = pd.concat([self.df_to_use, temp_df], axis = 1)

    def run_analysis(self):
        self.create_dataframe()        
        X = self.df_to_use
        y = self.in_constraint_range

        random_forest = RandomForestClassifier(n_estimators = 100).fit(X, y)
        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in random_forest.estimators_], columns = X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:self.num_to_plot].to_list()

        return sorted_labeled_importances, top_n

class TimeSeriesClustering:
    def __init__(self, df, output, region, scenario, n_clusters = 3, metric = "euclidean"):
        self.df = df
        self.output = output
        self.region = region
        self.scenario = scenario
        self.n_clusters = n_clusters
        self.df_for_clustering = self.df.pivot(columns = "Year", index = "Run #")
        self.metric = metric

    def generate_clusters(self):
        clusters = TimeSeriesKMeans(n_clusters = self.n_clusters, metric = self.metric).fit(self.df_for_clustering)

        return clusters
            

if __name__ == "__main__":
    db = SQLConnection("all_data_jan_2024")
    custom_output_example = "elec_prod_renewables_twh_pol-division-elec_prod_total_twh_pol-Renewable Share"
    
    # input-output-mapping
    # df = DataRetrieval(db, "emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref", 2050).input_output_mapping_df()
    # io = InputOutputMapping("emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref", 2050, df).random_forest()
    # print(io[-1])

    # time series clustering
    # time_series = TimeSeriesClustering(db, "emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref")
    # clusters = time_series.plot_clusters()

    # output/output mapping
    # df = DataRetrieval(db, "primary_energy_use_Biofuel_FirstGen_EJ", "GLB", "2C_med", 2050).mapping_df()
    # res = OutputOutputMapping("primary_energy_use_Biofuel_FirstGen_EJ", "GLB", "2C_med", 2050, df, other_outputs = [custom_output_example]).random_forest()
    # print(res[1])

    # permutation importances
    df = DataRetrieval(db, "emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref", 2050).mapping_df()
    io = InputOutputMapping("emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref", 2050, df).permutation_importance()
    print(io)

