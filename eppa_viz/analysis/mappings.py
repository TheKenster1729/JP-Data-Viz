"""Machine-learning mappings between inputs, outputs, and constraints."""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from tslearn.clustering import TimeSeriesKMeans

from eppa_viz.analysis.constants import RANDOM_STATE, RUNS_TO_DROP_BY_OUTPUT
from eppa_viz.analysis.corpus import matrix_for_runs, option_output_list
from eppa_viz.analysis.inputs import prepare_inputs
from eppa_viz.analysis.targets import discrete_from_percentile


def _align_runs_on_output(inputs_df, output_df):
    """Intersect input and output runs after custom-variable drops."""
    output_runs = set(output_df["Run #"].values)
    input_runs = set(inputs_df["Run #"].values)
    keep = output_runs & input_runs
    y_continuous = output_df[output_df["Run #"].isin(keep)]["Value"]
    inputs = inputs_df[inputs_df["Run #"].isin(keep)]
    return inputs, y_continuous


def _forest_importances(fit_model, columns, num_to_plot):
    feature_importances = pd.DataFrame(
        [estimator.feature_importances_ for estimator in fit_model.estimators_],
        columns=columns)
    sorted_labeled = feature_importances.mean().sort_values(ascending=False)
    top_n = sorted_labeled.index[:num_to_plot].to_list()
    return feature_importances, sorted_labeled, top_n


class InputOutputMapping:
    def __init__(self, output, region, scenario, year, df, threshold=70, gt=True,
                 num_to_plot=5, cart_depth=4, n_estimators=100, max_depth=4,
                 random_state=RANDOM_STATE):
        self.output = output
        self.df = df
        self.scenario = scenario
        self.year = year
        self.y_continuous = self.df["Value"]
        self.threshold = threshold
        self.gt = gt
        self.num_to_plot = num_to_plot
        self.cart_depth = cart_depth
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.region = region
        self.inputs = prepare_inputs(
            region, scenario, output=output, drop_runs_map=RUNS_TO_DROP_BY_OUTPUT)

    def preprocess_for_classification(self):
        try:
            assert len(self.inputs) == len(self.y_continuous)
        except AssertionError:
            self.inputs, self.y_continuous = _align_runs_on_output(
                self.inputs, self.df)

        X = self.inputs[self.inputs.columns[1:]]
        y_discrete = discrete_from_percentile(
            self.y_continuous.to_numpy(), self.threshold, self.gt)
        return X, y_discrete

    def CART(self):
        X, y = self.preprocess_for_classification()
        fit_model = DecisionTreeClassifier(
            max_depth=self.cart_depth, random_state=self.random_state)
        fit_model.fit(X, y)
        return fit_model

    def random_forest(self, n_jobs=1):
        X, y = self.preprocess_for_classification()
        fit_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=n_jobs,
            random_state=self.random_state).fit(X, y)
        return _forest_importances(fit_model, X.columns, self.num_to_plot)

    def permutation_importance(self):
        X, y = self.preprocess_for_classification()
        fit_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state).fit(X, y)
        results = permutation_importance(
            fit_model, X, y, n_repeats=10, random_state=self.random_state)
        important = []
        for i in results.importances_mean.argsort()[::-1]:
            if results.importances_mean[i] - 3 * results.importances_std[i] > 0:
                important.append({
                    "variable": X.columns[i],
                    "mean": results.importances_mean[i],
                    "std": results.importances_std[i],
                })
        return important


class OutputOutputMapping:
    def __init__(self, db_obj, output, region, scenario, year, df, threshold=70,
                 gt=True, num_to_plot=5, other_outputs=[], max_workers=8,
                 random_state=RANDOM_STATE):
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
        self.max_workers = max_workers  # kept for API compatibility; unused
        self.random_state = random_state

    def preprocess_for_classification(self):
        if not len(self.df) > 350:
            return "insufficient length (< 350)"
        if not len(self.df.query("Value==0")) < 175:
            return "too many (> half) zero values"

        target_output_run_numbers = self.df["Run #"]
        list_to_use = option_output_list(self.db_obj.dbname) + self.other_outputs
        matrix = matrix_for_runs(
            self.db_obj.repository,
            list_to_use,
            self.region,
            self.scenario,
            self.year,
            target_output_run_numbers,
            exclude_output=self.output,
        )
        if matrix is None or matrix.empty:
            return "insufficient length (< 350)"

        self.main_df = matrix.reset_index(drop=True)
        y_discrete = discrete_from_percentile(
            self.y_continuous.to_numpy(), self.threshold, self.gt)
        return y_discrete

    def random_forest(self):
        y = self.preprocess_for_classification()
        if isinstance(y, str):
            return y
        fit_model = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state).fit(self.main_df, y)
        return _forest_importances(fit_model, self.main_df.columns, self.num_to_plot)

    @classmethod
    def clear_cache(cls):
        """No-op: table-name cache removed when reads moved to series_values."""


class FilteredInputOutputMapping:
    def __init__(self, constraint_df, region, scenario, year, num_to_plot=5,
                 cart_depth=4, n_estimators=100, random_forest_depth=4,
                 random_state=RANDOM_STATE):
        self.constraint_df = constraint_df
        self.region = region
        self.scenario = scenario
        self.year = year
        self.inputs = prepare_inputs(region, scenario)
        self.num_to_plot = num_to_plot
        self.cart_depth = cart_depth
        self.n_estimators = n_estimators
        self.random_forest_depth = random_forest_depth
        self.random_state = random_state

    def preprocess_for_classification(self):
        self.X = self.inputs[self.inputs.columns[1:]]
        self.y_discrete = self.constraint_df["in_constraint_range"]
        assert len(self.X) == len(self.y_discrete)

    def CART(self):
        self.preprocess_for_classification()
        fit_model = DecisionTreeClassifier(
            max_depth=self.cart_depth, random_state=self.random_state)
        fit_model.fit(self.X, self.y_discrete)
        return fit_model

    def random_forest(self):
        self.preprocess_for_classification()
        fit_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.random_forest_depth,
            random_state=self.random_state).fit(self.X, self.y_discrete)
        return _forest_importances(fit_model, self.X.columns, self.num_to_plot)

    def permutation_importance(self):
        self.preprocess_for_classification()
        fit_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.random_forest_depth,
            random_state=self.random_state).fit(self.X, self.y_discrete)
        results = permutation_importance(
            fit_model, self.X, self.y_discrete, n_repeats=10,
            random_state=self.random_state)
        important = []
        for i in results.importances_mean.argsort()[::-1]:
            if results.importances_mean[i] - 3 * results.importances_std[i] > 0:
                important.append({
                    "variable": self.X.columns[i],
                    "mean": results.importances_mean[i],
                    "std": results.importances_std[i],
                })
        return important


class FilteredOutputOutputMapping:
    def __init__(self, db_obj, constraint_df, region, scenario, year,
                 num_to_plot=5, max_workers=8, random_state=RANDOM_STATE):
        self.db_obj = db_obj
        self.constraint_df = constraint_df
        self.region = region
        self.scenario = scenario
        self.year = year
        self.num_to_plot = num_to_plot
        self.max_workers = max_workers  # kept for API compatibility; unused
        self.random_state = random_state
        self.run_numbers = constraint_df["Run #"].values
        self.in_constraint_range = constraint_df["in_constraint_range"].values

    def create_dataframe(self):
        options_to_use = option_output_list(self.db_obj.dbname)
        matrix = matrix_for_runs(
            self.db_obj.repository,
            options_to_use,
            self.region,
            self.scenario,
            self.year,
            self.run_numbers,
            display_names=True,
        )
        if matrix is None or matrix.empty:
            self.df_to_use = pd.DataFrame()
        else:
            self.df_to_use = matrix.reindex(sorted(self.run_numbers))

    def run_analysis(self):
        self.create_dataframe()
        if self.df_to_use.empty:
            return None, []
        self.df_to_use["in_constraint_range"] = self.in_constraint_range
        self.df_to_use.dropna(how="any", inplace=True)

        X = self.df_to_use[self.df_to_use.columns[:-1]]
        y = self.df_to_use["in_constraint_range"]

        random_forest = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=self.random_state).fit(X, y)
        feature_importances = pd.DataFrame(
            [estimator.feature_importances_ for estimator in random_forest.estimators_],
            columns=X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(
            ascending=False)
        top_n = sorted_labeled_importances.index[:self.num_to_plot].to_list()
        return sorted_labeled_importances, top_n

    @classmethod
    def clear_cache(cls):
        """No-op: table-name cache removed when reads moved to series_values."""


class TimeSeriesClustering:
    def __init__(self, df, output, region, scenario, n_clusters=3,
                 metric="euclidean", num_to_plot=5, cart_depth=4,
                 n_estimators=100, max_depth=4, random_state=RANDOM_STATE):
        self.df = df
        self.output = output
        self.region = region
        self.scenario = scenario
        self.n_clusters = n_clusters
        self.df_for_clustering = self.df.pivot(columns="Year", index="Run #")
        self.metric = metric
        self.cart_depth = cart_depth
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_to_plot = num_to_plot
        self.random_state = random_state

    def generate_clusters(self):
        return TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric=self.metric,
            random_state=self.random_state).fit(self.df_for_clustering)

    def cluster_mapping(self):
        self.inputs = prepare_inputs(
            self.region, self.scenario, output=self.output,
            drop_runs_map=RUNS_TO_DROP_BY_OUTPUT)
        self.X = self.inputs[self.inputs.columns[1:]]
        self.y = self.generate_clusters().labels_
        fit_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state).fit(self.X, self.y)
        return _forest_importances(fit_model, self.X.columns, self.num_to_plot)
