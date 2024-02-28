import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sql_utils import SQLConnection, DataRetrieval
from styling import Readability

class InputOutputMapping:
    def __init__(self, output, region, scenario, year, df, threshold = 70, gt = True):
        self.output = output
        self.df = df
        self.scenario = scenario
        self.year = year
        self.y_continuous = self.df["Value"]
        self.threshold = threshold
        self.gt = gt
        self.inputs = pd.read_csv(r"Cleaned Data/InputsMaster.csv")
        # need to remove input pop/gdp not relevant to this region
        self.region = region
        columns_to_remove = []
        for column in self.inputs.columns:
            signifiers = [" GDP", "Non-{} GDP".format(self.region), " Pop", "Non-{} Pop".format(self.region)]
            if any(signifier in column for signifier in signifiers) and self.region not in column:
                columns_to_remove.append(column)
        self.inputs = self.inputs.drop(columns = columns_to_remove)

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
        X = self.inputs[self.inputs.columns[1:]]
        
        percentile = np.percentile(self.y_continuous, self.threshold)
        if self.gt:
            y_discrete = np.where(self.y_continuous.to_numpy() > percentile, 1, 0)
        else:
            y_discrete = np.where(self.y_continuous.to_numpy() < percentile, 1, 0)

        return X, y_discrete

    def CART(self):
        X, y = self.preprocess_for_classification()
        fit_model = DecisionTreeClassifier().fit(X, y)

        return fit_model

    def random_forest(self, num_to_plot = 5):
        X, y = self.preprocess_for_classification()
        fit_model = RandomForestClassifier(n_estimators = 100).fit(X, y)

        # get the average feature importances
        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in fit_model.estimators_], columns = X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n

if __name__ == "__main__":
    db = SQLConnection("all_data_jan_2024")
    df1 = DataRetrieval(db, "primary_energy_use_Total_EJ", "GLB", "Ref", 2050).input_output_mapping_df()
    df2 = DataRetrieval(db, "primary_energy_use_Renewables_EJ", "USA", "Ref", 2050).input_output_mapping_df()
    df = df2/df1

    top_n = InputOutputMapping("primary_energy_use_Total_EJ", "GLB", df2).random_forest()
    print(top_n)