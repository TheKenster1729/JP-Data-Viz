import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sql_utils import SQLConnection
from styling import Readability

class InputOutputMapping:
    def __init__(self, output, df, threshold = 70, gt = True):
        self.output = output
        self.df = df
        self.y_continuous = self.df["Value"]
        self.threshold = threshold
        self.gt = gt
        self.inputs = pd.read_csv(r"Cleaned Data/InputsMaster.csv")
        # need to remove input pop/gdp not relevant to this region
        self.region = self.df["Region"].unique()[0] # should only contain one value
        columns_to_remove = []
        for column in self.inputs.columns:
            signifiers = [" GDP", "Non-{} GDP".format(self.region), " Pop", "Non-{} Pop".format(self.region)]
            if any(signifier in column for signifier in signifiers) and self.region not in column:
                columns_to_remove.append(column)
        self.inputs = self.inputs.drop(columns = columns_to_remove)

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
        fit_model = RandomForestClassifier(n_estimators = 200).fit(X, y)

        # get the average feature importances
        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in fit_model.estimators_], columns = X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n

if __name__ == "__main__":
    df = SQLConnection("jp_data").input_output_mapping_df("elec_prod_Renewables_TWh", "USA", "Ref", 2050)
    # top_n = InputOutputMapping("elec_prod_Renewables_TWh", df).random_forest()
