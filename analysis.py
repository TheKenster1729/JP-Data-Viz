import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sql_utils import SQLConnection, DataRetrieval
from styling import Readability, Options
from tslearn.clustering import TimeSeriesKMeans
from sklearn.inspection import permutation_importance
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text

# Module-level cache for inputs CSV to avoid repeated file reads
_INPUTS_CACHE = None

def _get_inputs_df():
    """
    Cached loader for InputsMasterTFP.csv.
    Reads from disk only once, then returns cached copy.
    """
    global _INPUTS_CACHE
    if _INPUTS_CACHE is None:
        _INPUTS_CACHE = pd.read_csv(r"Cleaned Data/InputsMasterTFP.csv")
    return _INPUTS_CACHE.copy()  # Return copy to avoid mutation issues


class InputOutputMapping:
    def __init__(self, output, region, scenario, year, df, threshold = 70, gt = True, num_to_plot = 5, cart_depth = 4, n_estimators = 100, max_depth = 4):
        self.output = output
        self.df = df
        self.scenario = scenario
        self.year = year
        self.y_continuous = self.df["Value"]
        self.threshold = threshold
        self.gt = gt
        self.inputs = _get_inputs_df()  # Use cached version
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

    def random_forest(self, n_jobs=1):
        """
        Train Random Forest and return feature importances.
        
        Args:
            n_jobs: Number of parallel jobs for sklearn. Use 1 when calling from
                   ThreadPoolExecutor to avoid CPU over-subscription. Use -1 for
                   single-threaded callers to use all cores.
        """
        X, y = self.preprocess_for_classification()
        fit_model = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            max_depth=self.max_depth,
            n_jobs=n_jobs
        ).fit(X, y)

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
    """
    Maps relationships between outputs using Random Forest classification.
    
    Optimized with:
    1. Batch SQL queries - fetches all table names in a single query
    2. Concurrent data retrieval - uses ThreadPoolExecutor for parallel I/O
    3. Efficient DataFrame operations - minimizes memory copies
    """
    
    # Class-level cache for table name mappings (shared across instances)
    _table_name_cache = {}
    
    def __init__(self, db_obj, output, region, scenario, year, df, threshold = 70, gt = True, num_to_plot = 5, other_outputs = [], max_workers = 8):
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
        self.max_workers = max_workers  # Number of concurrent threads for DB queries

    def _get_table_names_batch(self, outputs):
        """
        Fetch all table names for the given outputs in a single SQL query.
        Uses class-level caching to avoid repeated lookups.
        """
        cache_key = (self.db_obj.dbname, self.region, self.scenario)
        
        # Check if we have cached mappings for this database/region/scenario combo
        if cache_key not in OutputOutputMapping._table_name_cache:
            OutputOutputMapping._table_name_cache[cache_key] = {}
        
        cached = OutputOutputMapping._table_name_cache[cache_key]
        
        # Determine which outputs need to be fetched
        outputs_to_fetch = [o for o in outputs if o not in cached]
        
        if outputs_to_fetch:
            # Build long names for all outputs we need
            long_names = []
            output_to_long_name = {}
            for output in outputs_to_fetch:
                long_name = f"{output}_{self.region}_{self.scenario}"
                long_names.append(long_name)
                output_to_long_name[long_name] = output
            
            # Single batch query for all table names
            placeholders = ", ".join([f":ln{i}" for i in range(len(long_names))])
            query = text(f"SELECT `Full Output Name`, `Assigned Name` FROM name_mappings WHERE `Full Output Name` IN ({placeholders})")
            params = {f"ln{i}": ln for i, ln in enumerate(long_names)}
            
            with self.db_obj.retrieval_engine.connect() as conn:
                result = conn.execute(query, params).fetchall()
            
            # Update cache
            for full_name, table_name in result:
                output = output_to_long_name.get(full_name)
                if output:
                    cached[output] = table_name
        
        # Return mapping for requested outputs
        return {o: cached.get(o) for o in outputs if o in cached}

    def _fetch_single_output(self, output, table_name, target_run_numbers_set):
        """
        Fetch data for a single output. Designed to be called concurrently.
        Returns tuple of (output_name, series) or (output_name, None) on failure.
        """
        try:
            # Read only the columns we need and filter by year in SQL if possible
            query = text(f"SELECT `Run #`, `Value` FROM `{table_name}` WHERE `Year` = :year")
            with self.db_obj.retrieval_engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"year": self.year})
            
            if df.empty:
                return (output, None)
            
            # Handle Eps values
            df["Value"] = df["Value"].replace("Eps", 0)
            df = df.dropna()
            
            # Check if target runs are a subset
            new_df_runs_set = set(df["Run #"].values)
            if target_run_numbers_set.issubset(new_df_runs_set):
                filtered_df = df[df["Run #"].isin(target_run_numbers_set)]
                return (output, filtered_df.set_index("Run #")["Value"])
            
            return (output, None)
        except Exception:
            return (output, None)

    def preprocess_for_classification(self):
        """
        Prepare data for classification using optimized batch queries and concurrent fetching.
        """
        # Validation checks
        if not len(self.df) > 350:
            return "insufficient length (< 350)"
        
        if not len(self.df.query("Value==0")) < 175:
            return "too many (> half) zero values"
        
        # Get the initial set of run numbers
        target_output_run_numbers = self.df["Run #"]
        target_runs_set = set(target_output_run_numbers.values)

        # Determine which outputs to use
        if self.db_obj.dbname == "all_data_aug_2024":
            options_to_use = list(Options().outputs)
        elif self.db_obj.dbname == "publication":
            options_to_use = list(Options().publication_outputs)
        else:
            raise ValueError("Invalid database name")
        
        list_to_use = options_to_use + self.other_outputs
        if self.output in list_to_use:
            list_to_use.remove(self.output)
        
        # Batch fetch all table names in a single query
        table_names = self._get_table_names_batch(list_to_use)
        
        # Filter to outputs that have valid table names
        outputs_with_tables = [(o, table_names[o]) for o in list_to_use if table_names.get(o)]
        
        # Concurrent data fetching using ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all fetch tasks
            future_to_output = {
                executor.submit(self._fetch_single_output, output, table_name, target_runs_set): output
                for output, table_name in outputs_with_tables
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_output):
                output_name, series = future.result()
                if series is not None:
                    results[output_name] = series
        
        # Build the main DataFrame efficiently
        main_df = pd.DataFrame({"Run #": target_output_run_numbers})
        main_df = main_df.set_index("Run #")
        
        # Concatenate all valid series at once (more efficient than iterative concat)
        if results:
            results_df = pd.DataFrame(results)
            # Align on index (Run #)
            main_df = main_df.join(results_df, how="left")
        
        # Calculate discrete target variable
        percentile = np.percentile(self.y_continuous, self.threshold)
        if self.gt:
            y_discrete = np.where(self.y_continuous.to_numpy() > percentile, 1, 0)
        else:
            y_discrete = np.where(self.y_continuous.to_numpy() < percentile, 1, 0)
        
        self.main_df = main_df.reset_index(drop=True)

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
    
    @classmethod
    def clear_cache(cls):
        """Clear the table name cache. Useful if database structure changes."""
        cls._table_name_cache = {}

class FilteredInputOutputMapping:
    def __init__(self, constraint_df, region, scenario, year, num_to_plot = 5, cart_depth = 4, n_estimators = 100, random_forest_depth = 4):
        self.constraint_df = constraint_df
        self.region = region
        self.scenario = scenario
        self.year = year
        self.inputs = _get_inputs_df()  # Use cached version
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
        self.preprocess_for_classification()

        fit_model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.random_forest_depth).fit(self.X, self.y_discrete)
        permutation_importance_results = permutation_importance(fit_model, self.X, self.y_discrete, n_repeats = 10)

        important = []
        for i in permutation_importance_results.importances_mean.argsort()[::-1]:
            if permutation_importance_results.importances_mean[i] - 3*permutation_importance_results.importances_std[i] > 0:
                important.append({"variable": self.X.columns[i], "mean": permutation_importance_results.importances_mean[i], "std": permutation_importance_results.importances_std[i]})

        return important

class FilteredOutputOutputMapping:
    """
    Maps relationships between ALL outputs and a user-defined constraint range.
    Uses Random Forest classification where the target is whether a run falls
    within the user's specified percentile constraints.
    
    Optimized with:
    1. Batch SQL queries - fetches all table names in a single query
    2. Concurrent data retrieval - uses ThreadPoolExecutor for parallel I/O
    3. Efficient DataFrame operations - minimizes memory copies
    """
    
    # Class-level cache for table name mappings (shared across instances)
    _table_name_cache = {}
    
    def __init__(self, db_obj, constraint_df, region, scenario, year, num_to_plot = 5, max_workers = 8):
        self.db_obj = db_obj
        self.constraint_df = constraint_df
        self.region = region
        self.scenario = scenario
        self.year = year
        self.num_to_plot = num_to_plot
        self.max_workers = max_workers
        
        # Get the run numbers from constraint_df
        self.run_numbers = constraint_df["Run #"].values
        self.in_constraint_range = constraint_df["in_constraint_range"].values

    def _get_table_names_batch(self, outputs):
        """
        Fetch all table names for the given outputs in a single SQL query.
        Uses class-level caching to avoid repeated lookups.
        """
        cache_key = (self.db_obj.dbname, self.region, self.scenario)
        
        if cache_key not in FilteredOutputOutputMapping._table_name_cache:
            FilteredOutputOutputMapping._table_name_cache[cache_key] = {}
        
        cached = FilteredOutputOutputMapping._table_name_cache[cache_key]
        
        outputs_to_fetch = [o for o in outputs if o not in cached]
        
        if outputs_to_fetch:
            long_names = []
            output_to_long_name = {}
            for output in outputs_to_fetch:
                long_name = f"{output}_{self.region}_{self.scenario}"
                long_names.append(long_name)
                output_to_long_name[long_name] = output
            
            placeholders = ", ".join([f":ln{i}" for i in range(len(long_names))])
            query = text(f"SELECT `Full Output Name`, `Assigned Name` FROM name_mappings WHERE `Full Output Name` IN ({placeholders})")
            params = {f"ln{i}": ln for i, ln in enumerate(long_names)}
            
            with self.db_obj.retrieval_engine.connect() as conn:
                result = conn.execute(query, params).fetchall()
            
            for full_name, table_name in result:
                output = output_to_long_name.get(full_name)
                if output:
                    cached[output] = table_name
        
        return {o: cached.get(o) for o in outputs if o in cached}

    def _fetch_single_output(self, output, table_name, target_run_numbers_set):
        """
        Fetch data for a single output. Designed to be called concurrently.
        Returns tuple of (output_name, series) or (output_name, None) on failure.
        """
        try:
            query = text(f"SELECT `Run #`, `Value` FROM `{table_name}` WHERE `Year` = :year")
            with self.db_obj.retrieval_engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"year": self.year})
            
            if df.empty:
                return (output, None)
            
            df["Value"] = df["Value"].replace("Eps", 0)
            df = df.dropna()
            
            new_df_runs_set = set(df["Run #"].values)
            if target_run_numbers_set.issubset(new_df_runs_set):
                filtered_df = df[df["Run #"].isin(target_run_numbers_set)]
                return (output, filtered_df.set_index("Run #")["Value"])
            
            return (output, None)
        except Exception:
            return (output, None)

    def create_dataframe(self):
        """
        Fetch ALL outputs concurrently using ThreadPoolExecutor.
        """
        target_runs_set = set(self.run_numbers)
        
        # Determine which outputs to use based on database
        if self.db_obj.dbname == "all_data_aug_2024":
            options_to_use = list(Options().outputs)
        elif self.db_obj.dbname == "publication":
            options_to_use = list(Options().publication_outputs)
        else:
            raise ValueError("Invalid database name")
        
        # Batch fetch all table names in a single query
        table_names = self._get_table_names_batch(options_to_use)
        
        # Filter to outputs that have valid table names
        outputs_with_tables = [(o, table_names[o]) for o in options_to_use if table_names.get(o)]
        
        # Concurrent data fetching using ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_output = {
                executor.submit(self._fetch_single_output, output, table_name, target_runs_set): output
                for output, table_name in outputs_with_tables
            }
            
            for future in as_completed(future_to_output):
                output_name, series = future.result()
                if series is not None:
                    # Use human-readable names for columns
                    readable_name = Readability().naming_dict_long_names_first.get(output_name, output_name)
                    results[readable_name] = series
        
        # Build DataFrame efficiently
        if results:
            self.df_to_use = pd.DataFrame(results)
            self.df_to_use = self.df_to_use.loc[sorted(self.run_numbers)]
        else:
            self.df_to_use = pd.DataFrame()

    def run_analysis(self):
        self.create_dataframe()
        
        if self.df_to_use.empty:
            return None, []
        
        self.df_to_use["in_constraint_range"] = self.in_constraint_range
        self.df_to_use.dropna(how="any", inplace=True)

        X = self.df_to_use[self.df_to_use.columns[:-1]]
        y = self.df_to_use["in_constraint_range"]

        random_forest = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X, y)
        feature_importances = pd.DataFrame(
            [estimator.feature_importances_ for estimator in random_forest.estimators_], 
            columns=X.columns
        )
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending=False)
        top_n = sorted_labeled_importances.index[:self.num_to_plot].to_list()

        return sorted_labeled_importances, top_n

class TimeSeriesClustering:
    def __init__(self, df, output, region, scenario, n_clusters = 3, metric = "euclidean", num_to_plot = 5, cart_depth = 4, n_estimators = 100, max_depth = 4):
        self.df = df
        self.output = output
        self.region = region
        self.scenario = scenario
        self.n_clusters = n_clusters
        self.df_for_clustering = self.df.pivot(columns = "Year", index = "Run #")
        self.metric = metric
        self.cart_depth = cart_depth
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_to_plot = num_to_plot

    def generate_clusters(self):
        clusters = TimeSeriesKMeans(n_clusters = self.n_clusters, metric = self.metric).fit(self.df_for_clustering)

        return clusters

    def cluster_mapping(self):
        # using the clusters as the target, use random forest to find input drivers
        # of the clusters
        self.inputs = _get_inputs_df()  # Use cached version
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
        
        self.X = self.inputs[self.inputs.columns[1:]]
        self.y = self.generate_clusters().labels_
        fit_model = RandomForestClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth).fit(self.X, self.y)

        # get the average feature importances
        feature_importances = pd.DataFrame([estimator.feature_importances_ for estimator in fit_model.estimators_], columns = self.X.columns)
        sorted_labeled_importances = feature_importances.mean().sort_values(ascending = False)
        top_n = sorted_labeled_importances.index[:self.num_to_plot].to_list()

        return feature_importances, sorted_labeled_importances, top_n

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
    # df = DataRetrieval(db, "emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref", 2050).mapping_df()
    # io = InputOutputMapping("emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref", 2050, df).permutation_importance()
    # print(io)

    # time series clustering cart
    df = DataRetrieval(db, "emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref", 2050).mapping_df()
    results = TimeSeriesClustering(df, "emissions_CO2eq_total_million_ton_CO2eq", "GLB", "Ref").cluster_mapping()