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

class RegionalHeatmaps(DashboardFigure):
    """
    Regional heatmaps showing feature importance across regions, scenarios, and years.
    
    Optimized with:
    1. Parallel data fetching (I/O bound) using ThreadPoolExecutor
    2. Sequential model training (CPU bound) using sklearn's internal parallelism
    3. Reduced n_estimators (30) for faster training
    """
    
    def __init__(self, db_obj, output, regions, scenarios, year=None, n_estimators=30, max_workers=16):
        super().__init__("regional-heatmaps")
        self.db_obj = db_obj
        self.output = output
        self.regions = regions
        self.scenarios = scenarios
        self.year = year  # adding for consistency with other figures
        self.n_estimators = n_estimators
        self.max_workers = max_workers
        
        # Fetch data and build the DataFrame
        self.df = self._build_importance_dataframe()
        self.fig = self.make_plot()

    def _fetch_single_combination(self, args):
        """
        Fetch data for a single (region, scenario, year) combination.
        I/O bound - designed for concurrent execution.
        """
        reg, sce, year = args
        key = (reg, sce, year)
        try:
            mapping_df = DataRetrieval(self.db_obj, self.output, reg, sce, year).mapping_df()
            return (key, mapping_df)
        except Exception:
            return (key, None)

    def _train_single_model(self, reg, sce, year, mapping_df):
        """
        Train Random Forest for a single combination.
        CPU bound - uses sklearn internal parallelism.
        """
        try:
            model = InputOutputMapping(self.output, reg, sce, year, mapping_df, n_estimators=self.n_estimators)
            importances, sorted_importances, top_n = model.random_forest(n_jobs=-1)
            results_to_add = sorted_importances[top_n]
            
            return [
                {"Year": year, "Region": reg, "Scenario": sce, "Input": inp, "Importance": imp}
                for inp, imp in zip(results_to_add.index, results_to_add.values)
            ]
        except Exception:
            return []

    def _build_importance_dataframe(self):
        """
        Two-phase data processing:
        1. Parallel data fetching (I/O bound)
        2. Sequential model training (CPU bound with sklearn parallelism)
        """
        years = Options().years
        
        # Build all combinations to process
        combinations = [
            (reg, sce, year)
            for reg in self.regions
            for sce in self.scenarios
            for year in years
        ]
        
        # Phase 1: Parallel data fetching
        fetched_data = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single_combination, combo): combo 
                for combo in combinations
            }
            
            for future in as_completed(futures):
                key, mapping_df = future.result()
                if mapping_df is not None:
                    fetched_data[key] = mapping_df
        
        # Phase 2: Sequential model training with sklearn parallelism.
        # Iterate combinations rather than fetched_data, whose insertion order
        # depends on which fetches finished first.
        all_results = []
        for reg, sce, year in combinations:
            mapping_df = fetched_data.get((reg, sce, year))
            if mapping_df is None:
                continue
            all_results.extend(self._train_single_model(reg, sce, year, mapping_df))
        
        return pd.DataFrame(all_results)

    def run_random_forest(self, reg, sce, year):
        """Legacy method - kept for backward compatibility."""
        importances, sorted_importances, top_n = InputOutputMapping(self.output, reg, sce, year, self.df).random_forest()
        return importances, sorted_importances, top_n

    def make_plot(self, show=False):
        fig = make_subplots(
            rows=len(self.regions), 
            cols=len(self.scenarios), 
            shared_xaxes=True,
            subplot_titles=[f"{sce}" for sce in self.scenarios],
            vertical_spacing=0.033
        )

        # Calculate summed importance for each input over all years/regions/scenarios,
        # then filter df to only top 5 summed importance inputs
        summed = self.df.groupby("Input")["Importance"].sum().sort_values(ascending=False)
        top5_inputs = summed.head(8).index
        self.df = self.df[self.df["Input"].isin(top5_inputs)]
        zmax = self.df["Importance"].max()
        # Reduce vertical space between subplots by a factor of 3:
        # To do this, update the layout after creating traces
        for j, reg in enumerate(self.regions):
            for k, sce in enumerate(self.scenarios):
                df_to_plot = self.df[self.df["Region"].isin([reg]) & self.df["Scenario"].isin([sce])]
                fig.add_trace(
                    go.Heatmap(
                        y=df_to_plot["Input"], 
                        x=df_to_plot["Year"], 
                        z=df_to_plot["Importance"], 
                        zmin=0, 
                        zmax=zmax, 
                        colorscale="bupu"  # Same colorscale as choropleth mapping
                    ), 
                    row=j + 1, col=k + 1
                )

                if k == 0:
                    fig.update_yaxes(title_text=reg, row=j + 1, col=1)
        
        # Reduce vertical space (set vertical_spacing to 1/3 of default 0.1 -> 0.033)
        fig.update_layout(
            showlegend=False,
            title="Regional Heatmaps",
            height=1000,
            width = 1200,
        )

        if show:
            fig.show()

        return fig

