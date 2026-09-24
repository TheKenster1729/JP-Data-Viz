"""Dashboard figure builders (plotly)."""

import sys
from pathlib import Path

# `python eppa_viz/figures/clustering.py` only adds this folder to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

from eppa_viz.analysis import TimeSeriesClustering
from eppa_viz.figures.base import DashboardFigure, OUTPUT_TIMESERIES
from eppa_viz.figures.cluster_label_cache import (
    ClusterLabelFileCache,
    _FittedClusters,
    fitted_from_labels,
    load_cluster_labels,
    reorder_clusters_by_terminal_year,
    save_cluster_labels,
    cluster_labels_csv_path,
)
from eppa_viz.figures.clustering_style import (
    PAPER_CLUSTER_COLORS,
    PAPER_CLUSTER_GRAY,
    apply_publication_cluster_style,
    apply_publication_cluster_cart_style,
    cart_y_axis_label,
    cluster_legend_labels,
    default_y_axis_label,
    paper_cart_title,
    paper_cluster_title,
    plot_years_from_pivot,
)
from sql_utils import DataRetrieval, SQLConnection
from styling import Color, Options, Readability



def _scenario_subplot_title(scenario):
    opts = Options()
    if scenario in opts.publication_scenario_display_names:
        return opts.publication_scenario_display_names[scenario]
    return opts.scenario_display_names[scenario]


class TimeSeriesClusteringPlot(ClusterLabelFileCache, TimeSeriesClustering, DashboardFigure):
    def __init__(
        self,
        df,
        output,
        region,
        scenario,
        n_clusters=3,
        metric="euclidean",
        year=None,
        num_to_plot=5,
        cart_depth=4,
        n_estimators=100,
        max_depth=4,
        dataset_name="publication",
    ):
        self.dataset_name = dataset_name
        super().__init__(
            df,
            output,
            region,
            scenario,
            n_clusters,
            metric=metric,
            num_to_plot=num_to_plot,
            cart_depth=cart_depth,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        DashboardFigure.__init__(self, "ts-clustering")
        self.colors = list(PAPER_CLUSTER_COLORS)
        self.year = year
        self.output = output
        self.region = region
        self.scenario = scenario
        self.fig = self.make_plot()

    def single_trace(self, x_years, data, cluster, color, showlegend=False):
        opacity = 1.0 if showlegend else 0.22
        line_thickness = 3.5 if showlegend else 0.6
        mode = "lines+markers" if showlegend else "lines"
        marker = dict(size=5, color=color, line=dict(width=0.5, color="white")) if showlegend else None
        trace = go.Scatter(
            x=x_years,
            y=data,
            legendgroup=cluster,
            mode=mode,
            line=dict(color=color, width=line_thickness),
            marker=marker,
            showlegend=showlegend,
            name=cluster,
            opacity=opacity,
        )
        return trace

    def make_plot(self, show=False):
        clusters = self.generate_clusters()
        fig = go.Figure()
        x_years = plot_years_from_pivot(self.df_for_clustering)
        legend_names = cluster_legend_labels(clusters.labels_, self.n_clusters)

        assert len(clusters.labels_) == len(self.df_for_clustering)
        for i in range(len(self.df_for_clustering)):
            inidvidual_time_series = self.df_for_clustering.iloc[i].values
            cluster_number = clusters.labels_[i]
            cluster_label = legend_names[cluster_number]
            color = self.colors[cluster_number % len(self.colors)]

            fig.add_trace(
                self.single_trace(x_years, inidvidual_time_series, cluster_label, color, showlegend=False)
            )

        cluster_centers = clusters.cluster_centers_
        for i, yi in enumerate(cluster_centers):
            color = self.colors[i % len(self.colors)]
            fig.add_trace(
                self.single_trace(
                    x_years, yi.ravel(), legend_names[i], color, showlegend=True,
                )
            )

        title = paper_cluster_title(self.output, self.region, self.scenario)
        y_label = default_y_axis_label(self.output)
        apply_publication_cluster_style(fig, title, y_label, x_years)

        if show:
            fig.show()

        return fig

class TimeSeriesClusteringPlotCART(ClusterLabelFileCache, TimeSeriesClustering, DashboardFigure):
    def __init__(
        self,
        df,
        output,
        region,
        scenario,
        n_clusters=3,
        metric="euclidean",
        year=None,
        num_to_plot=4,
        cart_depth=4,
        n_estimators=100,
        max_depth=4,
        dataset_name="publication",
        highlight_cluster=1,
    ):
        self.dataset_name = dataset_name
        self.highlight_cluster = int(highlight_cluster)
        super().__init__(
            df,
            output,
            region,
            scenario,
            n_clusters,
            metric=metric,
            num_to_plot=num_to_plot,
            cart_depth=cart_depth,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        DashboardFigure.__init__(self, "ts-clustering-cart")
        self.colors = list(PAPER_CLUSTER_COLORS)
        self.output = output
        self.region = region
        self.scenario = scenario
        self.fig = self.make_plot()
        self.year = year

    def make_plot(self, show=False):
        cluster_index = self.highlight_cluster - 1
        if cluster_index < 0 or cluster_index >= self.n_clusters:
            raise ValueError(
                "highlight_cluster must be between 1 and {}, got {}".format(
                    self.n_clusters, self.highlight_cluster,
                )
            )

        _, sorted_labeled_importances, top_n = self.cluster_mapping_binary(cluster_index)
        readability = Readability()
        feature_labels = [readability.display_name_for_output(name) for name in top_n]
        importances = sorted_labeled_importances[top_n].to_numpy()

        clusters = self.generate_clusters()
        labels_by_run = pd.Series(clusters.labels_, index=self.df_for_clustering.index)
        x_years = plot_years_from_pivot(self.df_for_clustering)
        cluster_color = self.colors[cluster_index % len(self.colors)]

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.42, 0.58],
            specs=[[{"type": "xy"}, {"type": "xy"}]],
        )
        fig.add_trace(
            go.Bar(
                x=feature_labels,
                y=importances,
                marker_color=cluster_color,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        for i in range(len(self.df_for_clustering)):
            series = self.df_for_clustering.iloc[i].values
            run_label = labels_by_run.iloc[i]
            in_cluster = run_label == cluster_index
            fig.add_trace(
                go.Scatter(
                    x=x_years,
                    y=series,
                    mode="lines",
                    line=dict(
                        color=cluster_color if in_cluster else PAPER_CLUSTER_GRAY,
                        width=1.2 if in_cluster else 0.5,
                    ),
                    opacity=0.9 if in_cluster else 0.45,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        title = paper_cart_title(
            self.output, self.region, self.scenario, self.highlight_cluster,
        )
        y_label = cart_y_axis_label(self.output)
        apply_publication_cluster_cart_style(fig, title, y_label, x_years)
        fig.update_layout(width=1000, height=750)

        if show:
            fig.show()

        return fig

if __name__ == "__main__":
    db_obj = SQLConnection("publication")
    renewable_share_spec = {
        "operation": "division",
        "output1": "elec_prod_Renewables_TWh",
        "output2": {
            "operation": "addition",
            "outputs": [
                "elec_prod_Renewables_TWh",
                "elec_prod_Hydro_TWh",
                "elec_prod_Nuclear_TWh",
                "elec_prod_Coal_CCS_TWh",
                "elec_prod_Coal_No_CCS_TWh",
                "elec_prod_Gas_No_CCS_TWh",
                "elec_prod_Gas_CCS_TWh",
                "elec_prod_Oil_TWh",
                "elec_prod_Biomass_No_CCS_TWh",
            ],
            "name": "Total Electricity",
        },
        "name": "Renewable Share",
    }
    region, scenario = "GLB", "2C"
    renewable_share = json.dumps(renewable_share_spec)
    df = DataRetrieval(db_obj, renewable_share, region, scenario).single_output_df()
    # plot = TimeSeriesClusteringPlot(df, renewable_share, region, scenario)
    from eppa_viz.figures.cluster_output_parcoords import ClusterOutputParcoordsPlot
    from eppa_viz.figures.multi_region_rf_heatmap import PublicationMultiRegionRFHeatmap
    # fig = PublicationMultiRegionRFHeatmap(db_obj, renewable_share, ["GLB", "USA", "EUR", "CHN"], ["2C", "Ref"]).make_plot()
    # fig.show()
    fig = ClusterOutputParcoordsPlot(db_obj, df, renewable_share, region, scenario, 2100).make_plot()
    # from styling import FinishedFigure
    # fig = FinishedFigure(plot).make_finished_figure()
    fig.show()
    # fig.write_image(f"Figures/GLB 2C 2100 Output-Output Parcoords.png", width=1000, height=750, scale = 2)
    # fig.write_image(f"Figures/GLB 2C 2100 Output-Output Parcoords.svg")
