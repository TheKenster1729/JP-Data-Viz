"""Parallel-coordinates view of author-chosen outputs, colored by time-series cluster."""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go

from eppa_viz.analysis import TimeSeriesClustering
from eppa_viz.figures.base import DashboardFigure
from eppa_viz.figures.cluster_label_cache import ClusterLabelFileCache
from eppa_viz.figures.clustering_style import (
    apply_publication_cluster_parcoords_style,
    cluster_parcoords_colorbar,
    cluster_parcoords_colorscale,
    paper_cluster_outputs_title,
    parcoords_axis_label,
    parcoords_dimension,
    PAPER_CLUSTER_FONT,
    PAPER_PARCOORDS_DOMAIN,
    PAPER_PARCOORDS_HEIGHT,
    PAPER_PARCOORDS_LABEL_SIZE,
    PAPER_PARCOORDS_RANGE_FONT_SIZE,
    PAPER_PARCOORDS_TICK_FONT_SIZE,
    PAPER_PARCOORDS_WIDTH,
)
from sql_utils import DataRetrieval

# Author-chosen outputs (not RF/CART). Labels match the publication figure axes.
TOTAL_ELECTRICITY_SPEC = {
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
}

TOTAL_PRIMARY_ENERGY_SPEC = {
    "operation": "addition",
    "outputs": [
        "primary_energy_use_Oil_EJ",
        "primary_energy_use_Biomass_EJ",
        "primary_energy_use_Hydro_EJ",
        "primary_energy_use_Gas_EJ",
        "primary_energy_use_Renewables_EJ",
        "primary_energy_use_Nuclear_EJ",
        "Primary_energy_use_Coal_EJ",
    ],
    "name": "Total Primary Energy",
}

ELECTRICITY_CCS_SPEC = {
    "operation": "addition",
    "outputs": ["elec_prod_Coal_CCS_TWh", "elec_prod_Gas_CCS_TWh"],
    "name": "Electricity CCS",
}

RENEWABLE_SHARE_SPEC = {
    "operation": "division",
    "output1": "elec_prod_Renewables_TWh",
    "output2": TOTAL_ELECTRICITY_SPEC,
    "name": "Renewables Share",
}

RENEWABLE_SHARE_AXIS_ID = "renewable_share_at_year"

PUBLICATION_CLUSTER_OUTPUT_AXES = [
    ("Total Elec", "(TWh)", json.dumps(TOTAL_ELECTRICITY_SPEC)),
    ("Total PrimEnergy", "(EJ)", json.dumps(TOTAL_PRIMARY_ENERGY_SPEC)),
    ("GDP Growth", "(%)", "gdp_growth"),
    ("Nuclear", "(TWh)", "elec_prod_Nuclear_TWh"),
    ("Gas", "(TWh)", "elec_prod_Gas_No_CCS_TWh"),
    ("CCS", "(TWh)", json.dumps(ELECTRICITY_CCS_SPEC)),
    ("Renew", "(TWh)", "elec_prod_Renewables_TWh"),
    ("Carbon Price", "(USD/t)", "carbon_price_USD2007_per_ton_CO2e"),
    (RENEWABLE_SHARE_AXIS_ID, None, json.dumps(RENEWABLE_SHARE_SPEC)),
]


def renewable_share_axis_label(year):
    return "{} Share".format(int(year))


def parcoords_axis_column_name(name, year):
    if name == RENEWABLE_SHARE_AXIS_ID:
        return renewable_share_axis_label(year)
    return name


def _values_at_year(db, output, region, scenario, year):
    frame = DataRetrieval(db, output, region, scenario, year).single_output_df()
    at_year = frame.loc[frame["Year"] == year, ["Run #", "Value"]]
    return at_year.set_index("Run #")["Value"]


def _gdp_growth_cagr(db, region, scenario, base_year=2020, end_year=2100):
    g0 = _values_at_year(db, "GDP_billion_USD2007", region, scenario, base_year)
    g1 = _values_at_year(db, "GDP_billion_USD2007", region, scenario, end_year)
    common = g0.index.intersection(g1.index)
    years = float(end_year - base_year)
    ratio = g1.reindex(common) / g0.reindex(common)
    return ratio.pow(1.0 / years) - 1.0


def build_cluster_output_matrix(db, region, scenario, year, run_index):
    """Wide matrix indexed by Run # for parcoords dimensions."""
    runs = list(run_index)
    matrix = pd.DataFrame(index=runs)
    for name, unit, spec in PUBLICATION_CLUSTER_OUTPUT_AXES:
        col = parcoords_axis_column_name(name, year)
        if spec == "gdp_growth":
            series = _gdp_growth_cagr(db, region, scenario)
        else:
            series = _values_at_year(db, spec, region, scenario, year)
        matrix[col] = series.reindex(runs)
    return matrix.dropna(how="any")


class ClusterOutputParcoordsPlot(ClusterLabelFileCache, TimeSeriesClustering, DashboardFigure):
    """
    Outputs vs clusters at a single year. Cluster colors match the time-series
    clustering figure; labels use the same reordered Cluster 1/2/3 scheme.
    """

    def __init__(
        self,
        db,
        df,
        cluster_output,
        region,
        scenario,
        year=2100,
        n_clusters=3,
        metric="euclidean",
        dataset_name="publication",
    ):
        self.db = db
        self.year = year
        self.dataset_name = dataset_name
        super().__init__(
            df,
            cluster_output,
            region,
            scenario,
            n_clusters=n_clusters,
            metric=metric,
        )
        DashboardFigure.__init__(self, "ts-clustering-outputs")
        self.output = cluster_output
        self.region = region
        self.scenario = scenario
        self.fig = self.make_plot()

    def make_plot(self, show=False):
        clusters = self.generate_clusters()
        labels = pd.Series(clusters.labels_, index=self.df_for_clustering.index)
        matrix = build_cluster_output_matrix(
            self.db, self.region, self.scenario, self.year, labels.index,
        )
        aligned_labels = labels.reindex(matrix.index).astype(int)

        dimensions = []
        for name, unit, _spec in PUBLICATION_CLUSTER_OUTPUT_AXES:
            col = parcoords_axis_column_name(name, self.year)
            if col not in matrix.columns:
                continue
            dimensions.append(
                parcoords_dimension(col, unit, matrix[col].to_numpy())
            )
        color_scale = cluster_parcoords_colorscale(self.n_clusters)
        fig = go.Figure(
            data=[
                go.Parcoords(
                    domain=PAPER_PARCOORDS_DOMAIN,
                    line=dict(
                        color=aligned_labels.to_numpy(),
                        colorscale=color_scale,
                        showscale=True,
                        colorbar=cluster_parcoords_colorbar(self.n_clusters),
                    ),
                    dimensions=dimensions,
                    labelfont=dict(
                        size=PAPER_PARCOORDS_LABEL_SIZE,
                        family=PAPER_CLUSTER_FONT,
                    ),
                    rangefont=dict(
                        size=PAPER_PARCOORDS_RANGE_FONT_SIZE,
                        family=PAPER_CLUSTER_FONT,
                    ),
                    tickfont=dict(
                        size=PAPER_PARCOORDS_TICK_FONT_SIZE,
                        family=PAPER_CLUSTER_FONT,
                    ),
                    labelside="top",
                    labelangle=0,
                )
            ]
        )
        title = paper_cluster_outputs_title(self.region, self.scenario, self.year)
        apply_publication_cluster_parcoords_style(fig, title)
        fig.update_layout(width=PAPER_PARCOORDS_WIDTH, height=PAPER_PARCOORDS_HEIGHT)

        if show:
            fig.show()
        return fig


if __name__ == "__main__":
    from styling import FinishedFigure
    from sql_utils import SQLConnection
    from eppa_viz.figures.clustering_style import (
        PAPER_PARCOORDS_HEIGHT,
        PAPER_PARCOORDS_WIDTH,
    )

    db_obj = SQLConnection("publication")
    share = json.dumps(RENEWABLE_SHARE_SPEC)
    region, scenario = "GLB", "Ref"
    df = DataRetrieval(db_obj, share, region, scenario).single_output_df()
    plot = ClusterOutputParcoordsPlot(
        db_obj, df, share, region, scenario, year=2100,
    )
    fig = FinishedFigure(plot).make_finished_figure()
    fig.write_image(
        f"Figures/{region} {scenario} Cluster Output Parcoords.png",
        width=PAPER_PARCOORDS_WIDTH,
        height=PAPER_PARCOORDS_HEIGHT,
        scale=2,
    )