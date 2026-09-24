"""Random-forest input importance heatmaps across regions, scenarios, and years."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eppa_viz.analysis import InputOutputMapping
from eppa_viz.figures.base import DashboardFigure
from eppa_viz.figures.clustering_style import PAPER_CLUSTER_FONT

PUBLICATION_MULTI_REGION_ORDER = ["GLB", "CHN", "EUR", "USA"]
# Left column = policy (2C), right column = reference (Ref).
PUBLICATION_MULTI_REGION_SCENARIOS = ["2C", "Ref"]
SCENARIO_COLUMN_TITLE = {"Ref": "Reference", "2C": "2C"}
MULTI_REGION_AXIS_FONT = 16
MULTI_REGION_COLORBAR_FONT = 13
MULTI_REGION_TITLE_FONT = 20
MULTI_REGION_ROW_HEIGHT = 300
MULTI_REGION_FIGURE_WIDTH = 2100
MULTI_REGION_COLORBAR_PAPER_Y = -0.12
MULTI_REGION_COLORBAR_TITLE = "Avg. Feature Importance"
MULTI_REGION_YEAR_TICK_STEP = 10
MULTI_REGION_VERTICAL_SPACING = 0.11
MULTI_REGION_HORIZONTAL_SPACING = 0.15
MULTI_REGION_MARGIN_LEFT = 155
MULTI_REGION_MARGIN_RIGHT = 80
# Monochromatic scales (light → dark).
POLICY_COLORSCALE = [[0.0, "#f4f7fb"], [1.0, "#002d9c"]]
REF_COLORSCALE = [[0.0, "#f6f2ff"], [1.0, "#6929c4"]]

from sql_utils import DataRetrieval
from styling import Options, Readability


def _scenario_colorscale(scenario):
    return REF_COLORSCALE if scenario == "Ref" else POLICY_COLORSCALE


def _layout_xaxis_key(axis_index):
    return "xaxis" if axis_index == 1 else "xaxis{}".format(axis_index)


def _subplot_x_center_paper(fig, row, col, n_cols):
    axis_index = (row - 1) * n_cols + col
    domain = fig.layout[_layout_xaxis_key(axis_index)].domain
    return (domain[0] + domain[1]) / 2.0


def _subplot_width_paper(fig, row, col, n_cols):
    axis_index = (row - 1) * n_cols + col
    domain = fig.layout[_layout_xaxis_key(axis_index)].domain
    return domain[1] - domain[0]


def _year_axis_tickvals(years, step=MULTI_REGION_YEAR_TICK_STEP):
    return [y for y in years if y % step == 0]


class PublicationMultiRegionRFHeatmap(DashboardFigure):
    """
    RF feature-importance heatmaps: one panel per (region, scenario), years on x,
    top-N inputs on y (N chosen by summed importance over the full year series).
    """

    def __init__(
        self,
        db,
        output,
        regions=None,
        scenarios=None,
        years=None,
        top_n=5,
        n_estimators=100,
        max_workers=16,
    ):
        DashboardFigure.__init__(self, "multi-region-rf-heatmap")
        self.db = db
        self.output = output
        self.regions = list(regions or PUBLICATION_MULTI_REGION_ORDER)
        self.scenarios = list(scenarios or PUBLICATION_MULTI_REGION_SCENARIOS)
        self.years = list(years or Options().years)
        self.top_n = top_n
        self.n_estimators = n_estimators
        self.max_workers = max_workers
        self.readability = Readability()
        self.importance_df = self._build_importance_dataframe()
        self.fig = self.make_plot()

    def _fetch_mapping_df(self, region, scenario, year):
        try:
            return DataRetrieval(self.db, self.output, region, scenario, year).mapping_df()
        except Exception:
            return None

    def _train_year(self, region, scenario, year, mapping_df):
        try:
            model = InputOutputMapping(
                self.output,
                region,
                scenario,
                year,
                mapping_df,
                n_estimators=self.n_estimators,
            )
            _, sorted_importances, _ = model.random_forest(n_jobs=-1)
            return [
                {
                    "Year": year,
                    "Region": region,
                    "Scenario": scenario,
                    "Input": inp,
                    "Importance": float(imp),
                }
                for inp, imp in sorted_importances.items()
            ]
        except Exception:
            return []

    def _build_importance_dataframe(self):
        combinations = [
            (reg, sce, year)
            for reg in self.regions
            for sce in self.scenarios
            for year in self.years
        ]
        fetched = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._fetch_mapping_df, reg, sce, year): (reg, sce, year)
                for reg, sce, year in combinations
            }
            for future in as_completed(futures):
                reg, sce, year = futures[future]
                mapping_df = future.result()
                if mapping_df is not None and len(mapping_df):
                    fetched[(reg, sce, year)] = mapping_df

        rows = []
        for reg, sce, year in combinations:
            mapping_df = fetched.get((reg, sce, year))
            if mapping_df is None:
                continue
            rows.extend(self._train_year(reg, sce, year, mapping_df))
        return pd.DataFrame(rows)

    def _top_inputs(self, region, scenario):
        sub = self.importance_df[
            (self.importance_df["Region"] == region)
            & (self.importance_df["Scenario"] == scenario)
        ]
        if sub.empty:
            return []
        ranked = (
            sub.groupby("Input")["Importance"]
            .sum()
            .sort_values(ascending=False)
        )
        return ranked.head(self.top_n).index.tolist()

    def _input_label(self, input_id):
        return self.readability.display_name_for_output(input_id)

    def _panel_matrix(self, region, scenario):
        inputs = self._top_inputs(region, scenario)
        if not inputs:
            return None, None, None
        sub = self.importance_df[
            (self.importance_df["Region"] == region)
            & (self.importance_df["Scenario"] == scenario)
            & (self.importance_df["Input"].isin(inputs))
        ]
        labels = [self._input_label(inp) for inp in inputs]
        years = sorted(sub["Year"].unique())
        z = []
        for inp in inputs:
            row = sub[sub["Input"] == inp].set_index("Year")["Importance"]
            z.append([row.get(y, 0.0) for y in years])
        return labels, years, np.array(z, dtype=float)

    def make_plot(self, show=False):
        n_rows = len(self.regions)
        n_cols = len(self.scenarios)
        subplot_titles = []
        for row_idx, _region in enumerate(self.regions):
            for col_idx, scenario in enumerate(self.scenarios):
                if row_idx == 0:
                    subplot_titles.append(SCENARIO_COLUMN_TITLE.get(scenario, scenario))
                else:
                    subplot_titles.append("")
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_xaxes=False,
            subplot_titles=subplot_titles,
            row_titles=self.regions,
            horizontal_spacing=MULTI_REGION_HORIZONTAL_SPACING,
            vertical_spacing=MULTI_REGION_VERTICAL_SPACING,
        )
        year_ticks = list(self.years)
        year_label_ticks = _year_axis_tickvals(year_ticks)

        for row_idx, region in enumerate(self.regions):
            for col_idx, scenario in enumerate(self.scenarios):
                labels, years, z = self._panel_matrix(region, scenario)
                if z is None:
                    continue
                zmax = float(np.nanmax(z)) if z.size else 1.0
                if zmax <= 0:
                    zmax = 1.0
                show_scale = row_idx == n_rows - 1
                colorbar = None
                if show_scale:
                    x_center = _subplot_x_center_paper(
                        fig, row_idx + 1, col_idx + 1, n_cols,
                    )
                    bar_len = min(0.42, _subplot_width_paper(
                        fig, row_idx + 1, col_idx + 1, n_cols,
                    ) * 0.88)
                    colorbar = dict(
                        orientation="h",
                        len=bar_len,
                        thickness=12,
                        x=x_center,
                        xref="paper",
                        xanchor="center",
                        y=MULTI_REGION_COLORBAR_PAPER_Y,
                        yref="paper",
                        yanchor="top",
                        tickfont=dict(
                            size=MULTI_REGION_COLORBAR_FONT,
                            family=PAPER_CLUSTER_FONT,
                        ),
                        title=dict(
                            text=MULTI_REGION_COLORBAR_TITLE,
                            side="top",
                            font=dict(
                                size=MULTI_REGION_COLORBAR_FONT,
                                family=PAPER_CLUSTER_FONT,
                            ),
                        ),
                    )
                fig.add_trace(
                    go.Heatmap(
                        x=years,
                        y=labels,
                        z=z,
                        zmin=0,
                        zmax=zmax,
                        colorscale=_scenario_colorscale(scenario),
                        showscale=show_scale,
                        colorbar=colorbar,
                        hoverongaps=False,
                    ),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )
                fig.update_yaxes(
                    autorange="reversed",
                    type="category",
                    categoryorder="array",
                    categoryarray=labels,
                    tickmode="array",
                    tickvals=labels,
                    ticktext=labels,
                    automargin=True,
                    tickfont=dict(size=MULTI_REGION_AXIS_FONT, family=PAPER_CLUSTER_FONT),
                    row=row_idx + 1,
                    col=col_idx + 1,
                )
                fig.update_xaxes(
                    tickangle=0,
                    showticklabels=True,
                    tickfont=dict(size=MULTI_REGION_AXIS_FONT, family=PAPER_CLUSTER_FONT),
                    tickmode="array",
                    tickvals=year_label_ticks,
                    row=row_idx + 1,
                    col=col_idx + 1,
                )

        for ann in fig.layout.annotations:
            ann.font = dict(size=MULTI_REGION_TITLE_FONT, family=PAPER_CLUSTER_FONT)

        fig_height = 100 + n_rows * MULTI_REGION_ROW_HEIGHT + 100
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family=PAPER_CLUSTER_FONT, color="black"),
            margin=dict(
                l=MULTI_REGION_MARGIN_LEFT,
                r=MULTI_REGION_MARGIN_RIGHT,
                t=85,
                b=140,
            ),
            height=fig_height,
            width=MULTI_REGION_FIGURE_WIDTH,
        )

        if show:
            fig.show()
        return fig


if __name__ == "__main__":
    import json

    from styling import FinishedFigure
    from sql_utils import SQLConnection

    from eppa_viz.figures.cluster_output_parcoords import RENEWABLE_SHARE_SPEC
    from eppa_viz.figures.multi_region_rf_heatmap import MULTI_REGION_FIGURE_WIDTH

    db = SQLConnection("publication")
    output = json.dumps(RENEWABLE_SHARE_SPEC)
    plot = PublicationMultiRegionRFHeatmap(db, output)
    FinishedFigure(plot).make_finished_figure().write_image(
        "Figures/Publication Multi Region RF Heatmap.png",
        width=MULTI_REGION_FIGURE_WIDTH,
        height=plot.fig.layout.height,
        scale=2,
    )
