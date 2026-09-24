"""The golden cases themselves.

Each entry is a name and a zero-argument callable. Names are namespaced
``layer/dataset/description`` so a failure immediately says whether retrieval,
a model, or a figure moved, and which database it came from.

Coverage is chosen to span the things a refactor is most likely to break:

- both databases, including output keys that exist in both with different data
- every retrieval shape: one series, a percentile band, a single year, a
  choropleth aggregate, and a multi-output join
- custom variables, both a plain division and a nested division-over-addition
- a non-global region, which exercises the region-based input filtering in
  ``_filter_inputs_by_region``
- each model entry point, and the figure built on top of it
- figures both raw and after ``FinishedFigure``, since the two paths disagree
  today and the refactor is meant to unify them
"""

import json

from analysis import (
    FilteredInputOutputMapping,
    InputOutputMapping,
    OutputOutputMapping,
    TimeSeriesClustering,
)
from figure import (
    ChoroplethMap,
    InputOutputMappingPlot,
    NewTimeSeries,
    OutputHistograms,
    PermutationImportance,
    PlotTree,
    TimeSeriesClusteringPlot,
    TimeSeriesClusteringPlotCART,
)
from sql_utils import DataRetrieval, MultiOutputRetrieval, SQLConnection
from styling import FinishedFigure

from tests.harness import canonical, summarize_figure, summarize_frame, summarize_tree

PUBLICATION = "publication"
FULL = "all_data_aug_2024"

# One connection per database, reused across cases. Each SQLConnection opens a
# pooled engine plus a raw mysql connector, so building one per case is slow.
_CONNECTIONS = {}


def db(name):
    if name not in _CONNECTIONS:
        _CONNECTIONS[name] = SQLConnection(name)
    return _CONNECTIONS[name]


# Publication has no elec_prod_Total_TWh, so the denominator has to be summed
# from the individual generation technologies.
RENEWABLE_SHARE_PUBLICATION = json.dumps({
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
})

# The full dataset ships a precomputed total, so the same quantity is a plain
# division there. Both are pinned because they take different code paths.
RENEWABLE_SHARE_FULL = json.dumps({
    "operation": "division",
    "output1": "elec_prod_Renewables_TWh_pol",
    "output2": "elec_prod_Total_TWh_pol",
    "name": "Renewable Share",
})

PUB_EMISSIONS = "total_emissions_CO2e_million_ton_CO2e"
FULL_EMISSIONS = "emissions_CO2eq_total_million_ton_CO2eq"


def _styled(fig_object):
    """Run a figure through the styling pipeline and summarize the result."""
    from eppa_viz.figures.pipeline import apply_finished_style
    if fig_object.fig is None and getattr(fig_object, "figure", None) is not None:
        fig_object.fig = fig_object.figure
    apply_finished_style(fig_object)
    return summarize_figure(fig_object.fig)


def cases():
    """Name -> callable. A dict so a single case can be run by name."""
    c = {}

    # ---- retrieval, publication -------------------------------------------
    c["retrieval/pub/emissions_GLB_2C_2050"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C", 2050).mapping_df())

    # GDP and population exist under the same key in both databases with
    # different numbers. These two and their full-dataset twins below are the
    # tripwire for dataset crossover.
    c["retrieval/pub/gdp_GLB_Ref_2050_COLLIDING"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), "GDP_billion_USD2007", "GLB", "Ref", 2050).mapping_df())
    c["retrieval/pub/population_USA_Ref_2050_COLLIDING"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), "population_million_people", "USA", "Ref", 2050).mapping_df())

    # Both of these were missing from the database until recently.
    c["retrieval/pub/gas_ccs_GLB_2C_series"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), "elec_prod_Gas_CCS_TWh", "GLB", "2C").single_output_df())
    c["retrieval/pub/carbon_price_GLB_Ref_series"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), "carbon_price_USD2007_per_ton_CO2e", "GLB", "Ref").single_output_df())

    c["retrieval/pub/gdp_GLB_2C_percentile_band"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), "GDP_billion_USD2007", "GLB", "2C").single_output_df_to_graph(5, 95))
    c["retrieval/pub/gdp_2C_2050_choropleth"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), "GDP_billion_USD2007", "GLB", "2C", 2050).choropleth_map_df(5, 95))
    c["retrieval/pub/renewable_share_nested_GLB_2C_2050"] = lambda: summarize_frame(
        DataRetrieval(db(PUBLICATION), RENEWABLE_SHARE_PUBLICATION, "GLB", "2C", 2050).mapping_df())

    # ---- retrieval, full dataset ------------------------------------------
    c["retrieval/full/gdp_GLB_Ref_2050_COLLIDING"] = lambda: summarize_frame(
        DataRetrieval(db(FULL), "GDP_billion_USD2007", "GLB", "Ref", 2050).mapping_df())
    c["retrieval/full/population_USA_Ref_2050_COLLIDING"] = lambda: summarize_frame(
        DataRetrieval(db(FULL), "population_million_people", "USA", "Ref", 2050).mapping_df())
    c["retrieval/full/emissions_GLB_15C_med_2050"] = lambda: summarize_frame(
        DataRetrieval(db(FULL), FULL_EMISSIONS, "GLB", "15C_med", 2050).mapping_df())
    c["retrieval/full/emissions_CHN_2C_med_2050"] = lambda: summarize_frame(
        DataRetrieval(db(FULL), FULL_EMISSIONS, "CHN", "2C_med", 2050).mapping_df())
    c["retrieval/full/renewable_share_division_GLB_15C_med_2050"] = lambda: summarize_frame(
        DataRetrieval(db(FULL), RENEWABLE_SHARE_FULL, "GLB", "15C_med", 2050).mapping_df())

    def _multi():
        m = MultiOutputRetrieval(
            db(FULL),
            [FULL_EMISSIONS, "GDP_billion_USD2007", "consumption_billion_USD2007"],
            "GLB", "15C_med", 2050)
        m.construct_df()
        return summarize_frame(m.df)
    c["retrieval/full/multi_output_GLB_15C_med_2050"] = _multi

    # ---- models ------------------------------------------------------------
    def _pub_mapping_df():
        return DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C", 2050).mapping_df()

    def _full_mapping_df():
        return DataRetrieval(db(FULL), FULL_EMISSIONS, "GLB", "15C_med", 2050).mapping_df()

    c["model/pub/iomap_cart_GLB"] = lambda: summarize_tree(
        InputOutputMapping(PUB_EMISSIONS, "GLB", "2C", 2050, _pub_mapping_df()).CART())

    def _pub_rf():
        _, importances, top_n = InputOutputMapping(
            PUB_EMISSIONS, "GLB", "2C", 2050, _pub_mapping_df()).random_forest()
        return {"top_n": canonical(top_n), "importances": canonical(importances.head(20))}
    c["model/pub/iomap_random_forest_GLB"] = _pub_rf

    c["model/pub/iomap_permutation_GLB"] = lambda: canonical(
        InputOutputMapping(PUB_EMISSIONS, "GLB", "2C", 2050, _pub_mapping_df()).permutation_importance())

    # A non-global region, which prunes the input set down to that region's
    # TFP/Pop and AEEI columns. The feature list itself is the assertion.
    def _pub_cart_chn():
        df = DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "CHN", "2C", 2050).mapping_df()
        mapping = InputOutputMapping(PUB_EMISSIONS, "CHN", "2C", 2050, df)
        return {
            "tree": summarize_tree(mapping.CART()),
            "input_columns": canonical(list(mapping.inputs.columns)),
        }
    c["model/pub/iomap_cart_CHN_region_filtered"] = _pub_cart_chn

    c["model/full/iomap_cart_GLB"] = lambda: summarize_tree(
        InputOutputMapping(FULL_EMISSIONS, "GLB", "15C_med", 2050, _full_mapping_df()).CART())

    def _full_rf():
        _, importances, top_n = InputOutputMapping(
            FULL_EMISSIONS, "GLB", "15C_med", 2050, _full_mapping_df()).random_forest()
        return {"top_n": canonical(top_n), "importances": canonical(importances.head(20))}
    c["model/full/iomap_random_forest_GLB"] = _full_rf

    def _clusters():
        df = DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C").single_output_df()
        fitted = TimeSeriesClustering(df, PUB_EMISSIONS, "GLB", "2C").generate_clusters()
        return {
            "labels": canonical(fitted.labels_),
            "inertia": canonical(fitted.inertia_),
            "centers": canonical(fitted.cluster_centers_),
        }
    c["model/pub/timeseries_clusters_GLB"] = _clusters

    def _cluster_mapping():
        df = DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C").single_output_df()
        _, importances, top_n = TimeSeriesClustering(
            df, PUB_EMISSIONS, "GLB", "2C", num_to_plot=4).cluster_mapping_binary(0)
        return {"top_n": canonical(top_n), "importances": canonical(importances.head(20))}
    c["model/pub/timeseries_cluster_mapping_GLB"] = _cluster_mapping

    def _output_output():
        mapping = OutputOutputMapping(
            db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C", 2050, _pub_mapping_df(),
            other_outputs=["GDP_billion_USD2007", "population_million_people",
                           "elec_prod_Renewables_TWh"])
        _, importances, top_n = mapping.random_forest()
        return {"top_n": canonical(top_n), "importances": canonical(importances.head(20))}
    c["model/pub/output_output_random_forest_GLB"] = _output_output

    def _filtered_iomap():
        df = _pub_mapping_df().copy()
        cutoff = df["Value"].quantile(0.7)
        df["in_constraint_range"] = df["Value"] > cutoff
        _, importances, top_n = FilteredInputOutputMapping(
            df, "GLB", "2C", 2050).random_forest()
        return {"top_n": canonical(top_n), "importances": canonical(importances.head(20))}
    c["model/pub/filtered_iomap_random_forest_GLB"] = _filtered_iomap

    # ---- figures -----------------------------------------------------------
    # NewTimeSeries stores its figure on .figure, while every other class here
    # uses .fig. The app only ever pulls .return_traces() off it, so nothing
    # caught the inconsistency.
    def _timeseries():
        band = DataRetrieval(db(PUBLICATION), "GDP_billion_USD2007", "GLB", "2C").single_output_df_to_graph(5, 95)
        series = NewTimeSeries("GDP_billion_USD2007", "GLB", "2C", 2050, band)
        return {
            "figure": summarize_figure(series.figure),
            "n_traces_from_return_traces": len(series.return_traces()),
        }
    c["figure/pub/timeseries_gdp_GLB_2C"] = _timeseries

    # OutputHistograms labels publication scenarios via publication_scenario_display_names.
    def _histograms_pub():
        fig = OutputHistograms("GDP_billion_USD2007", ["GLB", "USA"], ["2C", "Ref"], 2050,
                               db(PUBLICATION), styling_options={"color": "by-region"})
        return summarize_figure(fig.make_plot())
    c["figure/pub/histograms_gdp"] = _histograms_pub

    def _histograms_full():
        fig = OutputHistograms("GDP_billion_USD2007", ["GLB", "USA"], ["2C_med", "Ref"], 2050,
                               db(FULL), styling_options={"color": "by-region"})
        return summarize_figure(fig.make_plot())
    c["figure/full/histograms_gdp"] = _histograms_full

    def _choropleth():
        frame = DataRetrieval(db(PUBLICATION), "GDP_billion_USD2007", "GLB", "2C", 2050).choropleth_map_df(5, 95)
        return summarize_figure(
            ChoroplethMap(frame, "GDP_billion_USD2007", "2C", 2050, 5, 95).fig)
    c["figure/pub/choropleth_gdp_2C_2050"] = _choropleth

    c["figure/pub/iomap_GLB_2C_2050"] = lambda: summarize_figure(
        InputOutputMappingPlot(PUB_EMISSIONS, "GLB", "2C", 2050, _pub_mapping_df()).fig)

    def _tsclust_fig():
        df = DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C").single_output_df()
        return summarize_figure(TimeSeriesClusteringPlot(df, PUB_EMISSIONS, "GLB", "2C").fig)
    c["figure/pub/timeseries_clustering_GLB"] = _tsclust_fig

    def _tsclust_cart_fig():
        df = DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C").single_output_df()
        return summarize_figure(TimeSeriesClusteringPlotCART(df, PUB_EMISSIONS, "GLB", "2C").fig)
    c["figure/pub/timeseries_clustering_cart_GLB"] = _tsclust_cart_fig

    def _cluster_outputs_fig():
        from eppa_viz.figures.cluster_output_parcoords import (
            ClusterOutputParcoordsPlot,
            RENEWABLE_SHARE_SPEC,
        )
        share = json.dumps(RENEWABLE_SHARE_SPEC)
        df = DataRetrieval(db(PUBLICATION), share, "GLB", "Ref").single_output_df()
        plot = ClusterOutputParcoordsPlot(
            db(PUBLICATION), df, share, "GLB", "Ref", year=2100,
        )
        return summarize_figure(plot.fig)
    c["figure/pub/cluster_output_parcoords_GLB_Ref"] = _cluster_outputs_fig

    # PlotTree only returns its figure from make_plot(); it never assigns .fig.
    # This also pins the density and coverage numbers on every node, which were
    # being computed from the caller's y until recently.
    def _plot_tree():
        model = InputOutputMapping(PUB_EMISSIONS, "GLB", "2C", 2050, _pub_mapping_df()).CART()
        tree = PlotTree(model)
        root = tree.build_tree_from_CART(tree.fit_tree_model)

        def walk(node):
            if node is None:
                return None
            return {
                "feature": node.feature,
                "threshold": canonical(node.threshold),
                "density": canonical(node.density),
                "coverage": canonical(node.coverage),
                "left": walk(node.left),
                "right": walk(node.right),
            }

        return {"figure": summarize_figure(tree.make_plot()), "nodes": walk(root)}
    c["figure/pub/plot_tree_GLB_2C_2050"] = _plot_tree

    c["figure/pub/permutation_importance_GLB"] = lambda: summarize_figure(
        PermutationImportance(_pub_mapping_df(), PUB_EMISSIONS, "GLB", "2C", 2050).fig)

    # ---- figures after styling --------------------------------------------
    # These pin the styling contract that P3 is meant to unify.
    c["figure/styled/iomap_GLB_2C_2050"] = lambda: _styled(
        InputOutputMappingPlot(PUB_EMISSIONS, "GLB", "2C", 2050, _pub_mapping_df()))

    def _styled_choropleth():
        frame = DataRetrieval(db(PUBLICATION), "GDP_billion_USD2007", "GLB", "2C", 2050).choropleth_map_df(5, 95)
        return _styled(ChoroplethMap(frame, "GDP_billion_USD2007", "2C", 2050, 5, 95))
    c["figure/styled/choropleth_gdp_2C_2050"] = _styled_choropleth

    def _styled_tsclust():
        df = DataRetrieval(db(PUBLICATION), PUB_EMISSIONS, "GLB", "2C").single_output_df()
        return _styled(TimeSeriesClusteringPlot(df, PUB_EMISSIONS, "GLB", "2C"))
    c["figure/styled/timeseries_clustering_GLB"] = _styled_tsclust

    # Styled time series: figure type and .fig alias are unified in NewTimeSeries.
    def _styled_timeseries():
        band = DataRetrieval(db(PUBLICATION), "GDP_billion_USD2007", "GLB", "2C").single_output_df_to_graph(5, 95)
        return _styled(NewTimeSeries("GDP_billion_USD2007", "GLB", "2C", 2050, band))
    c["figure/styled/timeseries_gdp_GLB_2C"] = _styled_timeseries

    def _styled_custom_variable():
        df = DataRetrieval(db(PUBLICATION), RENEWABLE_SHARE_PUBLICATION, "GLB", "2C", 2050).mapping_df()
        return _styled(InputOutputMappingPlot(RENEWABLE_SHARE_PUBLICATION, "GLB", "2C", 2050, df))
    c["figure/styled/iomap_custom_renewable_share"] = _styled_custom_variable

    return c
