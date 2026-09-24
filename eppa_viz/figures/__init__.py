"""Plotly dashboard figures."""

from eppa_viz.figures.base import DashboardFigure, OUTPUT_TIMESERIES
from eppa_viz.figures.choropleth import ChoroplethMap
from eppa_viz.figures.clustering import TimeSeriesClusteringPlot, TimeSeriesClusteringPlotCART
from eppa_viz.figures.cluster_output_parcoords import ClusterOutputParcoordsPlot
from eppa_viz.figures.distributions import (
    InputDistribution,
    InputDistributionAlternate,
    OutputDistribution,
)
from eppa_viz.figures.heatmaps import RegionalHeatmaps
from eppa_viz.figures.multi_region_rf_heatmap import PublicationMultiRegionRFHeatmap
from eppa_viz.figures.histograms import OutputHistograms
from eppa_viz.figures.mapping_plots import (
    FilteredInputOutputMappingPlot,
    FilteredOutputOutputMappingPlot,
    InputOutputMappingPlot,
    OutputOutputMappingPlot,
)
from eppa_viz.figures.permutation import PermutationImportance
from eppa_viz.figures.pipeline import apply_finished_style
from eppa_viz.figures.timeseries import ModifyOutputTimeseries, NewTimeSeries, OldTimeSeries
from eppa_viz.figures.tree import PlotTree, TreeNode
from eppa_viz.figures.utils import TraceInfo, sanitize_uid

__all__ = [
    "DashboardFigure",
    "OUTPUT_TIMESERIES",
    "apply_finished_style",
    "ChoroplethMap",
    "FilteredInputOutputMappingPlot",
    "FilteredOutputOutputMappingPlot",
    "InputDistribution",
    "InputDistributionAlternate",
    "InputOutputMappingPlot",
    "ModifyOutputTimeseries",
    "NewTimeSeries",
    "OldTimeSeries",
    "OutputDistribution",
    "OutputHistograms",
    "OutputOutputMappingPlot",
    "PermutationImportance",
    "PlotTree",
    "RegionalHeatmaps",
    "PublicationMultiRegionRFHeatmap",
    "TimeSeriesClusteringPlot",
    "TimeSeriesClusteringPlotCART",
    "ClusterOutputParcoordsPlot",
    "TraceInfo",
    "TreeNode",
    "sanitize_uid",
]
