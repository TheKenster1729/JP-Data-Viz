"""Plot and tab callbacks for the Dash dashboard."""
import json
import io

import dash
from dash import dcc, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import product

from sql_utils import DataRetrieval, MultiOutputRetrieval
from styling import Readability, Color, FinishedFigure
from figure import (
    NewTimeSeries,
    InputOutputMappingPlot,
    ChoroplethMap,
    TimeSeriesClusteringPlot,
    OutputOutputMappingPlot,
    PlotTree,
    RegionalHeatmaps,
    InputDistributionAlternate,
    PermutationImportance,
    FilteredOutputOutputMappingPlot,
    FilteredInputOutputMappingPlot,
    TimeSeriesClusteringPlotCART,
    ModifyOutputTimeseries,
)
from analysis import InputOutputMapping
from eppa_viz.webapp.constants import CITATION_TEXT
from eppa_viz.webapp.connections import (
    db_full,
    db_publication,
    options_obj,
    readability_obj,
    database_for,
)


def _default_bounds(lower_bound, upper_bound):
    return (
        lower_bound if lower_bound is not None else 5,
        upper_bound if upper_bound is not None else 95,
    )


def register_plot_callbacks(app):

    # callback for output time series
    @app.callback(
        Output('output-time-series-plot', 'figure'),
        [Input('output-dropdown', 'value'),
         Input('region-dropdown', 'value'),
         Input('scenario-dropdown', 'value'),
        #  Input('chart-options', 'value'),
        #  Input('year-slider', 'value'),
         Input('output-color-scheme', 'value'),
         Input("time-series-plot-apply-bound-changes", "n_clicks"),
         Input("time-series-plot-apply-styling-changes", "n_clicks")],
        [State('output-time-series-plot', 'figure'),
         State("time-series-plot-upper-bound", "value"),
         State("time-series-plot-lower-bound", "value"),
         State("time-series-plot-color-picker", "value"),
         State("time-series-plot-toggle-gridlines", "checked"),
         State("overview-data-dropdown", "value")]
    )
    # add back in commented out inputs when ready to re-incorporate
    def update_timeseries_graph(output_name, selected_regions, selected_scenarios, color_scheme, n_clicks_bound_changes, 
                                n_clicks_styling_changes, existing_figure, upper_bound, lower_bound, plot_bgcolor, toggle_gridlines, overview_data_dropdown):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]
        
        if not selected_regions or not selected_scenarios:
            raise PreventUpdate

        lower_bound, upper_bound = _default_bounds(lower_bound, upper_bound)
        db = database_for(overview_data_dropdown)
        
        if not existing_figure or len(existing_figure.get('data')) == 0:
            # if no existing figure, create a new one
            region = selected_regions[0]
            scenario = selected_scenarios[0]
            new_trace_df = DataRetrieval(db, output_name, region, scenario).single_output_df_to_graph(lower_bound, upper_bound)
            traces_to_add = NewTimeSeries(output_name, region, scenario, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()

            fig = go.Figure(traces_to_add)
            fig.update_layout(
                height = 625,
                margin = dict(t = 40, b = 0, l = 10),
                title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name]),
                yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
                xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
                plot_bgcolor = plot_bgcolor,
                uirevision = output_name,
                datarevision = str(id(fig))
            )
            fig.update_xaxes(showgrid = toggle_gridlines)
            fig.update_yaxes(showgrid = toggle_gridlines)

            return fig.to_dict()

        else:
            existing_fig = go.Figure(existing_figure)
            # if the existing figure is a histogram, then we need to start from scratch
            # otherwise, the new time series is added to the histogram figure
            if existing_fig.data[0]["type"] == "histogram":
                fig = ModifyOutputTimeseries(output_name, selected_regions, selected_scenarios, go.Figure(), {"color": color_scheme}, db, lower_bound, upper_bound, True).create_new_figure()

                fig.update_layout(
                    height = 625,
                    margin = dict(t = 40, b = 0, l = 10),
                    title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name]),
                    yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
                    xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
                    plot_bgcolor = plot_bgcolor,
                    uirevision = output_name,
                    datarevision = str(id(fig))
                )
                fig.update_xaxes(showgrid = toggle_gridlines)
                fig.update_yaxes(showgrid = toggle_gridlines)

                return fig.to_dict()

            # the figure will only change if the regions/scenarios/output has changed
            # to make sure the figure is changed when styling options are changed, check if any of the styling options have changed
            if trigger_id == "output-color-scheme" or trigger_id == "time-series-plot-apply-styling-changes" or trigger_id == "time-series-plot-apply-bound-changes":
                change_fig = True
            else:
                change_fig = False
            
            # When output dropdown changes, we MUST start with a fresh figure
            # This is because UIDs won't match between different outputs (especially custom vs regular)
            # and ModifyOutputTimeseries.remove_traces() won't be able to remove old traces
            if trigger_id == "output-dropdown":
                existing_fig = go.Figure()
                change_fig = True
            
            fig = ModifyOutputTimeseries(output_name, selected_regions, selected_scenarios, existing_fig, {"color": color_scheme}, db, lower_bound, upper_bound, change_fig).create_new_figure()
            if db == db_full:
                if output_name not in readability_obj.naming_dict_long_names_first:
                    just_name = json.loads(output_name)["name"]
                else:
                    just_name = readability_obj.naming_dict_long_names_first[output_name]
            else:
                if output_name not in readability_obj.publication_naming_dict_long_names_first:
                    just_name = json.loads(output_name)["name"]
                else:
                    just_name = readability_obj.publication_naming_dict_long_names_first[output_name]
            title_text = "Time Series for " + just_name
            fig.update_layout(
                height = 625,
                margin = dict(t = 40, b = 0, l = 10),
                title_text = title_text,
                yaxis = dict(title = dict(text = just_name, font = dict(size = 16))),
                xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
                plot_bgcolor = plot_bgcolor,
                hovermode = 'closest',  # Ensure interactivity is preserved
                # uirevision forces Plotly.js to completely reset the figure when output changes
                # This prevents stale renders when switching from custom variables to regular outputs
                uirevision = output_name,
                # Add timestamp to force complete re-render (diagnostic)
                datarevision = str(id(fig))
            )
            fig.update_xaxes(showgrid = toggle_gridlines)
            fig.update_yaxes(showgrid = toggle_gridlines)

            # Convert to dict to ensure clean serialization to browser
            return fig.to_dict()

            # this code describes a histogram that was previously available, but was removed to keep things simple
            # current_trace_info = TraceInfo(existing_figure)
            # if current_trace_info.type[0] == "histogram": # means active figure is histogram, so need to generate scatter 
            #     for region, scenario in product(selected_regions, selected_scenarios):
            #         new_trace_df = DataRetrieval(db, output_name, region, scenario).single_output_df_to_graph(lower_bound, upper_bound)
            #         traces_to_add = NewTimeSeries(output_name, region, scenario, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()

            #     try:
            #         title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name])
            #     except KeyError:
            #         title_text = "Time Series for {}".format(json.loads(output_name)["name"])

            #     fig = go.Figure(traces_to_add)
            #     fig.update_layout(
            #         height = 625,
            #         margin = dict(t = 40, b = 0, l = 10),
            #         title_text = title_text,
            #         # yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
            #         xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
            #         plot_bgcolor = plot_bgcolor
            #     )
            #     fig.update_xaxes(showgrid = toggle_gridlines)
            #     fig.update_yaxes(showgrid = toggle_gridlines)
            #     return fig
            # else:
            #     combos_with_trace_name = list(product(selected_regions, selected_scenarios, ["lower", "median", "upper"]))
            #     current_traces = current_trace_info.traces
            #     custom_data_just_strings = [i[0] for i in current_trace_info.custom_data]
            #     existing_selections = set(custom_data_just_strings)
            #     all_selections = set(["{}|{}|{}|{}".format(output_name, reg, sce, trace_name) for reg, sce, trace_name in combos_with_trace_name])

            #     # changes to make
            #     if trigger_id == "output-color-scheme":
            #         # without this logic, the color of the figure will not update when the color scheme is changed
            #         # what this does is take all existing plots and changes color according to what the new color scheme dictates
            #         existing_figure_data = existing_figure["data"]
            #         for i in existing_figure_data:
            #             trace_name = i["customdata"][0]
            #             region = trace_name.split(' ')[-3]
            #             scenario = trace_name.split(' ')[-2]
            #             if color_scheme == "by-region":
            #                 color = Color().get_color_for_timeseries(color_scheme, region)
            #             elif color_scheme == "by-scenario":
            #                 color = Color().get_color_for_timeseries(color_scheme, scenario)
            #             elif color_scheme == "standard":
            #                 color = Color().get_color_for_timeseries(color_scheme, [region, scenario])

            #             i["line"]["color"] = color

            #     no_change = existing_selections.intersection(all_selections)
            #     to_delete = existing_selections.difference(all_selections)
            #     to_add = all_selections.difference(existing_selections)

            #     # removing traces - well, keeping ones that haven't been removed
            #     indices_to_delete = [custom_data_just_strings.index(i) for i in to_delete]
            #     indices_to_keep = [i for i in range(len(current_traces)) if i not in indices_to_delete]
            #     current_traces = [current_traces[i] for i in indices_to_keep]

            #     # adding traces
            #     new_traces = []
            #     decomposed_traces_to_add = set([i.split("|")[0] + "|" + i.split("|")[1] + "|" + i.split("|")[2] for i in to_add])
            #     for i in decomposed_traces_to_add:
            #         output, reg, sce = tuple(i.split("|"))
            #         new_trace_df = DataRetrieval(db, output_name, reg, sce).single_output_df_to_graph(lower_bound, upper_bound)
            #         traces_to_add = NewTimeSeries(output_name, reg, sce, 2050, new_trace_df, styling_options = {"color": color_scheme}).return_traces()
            #         new_traces += traces_to_add

            #     if output_name not in options_obj.outputs:
            #         title_text = "Time Series for " + json.loads(output_name)["name"]
            #     else:
            #         title_text = "Time Series for {}".format(readability_obj.naming_dict_long_names_first[output_name])

            #     fig = go.Figure(data = current_traces + new_traces)
            #     fig.update_layout(
            #         height = 625,
            #         margin = dict(t = 40, b = 0, l = 10),
            #         title_text = title_text,
            #         # yaxis = dict(title = dict(text = readability_obj.naming_dict_long_names_first[output_name], font = dict(size = 16))),
            #         xaxis = dict(title = dict(text = "Year", font = dict(size = 16))),
            #         plot_bgcolor = plot_bgcolor
            #     )
            #     fig.update_xaxes(showgrid = toggle_gridlines)
            #     fig.update_yaxes(showgrid = toggle_gridlines)
            #     return fig

        # else:
        #     if not selected_regions or not selected_scenarios:
        #         raise PreventUpdate

        #     styling_options = {"color": color_scheme}
        #     fig = OutputHistograms(output_name, selected_regions, selected_scenarios, year, db, styling_options = styling_options).make_plot()

        #     if output_name not in options_obj.outputs:
        #         title_text = "Histograms for " + output_name.split("-")[-1]
        #     else:
        #         title_text = "Histograms for {}".format(readability_obj.naming_dict_long_names_first[output_name])
        #     fig.update_layout(title_text = title_text)
        #     fig.update_layout(
        #         height = 550,
        #         margin = dict(t = 70, b = 20, l = 10)
        #     ) - doing this separately helps make everything more organized
    @app.callback(
        Output("time-series-plot-download-csv", "data"),
        Input("time-series-plot-download-data-button", "n_clicks"),
        State('output-dropdown', 'value'),
        State('region-dropdown', 'value'),
        State('scenario-dropdown', 'value'),
        State("time-series-plot-upper-bound", "value"),
        State("time-series-plot-lower-bound", "value"),
        State("overview-data-dropdown", "value"),
        prevent_initial_call=True
    )
    def timeseries_data_download(n_clicks, output, regions, scenarios, upper_bound, lower_bound, overview_data_dropdown):
        if not output or not regions or not scenarios or not n_clicks:
            raise PreventUpdate

        lower_bound, upper_bound = _default_bounds(lower_bound, upper_bound)
        db = database_for(overview_data_dropdown)

        full_data_df = pd.DataFrame()
        for reg in regions:
            for sce in scenarios:
                band = DataRetrieval(db, output, reg, sce).single_output_df_to_graph(
                    lower_bound, upper_bound
                )
                df = band.reset_index()
                df["Region"] = reg
                df["Scenario"] = sce
                full_data_df = pd.concat([full_data_df, df], axis=0)
        
        # Create a safe filename (handle custom variables)
        if output.startswith('{'):
            filename = "eppa_dashboard_data_custom_variable.csv"
        else:
            filename = f"eppa_dashboard_data_{output}.csv"
        
        return dcc.send_data_frame(full_data_df.to_csv, filename, index=False)

    # callback for high-res image download
    @app.callback(
        Output("time-series-plot-download-image", "data"),
        Input("time-series-plot-download-image-button", "n_clicks"),
        State('output-time-series-plot', 'figure'),
        State('output-dropdown', 'value'),
        prevent_initial_call=True
    )
    def timeseries_plot_image_download(n_clicks, figure_data, output):
        if not n_clicks or not figure_data:
            raise PreventUpdate
        
        figure = go.Figure(figure_data)
        
        # Create a safe filename
        if output and output.startswith('{'):
            filename = "time_series_custom_variable.png"
        else:
            filename = f"time_series_{output}.png" if output else "time_series_plot.png"

        return dcc.send_bytes(figure.to_image(format="png", scale=3), filename)

    # callback for SVG download
    @app.callback(
        Output("time-series-plot-download-svg", "data"),
        Input("time-series-plot-download-svg-button", "n_clicks"),
        State('output-time-series-plot', 'figure'),
        State('output-dropdown', 'value'),
        prevent_initial_call=True
    )
    def timeseries_plot_svg_download(n_clicks, figure_data, output):
        if not n_clicks or not figure_data:
            raise PreventUpdate
        
        figure = go.Figure(figure_data)
        
        # Create a safe filename
        if output and output.startswith('{'):
            filename = "time_series_custom_variable.svg"
        else:
            filename = f"time_series_{output}.svg" if output else "time_series_plot.svg"

        return dcc.send_bytes(figure.to_image(format="svg"), filename)

    # Citation download callbacks - triggered alongside main downloads
    @app.callback(
        Output("time-series-plot-download-citation-csv", "data"),
        Input("time-series-plot-download-data-button", "n_clicks"),
        prevent_initial_call=True
    )
    def download_citation_with_csv(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return dcc.send_string(CITATION_TEXT, "suggested_citation.txt")

    @app.callback(
        Output("time-series-plot-download-citation-image", "data"),
        Input("time-series-plot-download-image-button", "n_clicks"),
        prevent_initial_call=True
    )
    def download_citation_with_image(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return dcc.send_string(CITATION_TEXT, "suggested_citation.txt")

    @app.callback(
        Output("time-series-plot-download-citation-svg", "data"),
        Input("time-series-plot-download-svg-button", "n_clicks"),
        prevent_initial_call=True
    )
    def download_citation_with_svg(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return dcc.send_string(CITATION_TEXT, "suggested_citation.txt")

    # callback for inputs
    @app.callback(
        Output("input-dist-graph", "figure"),
        Input("input-dist-options", "value"))
    def update_input_dist(inputs):
        if not inputs:
            raise PreventUpdate
        figure = InputDistributionAlternate(inputs).make_plot()

        return figure

    # callback for i/o mapping
    @app.callback(
        Output("input-output-mapping-figure-container", "hidden"),
        Output("input-output-mapping-figure", "figure"),
        Output("input-output-mapping-output", "disabled"),
        Output("input-output-mapping-percentile", "disabled"),
        Output("input-output-mapping-setting", "disabled"),
        Output("input-output-mapping-run-count", "children"),
        State("input-output-mapping-output", "value"),
        State("input-output-mapping-region", "value"),
        State("input-output-mapping-scenario", "value"),
        State("input-output-mapping-year", "value"),    
        State("input-output-mapping-percentile", "value"),
        State("input-output-mapping-setting", "value"),
        State("input-output-mapping-n-estimators", "value"),
        State("input-output-mapping-max-depth", "value"),
        Input("input-output-mapping-mode", "value"),  # Input so mode change triggers callback
        State("slider-custom-io-mapping-1", "value"),
        State("slider-custom-io-mapping-2", "value"),
        State("slider-custom-io-mapping-3", "value"),
        State("custom-io-mapping-dropdown-1", "value"),
        State("custom-io-mapping-dropdown-2", "value"),
        State("custom-io-mapping-dropdown-3", "value"),
        Input("input-output-mapping-update-all-settings", "n_clicks"),
        State("overview-data-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_io_mapping_figure(output, region, scenario, year, percentile, setting, n_estimators, max_depth, mode, slider_1, slider_2, slider_3, dropdown_1, dropdown_2, dropdown_3, update_all_settings, publication_output):
        if not region or not output or not year or not scenario:
            raise PreventUpdate
        
        if publication_output == "full":
            db = db_full
        else:
            db = db_publication

        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]
        gt = True if setting == "above" else False

        # If only the mode changed, just update UI visibility without running analysis
        if trigger_id == "input-output-mapping-mode":
            if mode == "standard":
                # Show standard UI, hide filtered UI
                return True, dash.no_update, False, False, False, ""
            else:
                # Show filtered UI, hide standard UI elements
                return False, dash.no_update, True, True, True, ""

        if mode == "standard":
            df = DataRetrieval(db, output, region, scenario, year).mapping_df()
            unstyled_figure = InputOutputMappingPlot(output, region, scenario, year, df, threshold = percentile, gt = gt, n_estimators = n_estimators, max_depth = max_depth)
            finished_figure = FinishedFigure(unstyled_figure).make_finished_figure()

            return True, finished_figure, False, False, False, ""

        if mode == "filtered":
            outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
            if not outputs_to_include:
                raise PreventUpdate
                
            df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()
            
            # Apply percentile constraints to filter runs
            constraint_df = df.copy()
            constraint_df["in_constraint_range"] = 1  # Initialize all rows as within constraint range
            
            # Iterate through each dropdown/slider pair to apply constraints
            sliders = [slider_1, slider_2, slider_3]
            for i, (dropdown, slider) in enumerate(zip([dropdown_1, dropdown_2, dropdown_3], sliders)):
                if dropdown and slider:  # Ensure dropdown has a selection
                    col_name = readability_obj.naming_dict_long_names_first[dropdown]
                    lower_bound, upper_bound = np.percentile(df[col_name], slider)
                    constraint_df["in_constraint_range"] &= (
                        (constraint_df[col_name] >= lower_bound) & 
                        (constraint_df[col_name] <= upper_bound)
                    ).astype(int)
            
            # Count runs that meet all constraints
            total_runs = len(constraint_df)
            selected_runs = constraint_df["in_constraint_range"].sum()
            run_count_text = f"Selected {selected_runs} of {total_runs} runs based on percentile constraints"
            
            # Generate the feature importance plot
            fig = FilteredInputOutputMappingPlot(
                constraint_df, region, scenario, year, 
                n_estimators=n_estimators, random_forest_depth=max_depth
            ).make_plot()

            return False, fig, True, True, True, run_count_text

    # callback for i/o tree
    @app.callback(
        Output("full-cart-tree", "figure"), 
        State("input-output-mapping-output", "value"),
        State("input-output-mapping-region", "value"),
        State("input-output-mapping-scenario", "value"),
        State("input-output-mapping-year", "value"),
        State("input-output-mapping-mode", "value"),
        State("full-cart-tree-depth-dropdown", "value"),
        State("slider-custom-io-mapping-1", "value"),
        State("slider-custom-io-mapping-2", "value"),
        State("slider-custom-io-mapping-3", "value"),
        State("custom-io-mapping-dropdown-1", "value"),
        State("custom-io-mapping-dropdown-2", "value"),
        State("custom-io-mapping-dropdown-3", "value"),
        State("overview-data-dropdown", "value"),
        State("input-output-mapping-percentile", "value"),
        State("input-output-mapping-setting", "value"),
        Input("input-output-mapping-update-all-settings", "n_clicks"),
        prevent_initial_call = True
    )
    def update_tree(output, region, scenario, year, mode, cart_depth, slider_1, slider_2, slider_3, dropdown_1, dropdown_2, dropdown_3, publication_output, percentile, setting, update_all_settings_n_clicks):
        if not cart_depth or not output or not region or not scenario or not year:
            raise PreventUpdate
        
        if publication_output == "full":
            db = db_full
        else:
            db = db_publication

        gt = True if setting == "above" else False

        if mode == "standard":
            df = DataRetrieval(db, output, region, scenario, year).mapping_df()
            _, y = InputOutputMapping(output, region, scenario, year, df, threshold = percentile, gt = gt, cart_depth = cart_depth).preprocess_for_classification()
            tree = InputOutputMapping(output, region, scenario, year, df, threshold = percentile, gt = gt, cart_depth = cart_depth).CART()
            fig = PlotTree(tree, y).make_plot(show = False)

            return fig
        
        if mode == "filtered":
            outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
            if not outputs_to_include:
                return go.Figure()
                
            df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()

            constraint_df = df.copy()
            constraint_df["in_constraint_range"] = 1  # Initialize all rows as within constraint range
            
            # Iterate through each dropdown/slider pair to apply constraints
            for dropdown, slider in zip([dropdown_1, dropdown_2, dropdown_3], [slider_1, slider_2, slider_3]):
                if dropdown and slider:  # Ensure dropdown has a selection
                    lower_bound, upper_bound = np.percentile(df[readability_obj.naming_dict_long_names_first[dropdown]], slider)
                    constraint_df["in_constraint_range"] &= ((constraint_df[readability_obj.naming_dict_long_names_first[dropdown]] >= lower_bound) & (constraint_df[readability_obj.naming_dict_long_names_first[dropdown]] <= upper_bound)).astype(int)

            filtered_mapping = FilteredInputOutputMappingPlot(constraint_df, region, scenario, year, cart_depth = cart_depth)
            tree = filtered_mapping.CART()
            fig = PlotTree(tree, filtered_mapping.y_discrete).make_plot(show = False)

            return fig

    # callback for permutation importance
    @app.callback(
        Output("input-output-mapping-permutation-importance", "figure"),
        State("input-output-mapping-output", "value"),
        State("input-output-mapping-region", "value"),
        State("input-output-mapping-scenario", "value"),
        State("input-output-mapping-year", "value"),
        State("input-output-mapping-n-estimators", "value"),
        State("input-output-mapping-max-depth", "value"),
        State("input-output-mapping-mode", "value"),
        State("slider-custom-io-mapping-1", "value"),
        State("slider-custom-io-mapping-2", "value"),
        State("slider-custom-io-mapping-3", "value"),
        State("custom-io-mapping-dropdown-1", "value"),
        State("custom-io-mapping-dropdown-2", "value"),
        State("custom-io-mapping-dropdown-3", "value"),
        Input("input-output-mapping-update-all-settings", "n_clicks"),
        State("overview-data-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_permutation_importance(output, region, scenario, year, n_estimators, max_depth, mode, slider_1, slider_2, slider_3, dropdown_1, dropdown_2, dropdown_3, update_all_settings, publication_output):
        if not region or not output or not year or not scenario:
            raise PreventUpdate
        
        if publication_output == "full":
            db = db_full
        else:
            db = db_publication
        
        if mode == "standard":
            df = DataRetrieval(db, output, region, scenario, year).mapping_df()
            unstyled_figure = PermutationImportance(df, output, region, scenario, year, n_estimators = n_estimators, max_depth = max_depth)
            finished_figure = FinishedFigure(unstyled_figure).make_finished_figure()
            return finished_figure
        
        if mode == "filtered":
            outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
            if not outputs_to_include:
                return go.Figure()
                
            df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()
            
            # Apply percentile constraints to filter runs
            constraint_df = df.copy()
            constraint_df["in_constraint_range"] = 1
            
            for dropdown, slider in zip([dropdown_1, dropdown_2, dropdown_3], [slider_1, slider_2, slider_3]):
                if dropdown and slider:
                    col_name = readability_obj.naming_dict_long_names_first[dropdown]
                    lower_bound, upper_bound = np.percentile(df[col_name], slider)
                    constraint_df["in_constraint_range"] &= (
                        (constraint_df[col_name] >= lower_bound) & 
                        (constraint_df[col_name] <= upper_bound)
                    ).astype(int)
            
            # Use FilteredInputOutputMapping for permutation importance
            from analysis import FilteredInputOutputMapping
            filtered_mapping = FilteredInputOutputMapping(
                constraint_df, region, scenario, year, 
                n_estimators=n_estimators, random_forest_depth=max_depth
            )
            results = filtered_mapping.permutation_importance()
            
            # Create figure from results
            fig = go.Figure()
            if results:
                fig.add_trace(go.Bar(
                    x=[k["variable"] for k in results], 
                    y=[k["mean"] for k in results], 
                    error_y=dict(type="data", array=[k["std"] for k in results])
                ))
                fig.update_layout(
                    title="Permutation Importance (Filtered Mode)",
                    xaxis_title="Feature",
                    yaxis_title="Importance"
                )
            
            return fig

    # callback for o/o mapping
    @app.callback(
        Output("output-output-mapping-figure-container", "hidden"),
        Output("output-output-mapping-figure", "figure"),
        Output("output-output-mapping-output", "multi"),
        Output("output-output-mapping-output", "options"),
        Output("output-output-mapping-output", "disabled"),
        Output("output-output-mapping-run-count", "children"),
        Input("output-output-mapping-mode", "value"),  # Keep as Input so mode switching is immediate
        Input("output-output-mapping-update", "n_clicks"),  # Update button triggers the callback
        State("output-output-mapping-output", "value"),
        State("output-output-mapping-region", "value"),
        State("output-output-mapping-scenario", "value"),
        State("output-output-mapping-year", "value"),
        State("custom-oo-mapping-dropdown-1", "value"),
        State("custom-oo-mapping-dropdown-2", "value"),
        State("custom-oo-mapping-dropdown-3", "value"),
        State("slider-custom-oo-mapping-1", "value"),
        State("slider-custom-oo-mapping-2", "value"),
        State("slider-custom-oo-mapping-3", "value"),
        State("output-output-mapping-output", "options"),
        State("overview-data-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_output_output_mapping(mode, update_clicks, output, region, scenario, year, dropdown_1, dropdown_2, dropdown_3, slider_1, slider_2, slider_3, options, publication_output):
        if not region or not output or not scenario or not year:
            raise PreventUpdate
        
        if publication_output == "full":
            db = db_full
        else:
            db = db_publication

        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]

        # If only the mode changed, just update UI visibility without running analysis
        if trigger_id == "output-output-mapping-mode":
            if mode == "standard":
                # Enable dropdown in standard mode
                return True, dash.no_update, False, dash.no_update, False, ""
            else:
                # Disable dropdown in filtered mode (all outputs are used)
                return False, dash.no_update, True, dash.no_update, True, ""

        if mode == "standard":
            df = DataRetrieval(db, output, region, scenario, year).mapping_df()
            fig = OutputOutputMappingPlot(db, output, region, scenario, year, df)
            finished_fig = FinishedFigure(fig).make_finished_figure()

            return True, finished_fig, False, options, False, ""

        if mode == "filtered":
            # Fetch the outputs selected for constraint filtering
            outputs_to_include = [dropdown for dropdown in [dropdown_1, dropdown_2, dropdown_3] if dropdown]
            if not outputs_to_include:
                raise PreventUpdate
            
            df = MultiOutputRetrieval(db, outputs_to_include, region, scenario, year).construct_df()

            # Apply constraints based on percentile sliders
            constraint_df = df.copy()
            constraint_df["in_constraint_range"] = 1

            for dropdown, slider in zip([dropdown_1, dropdown_2, dropdown_3], [slider_1, slider_2, slider_3]):
                if dropdown:
                    output_name = readability_obj.naming_dict_long_names_first[dropdown] if dropdown in Options().outputs else json.loads(dropdown)["name"]
                    lower_bound, upper_bound = np.percentile(df[output_name], slider)
                    constraint_df["in_constraint_range"] &= ((constraint_df[output_name] >= lower_bound) & (constraint_df[output_name] <= upper_bound)).astype(int)

            # Count runs that meet constraints
            runs_selected = constraint_df["in_constraint_range"].sum()
            total_runs = len(constraint_df)
            run_count_text = f"Selected {runs_selected} of {total_runs} runs based on percentile constraints."

            # FilteredOutputOutputMappingPlot now uses ALL outputs as inputs, 
            # with in_constraint_range as the target variable
            fig = FilteredOutputOutputMappingPlot(db, constraint_df, region, scenario, year).make_plot()

            if options[0]["value"] == "all":
                new_options = options
            else:
                new_options = [{"label": "All", "value": "all"}] + options
            
            # Disable dropdown in filtered mode (all outputs are used)
            return False, fig, True, new_options, True, run_count_text
        
        # this mode has been deprecated, but I'm leaving the code here for now in case we need it in the future
        # the purpose of this mode was to look at the upper and lower ranges of the output variables, but that's 
        # just a more specific case of the filtered mode
        # if mode == "high-low":
        #     df_upper = DataRetrieval(db, output, region, scenario, year).mapping_df()
        #     df_lower = df_upper.copy()

            
        #     fig = OutputOutputMappingPlot(db, output, region, scenario, year, df)
        #     finished_fig = FinishedFigure(fig).make_finished_figure()

        #     return go.Figure(), True, True, finished_fig, False, options

        #     return filter_fig, False, False, fig, True, options

    # callback for regional heatmaps
    @app.callback(Output("regional-heatmaps-figure", "figure"),
                  Input("regional-heatmaps-apply-button", "n_clicks"),
                  State("regional-heatmaps-output", "value"),
                  State("regional-heatmaps-region", "value"),
                  State("regional-heatmaps-scenario", "value"),
                  State("overview-data-dropdown", "value"),
                  prevent_initial_call = True)
    def update_regional_heatmaps_figure(n_clicks, output, regions, scenarios, publication_output):
        """
        Generate regional heatmaps showing feature importance.
        All data fetching and processing logic is encapsulated in RegionalHeatmaps class.
        """
        if not regions or not output or not scenarios:
            raise PreventUpdate
        
        db = db_full if publication_output == "full" else db_publication
        
        # RegionalHeatmaps handles all data fetching and model training internally
        fig = RegionalHeatmaps(db, output, regions, scenarios).fig

        return fig

    # callback for choropleth mapping
    @app.callback(
        Output("choropleth-mapping-figure", "figure"),
        Input("choropleth-mapping-update", "n_clicks"),
        State("choropleth-mapping-output", "value"),
        State("choropleth-mapping-scenario", "value"),
        State("choropleth-mapping-year", "value"),
        State("overview-data-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_choropleth_figure(n_clicks, output, scenario, year, publication_output):
        if not n_clicks or not scenario or not output or not year:
            raise PreventUpdate
        
        if publication_output == "full":
            db = db_full
        else:
            db = db_publication
        
        df = DataRetrieval(db, output, "GLB", scenario, year).choropleth_map_df(5, 95)
        unstyled_fig = ChoroplethMap(df, output, scenario, year, 5, 95)
        finished_fig = FinishedFigure(unstyled_fig).make_finished_figure()

        return finished_fig

    # callback for ts clustering
    @app.callback(
        Output("ts-clustering-plot", "figure"),
        Output('ts-clustering-random-forest-plot', 'figure'),
        # Output("ts-clustering-cart-tree-plot", "figure"),
        Input("ts-clustering-update", "n_clicks"),
        State("ts-clustering-output", "value"),
        State("ts-clustering-region", "value"),
        State("ts-clustering-scenario", "value"),
        State("ts-clustering-n-clusters", "value"),
        State("ts-clustering-metric", "value"),
        State("overview-data-dropdown", "value"),
        prevent_initial_call = True
    )
    def update_ts_clustering_figure(n_clicks, output, region, scenario, n_clusters, metric, publication_output):
        if not n_clicks or not region or not output or not scenario:
            raise PreventUpdate
        
        if publication_output == "full":
            db = db_full
        else:
            db = db_publication

        df = DataRetrieval(db, output, region, scenario).single_output_df()
        dataset_name = "all_data_aug_2024" if publication_output == "full" else "publication"
        fig_obj = TimeSeriesClusteringPlot(
            df, output, region, scenario,
            n_clusters=n_clusters, metric=metric, dataset_name=dataset_name,
        )

        cart_fig_obj = TimeSeriesClusteringPlotCART(
            df, output, region, scenario,
            n_clusters=n_clusters, metric=metric, dataset_name=dataset_name,
        )
        finished_figure = FinishedFigure(fig_obj).make_finished_figure()
        finished_figure_cart = FinishedFigure(cart_fig_obj).make_finished_figure()

        return finished_figure, finished_figure_cart

    # callback for dynamic display of custom variables tab
    @app.callback(
        Output("custom-vars-output-dropdown-div", "children"),
        Output("custom-vars-operation", "value"),
        Output("custom-vars-var-name", "value"),
        Input("custom-vars-operation", "value"),
        Input("close-centered", "n_clicks"),
        State("output-dropdown", "options"),
        State("custom-vars-var-name", "value")
    )
    def dynamic_custom_variables_fill(operation_type, n_clicks, options, var_name):
        if not operation_type:
            raise PreventUpdate

        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split('.')[0]

        if trigger_id == "custom-vars-operation":
            if operation_type == "division":
                    return [(
                        dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                            placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                        html.Span("by", style = {'margin-right': '10px'}),
                        dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", options = options,
                                    placeholder = "Output 2",
                                    style = {"width": "500px"})
                            ), operation_type, var_name]
            if operation_type == "addition":
                return [(dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                            placeholder = "Select Outputs to Add", style = {"margin-right": "10px", "width": "1000px"}, multi = True),
                            dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", style = {"display": "none"})), operation_type, var_name]
            if operation_type == "multiplication":
                    return [(
                        dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                            placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                        dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", options = options,
                                    placeholder = "Output 2",
                                    style = {"width": "500px"})
                            ), operation_type, var_name]
            if operation_type == "subtraction":
                    return [(
                        dcc.Dropdown(id = "custom-vars-output-1-dropdown-div-1", options = options,
                            placeholder = "Output 1", style = {"margin-right": "10px", "width": "500px"}),
                        dcc.Dropdown(id = "custom-vars-output-2-dropdown-div-2", options = options,
                                    placeholder = "Output 2",
                                    style = {"width": "500px"})
                            ), operation_type, var_name]

        if trigger_id == "close-centered":
            return [[], "", ""]

    # callback for custom variables
    @app.callback(
        Output("stored-custom-variables", "data"),
        State("stored-custom-variables", "data"),
        State("custom-vars-var-name", "value"),
        State("custom-vars-output-1-dropdown-div-1", "value"),
        State("custom-vars-output-2-dropdown-div-2", "value"),
        State("custom-vars-operation", "value"),
        Input("create-custom-variable-button", "n_clicks"),
        prevent_initial_call = True
    )
    def update_custom_variables(current_data, var_name, output_1, output_2, operation, n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        current_data = current_data or {}
        if operation == "division":
            current_data[var_name] = {"operation": operation, "output1": output_1, "output2": output_2, "name": var_name}
        if operation == "addition":
            current_data[var_name] = {"operation": operation, "outputs": output_1, "name": var_name}
        if operation == "multiplication":
            current_data[var_name] = {"operation": operation, "output1": output_1, "output2": output_2, "name": var_name}
        if operation == "subtraction":
            current_data[var_name] = {"operation": operation, "output1": output_1, "output2": output_2, "name": var_name}

        return current_data

    # callback for modal pop-up on custom variables tab
    @app.callback(
        Output("custom-variable-created-modal", "is_open"),
        Input("create-custom-variable-button", "n_clicks"), 
        Input("close-centered", "n_clicks"),
        State("custom-variable-created-modal", "is_open"),
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open


