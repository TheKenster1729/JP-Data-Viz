"""Dropdown callbacks that must be registered once per component id (closure-safe)."""
import json

import dash
from dash import callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from styling import Readability
from eppa_viz.webapp.connections import options_obj

OUTPUT_DROPDOWN_IDS = [
    "output-dropdown",
    "input-output-mapping-output",
    "choropleth-mapping-output",
    "ts-clustering-output",
    "output-output-mapping-output",
    "regional-heatmaps-output",
    "custom-io-mapping-dropdown-1",
    "custom-io-mapping-dropdown-2",
    "custom-io-mapping-dropdown-3",
    "custom-oo-mapping-dropdown-1",
    "custom-oo-mapping-dropdown-2",
    "custom-oo-mapping-dropdown-3",
]

SCENARIO_DROPDOWN_IDS = [
    "scenario-dropdown",
    "ts-clustering-scenario",
    "regional-heatmaps-scenario",
    "output-output-mapping-scenario",
    "input-output-mapping-scenario",
    "choropleth-mapping-scenario",
]

MULTI_SELECT_SCENARIO_IDS = {"scenario-dropdown", "regional-heatmaps-scenario"}


def register_dropdown_callbacks(app):
    for dropdown_id in OUTPUT_DROPDOWN_IDS:
        _register_output_dropdown(app, dropdown_id)

    for dropdown_id in SCENARIO_DROPDOWN_IDS:
        _register_scenario_dropdown(app, dropdown_id)


def _register_output_dropdown(app, dropdown_id):
    @app.callback(
        Output(dropdown_id, "options", allow_duplicate=True),
        Output(dropdown_id, "value"),
        State(dropdown_id, "options"),
        Input("stored-custom-variables", "data"),
        Input("create-custom-variable-button", "n_clicks"),
        Input("custom-vars-operation", "value"),
        Input("overview-data-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_output_dropdowns(
        current_options, current_stored_data, n_clicks, operation, publication_output
    ):
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "overview-data-dropdown":
            if publication_output == "full":
                options = [
                    {"label": j, "value": i}
                    for i, j in Readability().naming_dict_long_names_first.items()
                ]
            else:
                options = [
                    {"label": j, "value": i}
                    for i, j in Readability().publication_naming_dict_long_names_first.items()
                ]
            if current_stored_data:
                custom_variable_options = [
                    {"label": i, "value": json.dumps(j)}
                    for i, j in current_stored_data.items()
                ]
                options = options + custom_variable_options
            return options, options[0]["value"]
        if n_clicks is None or not current_stored_data:
            raise PreventUpdate
        updated_options = current_options.copy()
        for custom_var_name, custom_var_data in current_stored_data.items():
            all_current_labels = [x["label"] for x in updated_options]
            if custom_var_name not in all_current_labels:
                value_to_use = json.dumps(custom_var_data)
                updated_options.append(
                    {"label": custom_var_name, "value": value_to_use}
                )
        return updated_options, dash.no_update


def _register_scenario_dropdown(app, dropdown_id):
    @app.callback(
        Output(dropdown_id, "options"),
        Output(dropdown_id, "value"),
        Input("overview-data-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_scenario_dropdowns(publication_output):
        if publication_output == "full":
            options = [
                {"label": k, "value": v}
                for k, v in options_obj.scenario_display_names_rev.items()
            ]
        else:
            options = [
                {"label": "Reference", "value": "Ref"},
                {"label": "2C", "value": "2C"},
            ]
        if dropdown_id in MULTI_SELECT_SCENARIO_IDS:
            return options, [options[0]["value"]]
        return options, options[0]["value"]
