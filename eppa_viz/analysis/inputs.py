"""InputsMasterTFP loading and region-specific column pruning."""

import pandas as pd

_INPUTS_CACHE = None

INPUTS_PATH = "Cleaned Data/InputsMasterTFP.csv"


def get_inputs_df():
    """Return a copy of the cached inputs spreadsheet."""
    global _INPUTS_CACHE
    if _INPUTS_CACHE is None:
        _INPUTS_CACHE = pd.read_csv(INPUTS_PATH)
    return _INPUTS_CACHE.copy()


def filter_inputs_by_region(inputs_df, region):
    """
    Drop TFP/Pop and AEEI columns that do not apply to this region.

    GLB keeps only aggregate TFP/Pop and all AEEI columns. Other regions keep
    {REG} and Non-{REG} TFP/Pop and only AEEI {REG}.
    """
    if region == "GLB":
        allowed_tfp_pop = {"TFP", "Pop"}
    else:
        allowed_tfp_pop = {
            f"{region} TFP",
            f"Non-{region} TFP",
            f"{region} Pop",
            f"Non-{region} Pop",
        }

    columns_to_drop = []
    for column in inputs_df.columns:
        if column in {"Run #", "\ufeffRun #"}:
            continue

        if ("TFP" in column) or (" Pop" in column) or (column == "Pop"):
            if column not in allowed_tfp_pop:
                columns_to_drop.append(column)
            continue

        if column.startswith("AEEI "):
            if region != "GLB" and column != f"AEEI {region}":
                columns_to_drop.append(column)

    return inputs_df.drop(columns=columns_to_drop)


def prepare_inputs(region, scenario, output=None, drop_runs_map=None):
    """Cached inputs, region filter, and optional scenario-specific run drops."""
    inputs = filter_inputs_by_region(get_inputs_df(), region)
    if output and drop_runs_map:
        drop_runs = drop_runs_map.get(output)
        if drop_runs:
            runs_to_drop = drop_runs.get(scenario)
            if runs_to_drop:
                inputs = inputs.drop(
                    inputs[inputs["Run #"].isin(runs_to_drop)].index)
    return inputs
