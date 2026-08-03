"""Fetch many outputs at one year through the repository."""

from styling import Options, Readability


def option_output_list(dbname):
    """Output ids the webapp lists for this database (not the union of both)."""
    if dbname == "all_data_aug_2024":
        return list(Options().outputs)
    if dbname == "publication":
        return list(Options().publication_outputs)
    raise ValueError("Invalid database name: {!r}".format(dbname))


def matrix_for_runs(repo, outputs, region, scenario, year, run_numbers,
                    exclude_output=None, display_names=False):
    """
    Wide matrix indexed by Run # with one column per output that covers every run.

    Only outputs whose year slice contains all of run_numbers are included.
    This matches the old name_mappings path, which joined on the full target run
    list rather than intersecting runs across outputs.
    """
    required = set(run_numbers)
    if exclude_output:
        outputs = [o for o in outputs if o != exclude_output]

    frames = repo.frames_at_year(outputs, region, scenario, year)
    naming = Readability().naming_dict_long_names_first if display_names else None

    columns = {}
    for output in outputs:
        frame = frames.get(output)
        if frame is None or frame.empty:
            continue
        runs_present = set(frame["Run #"].values)
        if not required.issubset(runs_present):
            continue
        subset = frame[frame["Run #"].isin(required)].set_index("Run #")["Value"]
        name = naming.get(output, output) if naming else output
        columns[name] = subset

    if not columns:
        return None

    order = list(run_numbers)
    matrix = None
    for name, series in columns.items():
        if matrix is None:
            matrix = series.to_frame(name)
        else:
            matrix[name] = series
    return matrix.reindex(order)
