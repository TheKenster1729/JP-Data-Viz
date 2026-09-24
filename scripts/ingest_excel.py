"""Load EPPA Excel workbooks straight into the normalized schema.

    python scripts/ingest_excel.py --dataset all_data_aug_2024 "Raw Data/New_Ensembles/2C_med" --dry-run
    python scripts/ingest_excel.py --dataset all_data_aug_2024 "Raw Data/New_Ensembles/2C_med"
    python scripts/ingest_excel.py --dataset all_data_aug_2024 path/to/one.xlsx --replace

This is the one-step path from workbooks to what the app reads. It parses
every workbook, checks it, writes series_values and the dimension tables, reads
each series back to confirm it landed intact, and registers any new output in
the display-name CSV the dropdowns are built from.

Paths may be workbooks, scenario folders, or a folder of scenario folders.
Nothing is written with --dry-run, which makes it a format check as well.

Workbook format
---------------
File name   {n}_{output name}_{scenario}.xlsx, e.g. 1_GDP_billion_USD2007_2C_med.xlsx.
            The leading number is ignored. The scenario is the database code,
            without periods (About15C_med, not About1.5C_med), and must match
            the containing folder or a scenario the dataset already has.
Sheets      One per region, named by region code (GLB, USA, ...). A sheet named
            "Data Note" is skipped; any other unknown name is skipped with a
            warning.
Row 1       Header. Year labels (2020, 2025, ...), as numbers or as text.
Column      The run number sits in the column immediately left of the first
            year. Anything further left (units, category labels) is ignored,
            so both the Archive layout (run in A) and the New_Ensembles layout
            (units in A, run in B) work.
Cells       Numbers. GAMS "Eps" becomes 0. Blank becomes NULL. Any other text
            rejects the workbook.

Only years on the app's grid (Options().years, 2020-2100 every 5) are loaded
unless --all-years is passed. Off-grid columns are reported, not silently
dropped: one Archive sheet has a stray 2015 base year that was once ingested
as the run number.

Each workbook is one transaction. Its rows are inserted, read back and compared
by count and float32 sum before commit, so a workbook either lands whole or not
at all. Existing series are left alone unless --replace is passed.

Do not re-run scripts/migrate_schema.py on a dataset after using this. That
script rebuilds series_values from the legacy per-series tables, which this one
does not write; it now refuses to run when it would discard series loaded here.
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import mysql.connector
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# styling and the display-name CSVs resolve paths against the repo root, so run
# from there; paths on the command line are still taken relative to the caller.
CALLER_CWD = os.getcwd()
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import config  # noqa: E402
from datasets import SCENARIOS  # noqa: E402
from styling import Options  # noqa: E402

REGIONS = tuple(Options().region_names)
SKIPPED_SHEETS = {"Data Note"}
LEADING_INDEX = re.compile(r"^\d+_")
YEAR_LABEL = re.compile(r"^\s*(\d{4})(\.0+)?\s*$")

# Which CSV the dropdowns read each dataset's outputs from.
DISPLAY_NAME_CSV = {
    "publication": "publication_output_names.csv",
    "all_data_aug_2024": "display_names.csv",
}

INSERT_BATCH = 20000


class WorkbookError(ValueError):
    pass


# ---- naming ------------------------------------------------------------------

def shown(path):
    """A path as the caller would recognise it."""
    relative = os.path.relpath(path, CALLER_CWD)
    return path if relative.startswith("..") else relative


def collect_workbooks(paths):
    found = []
    for path in (os.path.join(CALLER_CWD, p) for p in paths):
        if os.path.isdir(path):
            found += glob.glob(os.path.join(path, "**", "*.xlsx"), recursive=True)
        elif path.endswith(".xlsx") and os.path.exists(path):
            found.append(path)
        else:
            raise SystemExit(f"not a workbook or folder: {path}")
    # "~$" files are Excel's lock files for a workbook that is open.
    return sorted({p for p in found if not os.path.basename(p).startswith("~$")})


def name_parts(path, known_scenarios, forced_scenario=None):
    """(output, scenario) from a workbook path.

    Output names contain underscores, so the scenario can only be split off by
    matching it against a known list. The containing folder counts as known,
    with its periods removed, which is how About1.5C_med became About15C_med.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if forced_scenario:
        candidates = [forced_scenario]
    else:
        folder = os.path.basename(os.path.dirname(os.path.abspath(path))).replace(".", "")
        candidates = set(known_scenarios) | {folder}
    for scenario in sorted(candidates, key=len, reverse=True):
        if stem.endswith("_" + scenario):
            output = LEADING_INDEX.sub("", stem[: -len(scenario) - 1])
            if output:
                return output, scenario
    raise WorkbookError(
        f"cannot tell the scenario from the file name; expected it to end in "
        f"_<scenario>.xlsx with one of: {', '.join(sorted(candidates))}")


# ---- parsing (runs in worker processes) --------------------------------------

def parse_sheet(frame, years_wanted):
    """One region's sheet to (years, runs, values[year, run]) plus notes."""
    # With default NA parsing off, empty cells arrive as "" rather than NaN.
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    header = list(frame.iloc[0])
    year_columns = {}
    for index, label in enumerate(header):
        match = YEAR_LABEL.match(str(label)) if not pd.isna(label) else None
        if match:
            year = int(match.group(1))
            if year in year_columns:
                raise WorkbookError(f"year {year} appears twice in the header")
            year_columns[year] = index
    if not year_columns:
        return None, ["no year header in row 1"]

    first = min(year_columns.values())
    if first == 0:
        raise WorkbookError("the first year is in column A, leaving no column for the run number")
    notes = []
    off_grid = sorted(y for y in year_columns if years_wanted is not None and y not in years_wanted)
    if off_grid:
        notes.append(f"off-grid years not loaded: {off_grid}")
        for year in off_grid:
            del year_columns[year]
    if not year_columns:
        return None, notes + ["no on-grid years"]

    body = frame.iloc[1:]
    runs = pd.to_numeric(body.iloc[:, first - 1], errors="coerce")
    values = body.iloc[:, [year_columns[y] for y in sorted(year_columns)]]
    # Trailing rows with neither a run nor a value are formatting residue.
    blank = runs.isna() & values.isna().all(axis=1)
    runs, values = runs[~blank], values[~blank]
    if runs.isna().any():
        rows = [int(i) + 1 for i in runs.index[runs.isna()][:5]]
        raise WorkbookError(f"missing or non-numeric run number in rows {rows}")
    if (runs != runs.round()).any():
        raise WorkbookError("run numbers are not whole numbers")
    runs = runs.astype("int64")
    if runs.duplicated().any():
        raise WorkbookError(f"duplicate run numbers: {sorted(runs[runs.duplicated()].unique())[:5]}")

    values = values.replace("Eps", 0)
    numeric = values.apply(pd.to_numeric, errors="coerce")
    bad = values.notna() & numeric.isna()
    if bad.any().any():
        sample = sorted(set(values[bad].stack().astype(str)))[:5]
        raise WorkbookError(f"non-numeric cells: {sample}")

    return (sorted(year_columns), runs.to_numpy(), numeric.to_numpy(dtype="float64")), notes


def parse_workbook(job):
    """Parse one workbook. Returns a dict; never raises, so one bad file does
    not take the whole pool down."""
    path, output, scenario, years_wanted = job
    result = {"path": path, "output": output, "scenario": scenario,
              "series": {}, "notes": [], "error": None}
    try:
        # keep_default_na=False: pandas otherwise turns text such as "n/a",
        # "NA" or "#N/A" into blanks, which would load as NULL without a word.
        # Only a genuinely empty cell should.
        sheets = pd.read_excel(path, sheet_name=None, header=None, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001 - any reader failure is a bad file
        result["error"] = f"cannot open: {type(exc).__name__}: {exc}"
        return result
    try:
        for name, frame in sheets.items():
            if name in SKIPPED_SHEETS:
                continue
            if name not in REGIONS:
                result["notes"].append(f"sheet {name!r} is not a region code; skipped")
                continue
            if frame.empty:
                result["notes"].append(f"{name}: empty sheet; skipped")
                continue
            try:
                parsed, notes = parse_sheet(frame, years_wanted)
            except WorkbookError as exc:
                raise WorkbookError(f"sheet {name}: {exc}") from None
            result["notes"] += [f"{name}: {note}" for note in notes]
            if parsed is not None:
                result["series"][name] = parsed
        if not result["series"]:
            raise WorkbookError("no region sheet held any data")
    except WorkbookError as exc:
        result["error"] = str(exc)
        result["series"] = {}
    return result


# ---- database ----------------------------------------------------------------

def schema_statements():
    from migrate_schema import SCHEMA
    return [s.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1) for s in SCHEMA]


def connect(dataset, create):
    kwargs = config.connector_kwargs(dataset)
    if create:
        server = mysql.connector.connect(**{k: v for k, v in kwargs.items() if k != "database"})
        cursor = server.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{dataset}`")
        server.close()
    connection = mysql.connector.connect(**kwargs)
    if create:
        cursor = connection.cursor()
        for statement in schema_statements():
            cursor.execute(statement)
        connection.commit()
    return connection


class Catalog:
    """The dimension tables of one database, extended on demand."""

    def __init__(self, cursor, display_names):
        self.cursor = cursor
        self.display_names = display_names
        self.reload()

    def reload(self):
        self.regions = dict(self._fetch("SELECT code, region_id FROM regions"))
        self.scenarios = dict(self._fetch("SELECT code, scenario_id FROM scenarios"))
        self.outputs = dict(self._fetch("SELECT name, output_id FROM outputs"))

    def _fetch(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def ensure(self, output, scenario, regions):
        scenario_display = dict(Options().scenario_display_names)
        scenario_display.update(Options().publication_scenario_display_names)
        added = []
        for code in sorted(set(regions) - set(self.regions)):
            self.cursor.execute("INSERT INTO regions (code) VALUES (%s)", (code,))
            added.append(f"region {code}")
        if scenario not in self.scenarios:
            self.cursor.execute("INSERT INTO scenarios (code, display_name) VALUES (%s, %s)",
                                (scenario, scenario_display.get(scenario)))
            added.append(f"scenario {scenario}")
        if output not in self.outputs:
            self.cursor.execute("INSERT INTO outputs (name, display_name) VALUES (%s, %s)",
                                (output, self.display_names.get(output, output)))
            added.append(f"output {output}")
        if added:
            self.reload()
        return added

    def existing_regions(self, output, scenario):
        if output not in self.outputs or scenario not in self.scenarios:
            return set()
        rows = self._fetch(
            "SELECT DISTINCT region_id FROM series_values "
            f"WHERE output_id={int(self.outputs[output])} "
            f"AND scenario_id={int(self.scenarios[scenario])}")
        by_id = {v: k for k, v in self.regions.items()}
        return {by_id[r] for (r,) in rows}


def float32_sum(values):
    """What SUM(CAST(value AS DOUBLE)) returns once the values are stored as FLOAT."""
    values = values[~np.isnan(values)]
    return float(values.astype(np.float32).astype(np.float64).sum())


def write_workbook(cursor, catalog, parsed, regions):
    """Insert the chosen regions of one parsed workbook, then read them back.

    Raises on any disagreement so the caller can roll the workbook back.
    """
    output, scenario = parsed["output"], parsed["scenario"]
    output_id = catalog.outputs[output]
    scenario_id = catalog.scenarios[scenario]
    rows_written = 0
    for region in sorted(regions, key=lambda r: catalog.regions[r]):
        region_id = catalog.regions[region]
        years, runs, values = parsed["series"][region]
        cursor.execute("DELETE FROM series_values WHERE output_id=%s AND region_id=%s "
                       "AND scenario_id=%s", (output_id, region_id, scenario_id))

        order = np.argsort(runs, kind="stable")
        rows = [(output_id, region_id, scenario_id, year, int(run),
                 None if np.isnan(value) else float(value))
                for y, year in enumerate(years)
                for run, value in zip(runs[order], values[order, y])]
        for start in range(0, len(rows), INSERT_BATCH):
            cursor.executemany(
                "INSERT INTO series_values (output_id, region_id, scenario_id, year, run, value) "
                "VALUES (%s, %s, %s, %s, %s, %s)", rows[start:start + INSERT_BATCH])
        rows_written += len(rows)

        cursor.execute("SELECT COUNT(*), COUNT(value), COALESCE(SUM(CAST(value AS DOUBLE)), 0) "
                       "FROM series_values WHERE output_id=%s AND region_id=%s "
                       "AND scenario_id=%s", (output_id, region_id, scenario_id))
        count, non_null, total = cursor.fetchone()
        expected_total = float32_sum(values.ravel())
        scale = max(abs(expected_total), abs(total), 1e-12)
        if (count, non_null) != (len(rows), int((~np.isnan(values)).sum())) or \
                abs(expected_total - total) / scale > 1e-9:
            raise RuntimeError(
                f"{region}: read-back mismatch, wrote {len(rows)} rows summing to "
                f"{expected_total}, found {count} rows ({non_null} non-null) summing to {total}")
    return rows_written


# ---- display names -------------------------------------------------------------

def load_display_names(dataset):
    names = {}
    for path in ("publication_output_names.csv", "display_names.csv",
                 DISPLAY_NAME_CSV.get(dataset, "display_names.csv")):
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as handle:
                names.update({row["Full Output Name"]: row["Display Name"]
                              for row in csv.DictReader(handle)})
    return names


def register_display_names(dataset, outputs):
    """Append outputs the dropdown CSV lacks, named after themselves for now."""
    path = DISPLAY_NAME_CSV.get(dataset)
    if not path or not outputs:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        present = {row["Full Output Name"] for row in csv.DictReader(handle)}
    missing = sorted(set(outputs) - present)
    if missing:
        with open(path, "rb+") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) not in (b"\n", b"\r")
            else:
                needs_newline = False
        with open(path, "a", newline="", encoding="utf-8") as handle:
            if needs_newline:
                handle.write("\n")
            writer = csv.writer(handle)
            writer.writerows([(name, name) for name in missing])
    return missing


# ---- driver --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="workbooks or folders of workbooks")
    parser.add_argument("--dataset", required=True,
                        help=f"target database ({', '.join(sorted(SCENARIOS))}, or a new one "
                             "with --create-database)")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and check everything, write nothing")
    parser.add_argument("--replace", action="store_true",
                        help="overwrite series that already exist in the database")
    parser.add_argument("--scenario", help="use this scenario code for every workbook")
    parser.add_argument("--all-years", action="store_true",
                        help="load every year column, not just the app's 2020-2100 grid")
    parser.add_argument("--create-database", action="store_true",
                        help="create the database and schema if they do not exist")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    workbooks = collect_workbooks(args.paths)
    if not workbooks:
        raise SystemExit("no .xlsx workbooks found")

    connection = None
    known_scenarios = set(SCENARIOS.get(args.dataset, ()))
    if not args.dry_run or not args.create_database:
        try:
            connection = connect(args.dataset, args.create_database and not args.dry_run)
        except mysql.connector.Error as exc:
            raise SystemExit(f"cannot connect to database {args.dataset!r}: {exc}"
                             + ("" if args.create_database else
                                "\n(pass --create-database to create a new one)"))
        cursor = connection.cursor()
        catalog = Catalog(cursor, load_display_names(args.dataset))
        known_scenarios |= set(catalog.scenarios)

    print(f"=== {args.dataset}{'  (dry run)' if args.dry_run else ''} ===")
    if args.dataset not in SCENARIOS:
        print(f"note: {args.dataset!r} is not registered in datasets.py, so the app will "
              "not open it until it is added to SCENARIOS there")

    years_wanted = None if args.all_years else set(Options().years)
    jobs, failures = [], []
    for path in workbooks:
        try:
            output, scenario = name_parts(path, known_scenarios, args.scenario)
        except WorkbookError as exc:
            failures.append((path, str(exc)))
            continue
        jobs.append((path, output, scenario, years_wanted))

    duplicates = {}
    for path, output, scenario, _ in jobs:
        duplicates.setdefault((output, scenario), []).append(path)
    for (output, scenario), paths in duplicates.items():
        if len(paths) > 1:
            failures += [(p, f"{len(paths)} workbooks map to {output} [{scenario}]") for p in paths]
    clashing = {p for p, _ in failures}
    jobs = [job for job in jobs if job[0] not in clashing]

    print(f"workbooks: {len(workbooks)}   parsing across {args.workers} workers ...")
    started = time.time()
    parsed_all = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, parsed in enumerate(pool.map(parse_workbook, jobs, chunksize=4), 1):
            if parsed["error"]:
                failures.append((parsed["path"], parsed["error"]))
            else:
                parsed_all.append(parsed)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} parsed  {time.time() - started:.0f}s")

    notes = [(p["path"], n) for p in parsed_all for n in p["notes"]]
    if notes:
        print(f"\n{len(notes)} note(s):")
        for path, note in notes[:40]:
            print(f"  {shown(path)}: {note}")
        if len(notes) > 40:
            print(f"  ... and {len(notes) - 40} more")

    new_scenarios = sorted({p["scenario"] for p in parsed_all} - set(SCENARIOS.get(args.dataset, ())))
    if new_scenarios:
        print(f"\nscenarios not declared for {args.dataset} in datasets.py: "
              f"{', '.join(new_scenarios)}.\n  They load, but the dropdowns list only "
              "Options().scenarios in styling.py; add them there (and to "
              "scenario_display_names) to show them in the app.")

    written, skipped, rows_total, rolled_back = [], [], 0, []
    if args.dry_run and connection is not None:
        for parsed in parsed_all:
            existing = catalog.existing_regions(parsed["output"], parsed["scenario"])
            existing &= set(parsed["series"])
            if existing:
                skipped.append((parsed["path"], sorted(existing)))
    if not args.dry_run:
        print(f"\nwriting {len(parsed_all)} workbook(s) ...")
        started = time.time()
        for parsed in parsed_all:
            output, scenario = parsed["output"], parsed["scenario"]
            regions = set(parsed["series"])
            existing = catalog.existing_regions(output, scenario) & regions
            if existing and not args.replace:
                skipped.append((parsed["path"], sorted(existing)))
                regions -= existing
            if not regions:
                continue
            try:
                added = catalog.ensure(output, scenario, regions)
                rows = write_workbook(cursor, catalog, parsed, regions)
                connection.commit()
            except Exception as exc:  # noqa: BLE001 - roll back and keep going
                connection.rollback()
                catalog.reload()
                rolled_back.append((parsed["path"], str(exc)))
                continue
            rows_total += rows
            written.append((parsed["path"], output, scenario, len(regions), rows, added))
            print(f"  {os.path.basename(parsed['path'])}: {len(regions)} region(s), "
                  f"{rows:,} rows" + (f"  (+ {', '.join(added)})" if added else ""))
        print(f"  {rows_total:,} rows in {time.time() - started:.0f}s")

    if skipped:
        verb = "would be replaced" if args.replace else "left untouched (pass --replace to overwrite)"
        print(f"\n{len(skipped)} workbook(s) already in the database, {verb}:")
        for path, regions in skipped[:20]:
            print(f"  {shown(path)}: {len(regions)} region(s)")

    registered = []
    if not args.dry_run and written:
        new_outputs = [w[1] for w in written if any(a.startswith("output ") for a in w[5])]
        registered = register_display_names(args.dataset, new_outputs)
        if registered:
            print(f"\nadded {len(registered)} output(s) to {DISPLAY_NAME_CSV[args.dataset]} "
                  "with their raw names as display names; edit them there, and run\n"
                  f"  UPDATE outputs SET display_name=... WHERE name=...\n"
                  "to match if you want the new names in the database too:")
            for name in registered:
                print(f"  {name}")

    problems = failures + rolled_back
    if problems:
        print(f"\n{len(problems)} workbook(s) FAILED and were not loaded:")
        for path, error in problems:
            print(f"  {shown(path)}: {error}")

    print(f"\nsummary: {len(parsed_all)} parsed ok, {len(failures)} rejected"
          + ("" if args.dry_run else
             f", {len(written)} written, {len(skipped)} skipped as existing, "
             f"{len(rolled_back)} rolled back"))
    if written:
        print("restart the app to pick up the changes; the catalog is cached per process.")

    if connection is not None:
        connection.close()
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
