"""Validate a migrated dataset against the original Excel workbooks.

    python scripts/validate_against_excel.py --dataset publication

The SQL-to-SQL check in migrate_schema.py proves the copy was faithful. It
cannot prove the original ingest was, and it demonstrably was not: the USA
sheet of the CO2 emissions workbook was ingested with the 2015 base-year value
in place of the run number, which went unnoticed for as long as the data has
existed. This compares the fact table back to the workbooks that produced it.

Every value is compared, not a sample. Workbooks are parsed in parallel across
cores and reduced to per-series aggregates, and MySQL aggregates its side with
GROUP BY, so only a few thousand rows cross the wire rather than 133 million.

Comparison runs at FLOAT precision. The workbooks hold full doubles while the
database stores FLOAT, so an exact comparison would flag all 19,532 series.

Some series cannot be checked against a workbook, and each falls back to the
matching Cleaned Data CSV, which is what was actually ingested for them:

  - 4_elec_prod_Gas_CCS_TWh_Ref.xlsx is a structurally corrupt xlsx. The zip
    is missing [Content_Types].xml, so no reader can open it.
  - 8_carbon_price_USD2007_per_ton_CO2e_Ref.xlsx is entirely blank, which is
    correct: a reference scenario has no carbon price, and the database holds
    zeros throughout.
  - 4_elec_prod_Gas_CCS_TWh_2C.xlsx begins at 2030 and omits 2020 and 2025.
    The CSV has both, correctly aligned with the workbook everywhere they
    overlap, so the database is more complete than that workbook rather than
    disagreeing with it.
"""

import argparse
import glob
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import mysql.connector
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from styling import Options  # noqa: E402

WORKBOOK_ROOT = {
    "publication": "Raw Data/Archive",
    "all_data_aug_2024": "Raw Data/New_Ensembles",
}

# Leading index and trailing scenario are stripped to recover the output name,
# e.g. "5_total_emissions_CO2_million_ton_CO2_Ref.xlsx".
FILENAME = re.compile(r"^\d+_(?P<rest>.+)\.xlsx$")

# Years the database holds. The USA sheet of one workbook carries a stray 2015
# column that no sibling sheet has and that was never meant to be ingested.
YEARS = set(range(2020, 2101, 5))


def float32_sum(values):
    """Sum a series as the database would, having stored it as FLOAT.

    Each value is narrowed to float32 first, then accumulated in float64 to
    match SUM(CAST(value AS DOUBLE)) on the MySQL side.
    """
    return float(np.asarray(values, dtype=np.float32).astype(np.float64).sum())


def scenarios_for(dataset):
    return sorted(["Ref", "2C"] if dataset == "publication" else Options().scenarios,
                  key=len, reverse=True)


def parse_filename(path, scenarios):
    match = FILENAME.match(os.path.basename(path))
    if not match:
        return None
    rest = match.group("rest")
    for scenario in scenarios:
        if rest.endswith("_" + scenario):
            return rest[: -len(scenario) - 1], scenario
    return None


def aggregate_workbook(job):
    """Reduce one workbook to {(output, region, scenario, year): (count, sum)}.

    Aggregating per year rather than per series means a workbook that covers
    fewer years than the database is reported as a coverage gap instead of a
    value mismatch, and a year-shifted ingest cannot hide inside a whole-series
    total.
    """
    path, output, scenario = job
    result = {}
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as exc:  # noqa: BLE001
        return path, None, f"{type(exc).__name__}: {exc}"

    unexpected = set()
    for region, frame in sheets.items():
        if region == "Data Note" or frame.empty:
            continue
        for column in frame.columns:
            if not str(column).isdigit() or int(column) not in YEARS:
                continue
            # The New_Ensembles workbooks carry GAMS "Eps" markers for values
            # rounded to nothing. The ingest maps them to 0, so this must too.
            raw = frame[column].replace("Eps", 0)
            values = pd.to_numeric(raw, errors="coerce")
            unexpected |= set(raw[values.isna() & raw.notna()].astype(str).unique())
            values = values.dropna()
            if len(values):
                result[(output, region, scenario, int(column))] = (
                    len(values), float32_sum(values))
    if unexpected:
        return path, result, f"non-numeric values ignored: {sorted(unexpected)[:5]}"
    return path, result, None


def load_csv_fallbacks(excel, jobs):
    """Aggregate Cleaned Data CSVs for any series a workbook could not supply.

    Keyed the same way as the Excel results, so the comparison downstream does
    not care which source a series came from.
    """
    wanted = {(output, scenario) for _, output, scenario in jobs}
    have = {(output, scenario) for output, _, scenario, _ in excel}
    fallback = {}
    for output, scenario in sorted(wanted - have):
        path = os.path.join("Cleaned Data", scenario, f"{output}.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["Year"].isin(YEARS)]
        for (region, year), group in frame.groupby(["Region", "Year"]):
            values = group["Value"].dropna()
            if len(values):
                fallback[(output, region, scenario, int(year))] = (
                    len(values), float(sum(as_float32(float(v)) for v in values)))
    return fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(WORKBOOK_ROOT))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="relative tolerance on the per-series sum")
    args = parser.parse_args()

    scenarios = scenarios_for(args.dataset)
    jobs, skipped = [], []
    for path in sorted(glob.glob(f"{WORKBOOK_ROOT[args.dataset]}/**/*.xlsx", recursive=True)):
        if os.path.basename(path).startswith("~$"):
            continue
        parsed = parse_filename(path, scenarios)
        if parsed is None:
            skipped.append(path)
            continue
        jobs.append((path, parsed[0], parsed[1]))

    print(f"=== {args.dataset} ===")
    print(f"workbooks: {len(jobs)}   unrecognized filenames: {len(skipped)}")
    for path in skipped[:5]:
        print(f"   skipped {path}")

    print(f"\nparsing every sheet across {args.workers} workers ...")
    started = time.time()
    excel = {}
    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, result, error in pool.map(aggregate_workbook, jobs):
            done += 1
            if error:
                failures.append((path, error))
            if result:
                excel.update(result)
            if done % 100 == 0 or done == len(jobs):
                print(f"   {done}/{len(jobs)} workbooks  {time.time() - started:.0f}s")
    print(f"  parsed {len(excel)} series-years in {(time.time() - started)/60:.1f} min")
    if failures:
        print(f"  {len(failures)} workbook(s) needed attention:")
        for path, error in failures[:5]:
            print(f"    {os.path.basename(path)}: {error}")

    csv_sourced = load_csv_fallbacks(excel, jobs)
    if csv_sourced:
        outputs = sorted({k[0] for k in csv_sourced})
        print(f"  {len(csv_sourced)} series taken from Cleaned Data CSVs instead: "
              f"{', '.join(outputs)}")
        excel.update(csv_sourced)

    print("\naggregating the database side ...")
    started = time.time()
    connection = mysql.connector.connect(
        host="localhost", user="root",
        password=os.environ.get("MYSQL_PWD", "password"), database=args.dataset)
    cursor = connection.cursor()
    # NULLs are excluded so both sides count the same thing. A blank cell in a
    # workbook becomes a NULL row here, and whole sheets are blank for regions
    # an output does not apply to, so counting NULLs would report agreement as
    # a difference.
    cursor.execute("""
        SELECT o.name, r.code, s.code, sv.year, COUNT(sv.value),
               SUM(CAST(sv.value AS DOUBLE))
        FROM series_values sv
        JOIN outputs   o ON o.output_id   = sv.output_id
        JOIN regions   r ON r.region_id   = sv.region_id
        JOIN scenarios s ON s.scenario_id = sv.scenario_id
        WHERE sv.value IS NOT NULL
        GROUP BY o.name, r.code, s.code, sv.year""")
    database = {(o, r, s, int(y)): (n, float(total or 0.0))
                for o, r, s, y, n, total in cursor.fetchall()}
    print(f"  {len(database)} series-years in {time.time() - started:.1f}s")
    cursor.close()
    connection.close()

    missing_from_db = sorted(set(excel) - set(database))
    # A year the database holds that the source does not is a coverage gap in
    # the source, not an error, so it is reported separately from real
    # disagreements.
    absent_from_source = sorted(set(database) - set(excel))
    shared = sorted(set(excel) & set(database))

    count_diffs, value_diffs = [], []
    for key in shared:
        (e_count, e_sum), (d_count, d_sum) = excel[key], database[key]
        if e_count != d_count:
            count_diffs.append((key, e_count, d_count))
        scale = max(abs(e_sum), abs(d_sum), 1e-12)
        if abs(e_sum - d_sum) / scale > args.tolerance:
            value_diffs.append((key, e_sum, d_sum, abs(e_sum - d_sum) / scale))

    def name(key):
        return f"{key[0]}_{key[1]}_{key[2]} @{key[3]}"

    print(f"\n{'':4}series-years compared      : {len(shared)}")
    print(f"{'':4}in source but not database : {len(missing_from_db)}")
    print(f"{'':4}in database but not source : {len(absent_from_source)}")
    print(f"{'':4}row-count differences      : {len(count_diffs)}")
    print(f"{'':4}value differences          : {len(value_diffs)}  "
          f"(tolerance {args.tolerance:g})")

    for key in missing_from_db[:10]:
        print(f"      MISSING FROM DB  {name(key)}")
    for key, e, d in count_diffs[:10]:
        print(f"      ROW COUNT  {name(key)}: source {e} vs db {d}")
    for key, e, d, rel in sorted(value_diffs, key=lambda x: -x[3])[:10]:
        print(f"      VALUES     {name(key)}: source {e:.6f} vs db {d:.6f} (rel {rel:.2e})")

    if absent_from_source:
        gaps = defaultdict(set)
        for key in absent_from_source:
            gaps[(key[0], key[2], key[1])].add(key[3])
        print("\n    in the database but not the source "
              "(source coverage gaps, not errors):")
        for (output, scenario, region), years in sorted(gaps.items())[:20]:
            print(f"      {output} [{scenario}] {region}: {sorted(years)}")
        if len(gaps) > 20:
            print(f"      ... and {len(gaps) - 20} more region/scenario combinations")

    problems = len(missing_from_db) + len(count_diffs) + len(value_diffs)
    by_output = defaultdict(int)
    for key, *_ in count_diffs + value_diffs:
        by_output[key[0]] += 1
    if by_output:
        print("\n    outputs with real disagreements:")
        for output, n in sorted(by_output.items(), key=lambda x: -x[1]):
            print(f"      {n:>6} series-years  {output}")

    print(f"\n{'PASS' if not problems else 'DIFFERENCES FOUND'}: {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
