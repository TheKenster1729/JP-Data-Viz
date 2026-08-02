"""Repair two data defects in the publication database.

    python scripts/repair_publication_data.py --check
    python scripts/repair_publication_data.py --apply

Both were found by migrating to a schema with a primary key and validating
every value against the source workbooks. Neither could have been noticed in
the old layout, which has no keys, no constraints and one table per series.

Both repairs are idempotent, and both fix the legacy per-series tables so the
migration stays a faithful copy rather than a copy that silently corrects.

1. total_emissions_CO2_million_ton_CO2_USA_Ref had every run numbered 4681.

   The USA sheet of Raw Data/Archive/Ref/5_total_emissions_CO2_million_ton_CO2_Ref.xlsx
   carries a stray 2015 base-year column that no other sheet in that workbook
   has: 19 columns against 18. The original ingest addressed columns by
   position rather than by header, so it read the 2015 value as the run number.
   Every run shares that value, 4681.242924, so all 6,800 rows collapsed onto
   run 4681.

   The years and values came through correctly and only the run identity was
   lost, but the run number is what joins an output to its model inputs, so any
   analysis of this series was joining against meaningless labels. Rewritten
   from the workbook, taking the run from the unnamed first column and the
   years from the headers, and dropping 2015 to match the 17-year 2020-2100
   shape of the other 1,291 series.

2. elec_prod_Gas_CCS_TWh_ROE_Ref carries 400 rows for year 2015.

   These are the only pre-2020 rows in the entire dataset, they are all zero,
   and they give that one series 18 years where every other series has 17. They
   arrived through Cleaned Data/Ref/elec_prod_Gas_CCS_TWh.csv, which has the
   same stray block. Deleted so the year grid is uniform.
"""

import argparse
import os

import mysql.connector
import pandas as pd

WORKBOOK = "Raw Data/Archive/Ref/5_total_emissions_CO2_million_ton_CO2_Ref.xlsx"
SHEET = "USA"
RUN_SERIES = "total_emissions_CO2_million_ton_CO2_USA_Ref"
STRAY_YEAR_SERIES = "elec_prod_Gas_CCS_TWh_ROE_Ref"
DROP_YEARS = {"2015"}
FIRST_YEAR = 2020


def table_for(cursor, series):
    cursor.execute("SELECT `Assigned Name` FROM name_mappings WHERE `Full Output Name`=%s",
                   (series,))
    found = cursor.fetchall()
    if len(found) != 1:
        raise SystemExit(f"expected one name_mappings row for {series}, got {len(found)}")
    return found[0][0]


def shape(cursor, table):
    cursor.execute(f"SELECT COUNT(*), COUNT(DISTINCT `Run #`), MIN(`Run #`), MAX(`Run #`), "
                   f"COUNT(DISTINCT `Year`), MIN(`Year`), MAX(`Year`) FROM `{table}`")
    n, dr, r0, r1, dy, y0, y1 = cursor.fetchone()
    return f"{n} rows, {dr} runs [{r0}..{r1}], {dy} years [{y0}..{y1}]"


def repair_run_numbers(cursor, apply_changes):
    table = table_for(cursor, RUN_SERIES)
    print(f"\n1. {RUN_SERIES} -> {table}")
    print(f"   before: {shape(cursor, table)}")

    frame = pd.ExcelFile(WORKBOOK).parse(SHEET)
    runs = frame[frame.columns[0]]
    if list(runs) != list(range(1, len(runs) + 1)):
        raise SystemExit("run column in the workbook is not a clean 1..N sequence")

    years = [c for c in frame.columns if str(c).isdigit() and str(c) not in DROP_YEARS]
    rows = [(int(run), int(year), None if pd.isna(value) else float(value))
            for year in years for run, value in zip(runs, frame[year])]
    print(f"   workbook: {len(rows)} rows across {len(years)} years "
          f"({years[0]}..{years[-1]}), {'/'.join(sorted(DROP_YEARS))} dropped")

    if not apply_changes:
        return
    cursor.execute(f"DELETE FROM `{table}`")
    cursor.executemany(
        f"INSERT INTO `{table}` (`Run #`, `Year`, `Value`) VALUES (%s, %s, %s)", rows)
    print(f"   after : {shape(cursor, table)}")


def repair_stray_year(cursor, apply_changes):
    table = table_for(cursor, STRAY_YEAR_SERIES)
    print(f"\n2. {STRAY_YEAR_SERIES} -> {table}")
    print(f"   before: {shape(cursor, table)}")

    cursor.execute(f"SELECT COUNT(*), MAX(ABS(`Value`)) FROM `{table}` WHERE `Year` < %s",
                   (FIRST_YEAR,))
    count, largest = cursor.fetchone()
    print(f"   rows before {FIRST_YEAR}: {count}"
          + (f", largest absolute value {largest}" if count else ""))
    if count and largest:
        raise SystemExit("stray rows are not all zero; refusing to delete them blindly")

    if not apply_changes or not count:
        return
    cursor.execute(f"DELETE FROM `{table}` WHERE `Year` < %s", (FIRST_YEAR,))
    print(f"   after : {shape(cursor, table)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the repairs")
    parser.add_argument("--check", action="store_true", help="report state only")
    args = parser.parse_args()
    if not (args.apply or args.check):
        parser.error("pass --check or --apply")

    connection = mysql.connector.connect(
        host="localhost", user="root",
        password=os.environ.get("MYSQL_PWD", "password"), database="publication")
    cursor = connection.cursor()

    repair_run_numbers(cursor, args.apply)
    repair_stray_year(cursor, args.apply)

    if args.apply:
        connection.commit()
        print("\ncommitted. re-run scripts/migrate_schema.py to rebuild series_values.")
    else:
        print("\nno changes written.")

    cursor.close()
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
