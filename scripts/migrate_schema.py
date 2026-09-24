"""Migrate a dataset from one-table-per-series to a normalized schema.

    python scripts/migrate_schema.py --dataset publication
    python scripts/migrate_schema.py --dataset publication --validate-only

The source layout stores every (output, region, scenario) series in its own
randomly-named table, with a name_mappings table translating
"{output}_{region}_{scenario}" to the table name. That means thousands of
tables, no indexes, and a full scan of name_mappings for every lookup.

This builds three dimension tables and one fact table alongside the originals.
Nothing is dropped: the old tables stay until the new ones are signed off.

Runs against one database at a time and never opens a second connection, so
publication and all_data_aug_2024 cannot be mixed. They share 209 output keys
holding different data, so that separation is load-bearing.

The copy is INSERT ... SELECT, which stays inside the server and preserves
float bits exactly. That matters because mysqldump does not: it writes FLOAT
as six significant digits.

A note for anything querying series_values: always constrain output_id, with
an equality or an IN list, so the leading column of the primary key is bound.
Leaving it open forces a full scan. Fetching all 34 outputs for one region,
scenario and year takes 7 ms with "output_id IN (...)" and 1,311 ms without.
The same applies to region_id when sweeping regions for a choropleth: 3.7 ms
with an IN list against 41.9 ms without.
"""

import argparse
import os
import sys
import time

import mysql.connector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from styling import Options, Readability  # noqa: E402

# The scenario vocabulary differs per dataset, and output names contain
# underscores, so "{output}_{region}_{scenario}" cannot be split without
# knowing the region and scenario lists up front.
SCENARIOS_BY_DATASET = {
    "publication": ["Ref", "2C"],
    "all_data_aug_2024": Options().scenarios,
}

SCHEMA = [
    """
    CREATE TABLE regions (
        region_id    TINYINT UNSIGNED NOT NULL AUTO_INCREMENT,
        code         VARCHAR(8)  NOT NULL,
        PRIMARY KEY (region_id),
        UNIQUE KEY uq_regions_code (code)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE scenarios (
        scenario_id  TINYINT UNSIGNED NOT NULL AUTO_INCREMENT,
        code         VARCHAR(32) NOT NULL,
        display_name VARCHAR(64) NULL,
        PRIMARY KEY (scenario_id),
        UNIQUE KEY uq_scenarios_code (code)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE outputs (
        output_id    SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
        name         VARCHAR(160) NOT NULL,
        display_name VARCHAR(255) NULL,
        PRIMARY KEY (output_id),
        UNIQUE KEY uq_outputs_name (name)
    ) ENGINE=InnoDB
    """,
    # Clustered on the access pattern rather than a surrogate id. Fetching one
    # series for one year, and one series across all years, are both prefix
    # range scans. Year precedes run because every query filters year and none
    # filters run.
    #
    # Deliberately no foreign keys. MySQL requires an index on each referencing
    # column, and since the primary key does not begin with region_id or
    # scenario_id it silently creates single-column indexes on both. Those two
    # columns have 19 and 2 distinct values across millions of rows, so the
    # indexes are useless for filtering but attractive enough that the optimizer
    # picks an index_merge over them instead of the primary key: 20.3 ms against
    # 2.6 ms for a full series. They also cost space and insert time on a table
    # this size. Referential integrity is enforced instead by the repository,
    # which resolves every id through the dimension tables, and by the orphan
    # check in validate().
    """
    CREATE TABLE series_values (
        output_id   SMALLINT UNSIGNED NOT NULL,
        region_id   TINYINT  UNSIGNED NOT NULL,
        scenario_id TINYINT  UNSIGNED NOT NULL,
        year        SMALLINT UNSIGNED NOT NULL,
        run         SMALLINT UNSIGNED NOT NULL,
        value       FLOAT NULL,
        PRIMARY KEY (output_id, region_id, scenario_id, year, run)
    ) ENGINE=InnoDB
    """,
]

NEW_TABLES = ["series_values", "outputs", "regions", "scenarios"]


def connect(dataset):
    return mysql.connector.connect(
        host="localhost", user="root",
        password=os.environ.get("MYSQL_PWD", "password"), database=dataset)


def parse_series_keys(cursor, dataset):
    """Split every name_mappings key into (output, region, scenario, table)."""
    regions = set(Options().region_names)
    scenarios = SCENARIOS_BY_DATASET[dataset]
    # longest first, so "2C_med" is matched before a bare "2C" could shadow it
    scenarios = sorted(scenarios, key=len, reverse=True)

    cursor.execute("SELECT `Full Output Name`, `Assigned Name` FROM name_mappings")
    parsed, unparsed = [], []
    for full_name, table in cursor.fetchall():
        for scenario in scenarios:
            if not full_name.endswith("_" + scenario):
                continue
            head = full_name[: -len(scenario) - 1]
            output, _, region = head.rpartition("_")
            if region in regions and output:
                parsed.append((output, region, scenario, table, full_name))
                break
        else:
            unparsed.append(full_name)
    return parsed, unparsed


def create_schema(cursor):
    for table in NEW_TABLES:
        cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
    for statement in SCHEMA:
        cursor.execute(statement)


def populate_dimensions(cursor, parsed, dataset):
    readability = Readability()
    display = dict(readability.publication_naming_dict_long_names_first)
    display.update(readability.naming_dict_long_names_first)
    scenario_display = Options().scenario_display_names
    scenario_display.update(Options().publication_scenario_display_names)

    regions = sorted({p[1] for p in parsed})
    scenarios = sorted({p[2] for p in parsed})
    outputs = sorted({p[0] for p in parsed})

    cursor.executemany("INSERT INTO regions (code) VALUES (%s)", [(r,) for r in regions])
    cursor.executemany("INSERT INTO scenarios (code, display_name) VALUES (%s, %s)",
                       [(s, scenario_display.get(s)) for s in scenarios])
    cursor.executemany("INSERT INTO outputs (name, display_name) VALUES (%s, %s)",
                       [(o, display.get(o)) for o in outputs])

    def ids(table, key, value):
        cursor.execute(f"SELECT {key}, {value} FROM {table}")
        return {k: v for k, v in cursor.fetchall()}

    return (ids("regions", "code", "region_id"),
            ids("scenarios", "code", "scenario_id"),
            ids("outputs", "name", "output_id"))


def copy_series(cursor, parsed, region_ids, scenario_ids, output_ids):
    """Copy every source table into the fact table, server-side.

    Series are processed in primary-key order and each is inserted sorted, so
    the clustered index is built by appending rather than by splitting pages.
    """
    ordered = sorted(parsed, key=lambda p: (output_ids[p[0]], region_ids[p[1]],
                                            scenario_ids[p[2]]))
    total_rows, started, last_report = 0, time.time(), time.time()
    for i, (output, region, scenario, table, _) in enumerate(ordered, 1):
        cursor.execute(
            "INSERT INTO series_values (output_id, region_id, scenario_id, year, run, value) "
            f"SELECT %s, %s, %s, `Year`, `Run #`, `Value` FROM `{table}` "
            "WHERE `Year` IS NOT NULL AND `Run #` IS NOT NULL "
            "ORDER BY `Year`, `Run #`",
            (output_ids[output], region_ids[region], scenario_ids[scenario]))
        total_rows += cursor.rowcount
        if time.time() - last_report > 5 or i == len(ordered):
            rate = total_rows / max(time.time() - started, 1e-9)
            print(f"  {i:>6}/{len(ordered)} series  {total_rows:>12,} rows  "
                  f"{rate/1000:,.0f}k rows/s")
            last_report = time.time()
    return total_rows, time.time() - started


def validate(cursor, parsed, region_ids, scenario_ids, output_ids):
    """Compare every source table against its slice of the fact table.

    Order-independent aggregates, summed over the raw number rather than a text
    rendering: MySQL renders FLOAT as six significant digits, which would hide
    exactly the precision loss worth catching.

    The source is cast through FLOAT before summing. 57 of the publication
    tables store Value as DOUBLE while the other 1,235 use FLOAT, an artifact of
    those two outputs having been ingested from CSV rather than Excel. The fact
    table standardizes on FLOAT, so casting the source the same way compares
    like with like and keeps a genuine copy error distinguishable from that
    deliberate narrowing.
    """
    print(f"\nvalidating {len(parsed)} series against their source tables ...")
    started, mismatches, narrowed = time.time(), [], []
    for output, region, scenario, table, full_name in parsed:
        cursor.execute(
            "SELECT COUNT(*), SUM(CAST(CAST(`Value` AS FLOAT) AS DOUBLE)), "
            "SUM(`Year`), SUM(`Run #`), COUNT(DISTINCT `Run #`), COUNT(DISTINCT `Year`) "
            f"FROM `{table}` WHERE `Year` IS NOT NULL AND `Run #` IS NOT NULL")
        src = cursor.fetchone()
        cursor.execute(
            "SELECT COUNT(*), SUM(CAST(value AS DOUBLE)), "
            "SUM(year), SUM(run), COUNT(DISTINCT run), COUNT(DISTINCT year) "
            "FROM series_values WHERE output_id=%s AND region_id=%s AND scenario_id=%s",
            (output_ids[output], region_ids[region], scenario_ids[scenario]))
        dst = cursor.fetchone()
        if src != dst:
            mismatches.append((full_name, src, dst))

        cursor.execute("SELECT column_type FROM information_schema.columns "
                       "WHERE table_schema=DATABASE() AND table_name=%s AND column_name='Value'",
                       (table,))
        column_type = cursor.fetchone()[0]
        if isinstance(column_type, (bytes, bytearray)):
            column_type = column_type.decode()
        if column_type != "float":
            narrowed.append(full_name)

    # Stands in for the foreign keys the schema deliberately omits.
    for column, table_name, key in (("output_id", "outputs", "output_id"),
                                    ("region_id", "regions", "region_id"),
                                    ("scenario_id", "scenarios", "scenario_id")):
        cursor.execute(f"SELECT COUNT(*) FROM series_values sv "
                       f"LEFT JOIN {table_name} d ON d.{key} = sv.{column} "
                       f"WHERE d.{key} IS NULL")
        orphans = cursor.fetchone()[0]
        if orphans:
            mismatches.append((f"orphan {column}", orphans, 0))
            print(f"    ORPHANS: {orphans} rows reference a missing {table_name} row")

    print(f"  {len(parsed) - len(mismatches)}/{len(parsed)} series match "
          f"({time.time() - started:.1f}s)")
    if narrowed:
        outputs = sorted({n.rsplit('_', 2)[0] for n in narrowed})
        print(f"  {len(narrowed)} series narrowed DOUBLE -> FLOAT for consistency "
              f"with the other {len(parsed) - len(narrowed)}: {', '.join(outputs)}")
    for name, src, dst in mismatches[:10]:
        print(f"    MISMATCH {name}\n      source {src}\n      target {dst}")
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(SCENARIOS_BY_DATASET))
    parser.add_argument("--validate-only", action="store_true",
                        help="re-check an existing migration without rebuilding it")
    parser.add_argument("--discard-unmapped", action="store_true",
                        help="rebuild even though series_values holds series that "
                             "name_mappings does not, deleting them")
    args = parser.parse_args()

    connection = connect(args.dataset)
    cursor = connection.cursor()
    print(f"=== {args.dataset} ===")

    parsed, unparsed = parse_series_keys(cursor, args.dataset)
    print(f"series in name_mappings : {len(parsed) + len(unparsed)}")
    print(f"  parsed                : {len(parsed)}")
    print(f"  UNPARSED              : {len(unparsed)}")
    for name in unparsed[:10]:
        print(f"    {name}")
    if unparsed:
        print("\nrefusing to migrate with unparsed keys; every series must map to "
              "an (output, region, scenario) triple.")
        return 1

    if args.validate_only:
        region_ids, scenario_ids, output_ids = (
            {c: i for c, i in _fetch(cursor, "SELECT code, region_id FROM regions")},
            {c: i for c, i in _fetch(cursor, "SELECT code, scenario_id FROM scenarios")},
            {c: i for c, i in _fetch(cursor, "SELECT name, output_id FROM outputs")})
    else:
        unmapped = unmapped_series(cursor, parsed)
        if unmapped and not args.discard_unmapped:
            print(f"\nrefusing to rebuild: series_values holds {len(unmapped)} series with "
                  "no legacy table, most likely loaded by scripts/ingest_excel.py. "
                  "Rebuilding would delete them. For example:")
            for output, region, scenario in unmapped[:10]:
                print(f"    {output} {region} {scenario}")
            print("pass --discard-unmapped to rebuild anyway.")
            return 1
        print("\nbuilding schema ...")
        create_schema(cursor)
        region_ids, scenario_ids, output_ids = populate_dimensions(cursor, parsed, args.dataset)
        connection.commit()
        print(f"  {len(region_ids)} regions, {len(scenario_ids)} scenarios, "
              f"{len(output_ids)} outputs")

        print(f"\ncopying {len(parsed)} series ...")
        rows, elapsed = copy_series(cursor, parsed, region_ids, scenario_ids, output_ids)
        connection.commit()
        print(f"  copied {rows:,} rows in {elapsed/60:.1f} min")

    mismatches = validate(cursor, parsed, region_ids, scenario_ids, output_ids)

    cursor.execute("""SELECT ROUND((data_length+index_length)/1024/1024,1)
                      FROM information_schema.tables
                      WHERE table_schema=%s AND table_name='series_values'""",
                   (args.dataset,))
    size = cursor.fetchone()[0]
    print(f"\nseries_values on disk: {size} MB")

    cursor.close()
    connection.close()
    return 1 if mismatches else 0


def unmapped_series(cursor, parsed):
    """Series in an existing series_values that the rebuild would not recreate.

    This script rebuilds from the legacy per-series tables only, so anything
    loaded straight into series_values would be lost.
    """
    cursor.execute("SELECT COUNT(*) FROM information_schema.tables "
                   "WHERE table_schema=DATABASE() AND table_name='series_values'")
    if not cursor.fetchone()[0]:
        return []
    cursor.execute("""
        SELECT o.name, r.code, s.code
        FROM (SELECT DISTINCT output_id, region_id, scenario_id FROM series_values) sv
        JOIN outputs   o ON o.output_id   = sv.output_id
        JOIN regions   r ON r.region_id   = sv.region_id
        JOIN scenarios s ON s.scenario_id = sv.scenario_id""")
    mapped = {(output, region, scenario) for output, region, scenario, _, _ in parsed}
    return sorted(set(cursor.fetchall()) - mapped)


def _fetch(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()


if __name__ == "__main__":
    raise SystemExit(main())
