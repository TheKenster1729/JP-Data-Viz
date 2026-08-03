"""Scratch: old data layer against the new one, values and index.

Rebuilds the old read path locally (name_mappings -> per-series table) so both
can be compared inside one process, and times the four shapes the new layer is
meant to serve.
"""
import json
import os
import random
import sys
import time

import mysql.connector
import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from data.repository import repository  # noqa: E402
from styling import Options, Readability  # noqa: E402

PUB = "publication"
FULL = "all_data_aug_2024"


# ---- the old layer, verbatim ------------------------------------------------

def old_get_df(engine, output, region, scenario, year=None):
    long_name = output + "_" + region + "_" + scenario
    query = text("SELECT `Assigned Name` FROM name_mappings WHERE `Full Output Name`=:long_name")
    with engine.connect() as conn:
        table = conn.execute(query, parameters={"long_name": long_name}).fetchall()[0][0]
    try:
        df = pd.read_sql_table(table, con=engine).drop(columns="index_name")
    except KeyError:
        df = pd.read_sql_table(table, con=engine)
    df = df.dropna()
    if output == "percapita_consumption_loss_percent":
        df["Value"] = df["Value"] * 100
    df["Value"] = df["Value"].replace("Eps", 0)
    if year:
        df = df[df["Year"] == year]
    return df


def old_number_to_ordinal(n):
    if 11 <= n <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return str(n) + suffix


def old_band(engine, output, region, scenario, lower, upper, year=None):
    df = old_get_df(engine, output, region, scenario, year)
    band = df.groupby(["Year"])["Value"].agg(
        [lambda x: np.percentile(x, lower), np.median, lambda x: np.percentile(x, upper)])
    band.columns = ["{} Percentile".format(old_number_to_ordinal(lower)), "Median",
                    "{} Percentile".format(old_number_to_ordinal(upper))]
    return band


def old_choropleth(engine, output, scenario, year, lower, upper):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    regions = Options().region_names[1:]

    def one(region):
        try:
            row = old_band(engine, output, region, scenario, lower, upper, year).loc[year]
            return {"Region": region,
                    f"{old_number_to_ordinal(lower)} Percentile": row.iloc[0],
                    "Median": row.iloc[1],
                    f"{old_number_to_ordinal(upper)} Percentile": row.iloc[2]}
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, r): r for r in regions}
        for future in as_completed(futures):
            got = future.result()
            if got is not None:
                results.append(got)
    lower_col = f"{old_number_to_ordinal(lower)} Percentile"
    upper_col = f"{old_number_to_ordinal(upper)} Percentile"
    frame = pd.DataFrame(results, columns=["Region", lower_col, "Median", upper_col])
    order = {r: i for i, r in enumerate(regions)}
    frame["_sort"] = frame["Region"].map(order)
    return frame.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def old_multi(engine, outputs, region, scenario, year):
    frame = pd.DataFrame()
    for output in outputs:
        name = (Readability().naming_dict_long_names_first[output]
                if output in Options().outputs else json.loads(output)["name"])
        piece = old_get_df(engine, output, region, scenario, year)
        add = pd.DataFrame()
        add["Run #"] = piece["Run #"]
        add[name] = piece["Value"]
        if len(frame) == 0:
            frame = add
        else:
            frame = frame[frame["Run #"].isin(set(piece["Run #"]))]
            frame[name] = piece["Value"]
    return frame


# ---- comparison -------------------------------------------------------------

def sample_series(dbname, n):
    conn = mysql.connector.connect(**config.connector_kwargs(dbname))
    cur = conn.cursor()
    cur.execute("SELECT `Full Output Name` FROM name_mappings")
    names = [r[0] for r in cur.fetchall()]
    conn.close()
    repo = repository(dbname)
    triples = []
    for full in names:
        for scenario in sorted(repo.dataset.scenarios, key=len, reverse=True):
            if not full.endswith("_" + scenario):
                continue
            output, _, region = full[: -len(scenario) - 1].rpartition("_")
            if repo.dataset.has_output(output) and repo.dataset.has_region(region):
                triples.append((output, region, scenario))
            break
    random.seed(0)
    return random.sample(triples, min(n, len(triples)))


def compare(dbname, n):
    repo = repository(dbname)
    engine = config.engine(dbname)
    bad_values = bad_index = bad_shape = 0
    print(f"\n--- {dbname}: {n} random series, full read ---")
    for output, region, scenario in sample_series(dbname, n):
        old = old_get_df(engine, output, region, scenario)
        new = repo.series(output, region, scenario)
        if old.shape != new.shape:
            bad_shape += 1
            print(f"  SHAPE  {output} {region} {scenario}: {old.shape} vs {new.shape}")
            continue
        if not old.reset_index(drop=True).equals(new.reset_index(drop=True)):
            bad_values += 1
            diff = old.reset_index(drop=True).compare(new.reset_index(drop=True))
            print(f"  VALUES {output} {region} {scenario}: {len(diff)} rows differ")
        if list(old.index) != list(new.index):
            bad_index += 1
            print(f"  INDEX  {output} {region} {scenario}")
    print(f"  shape mismatches {bad_shape}, value mismatches {bad_values}, "
          f"index mismatches {bad_index}")


def compare_year(dbname, n, year=2050):
    repo = repository(dbname)
    engine = config.engine(dbname)
    bad_values = index_differs = 0
    print(f"\n--- {dbname}: {n} random series, single year {year} ---")
    for output, region, scenario in sample_series(dbname, n):
        old = old_get_df(engine, output, region, scenario, year)
        new = repo.series(output, region, scenario, year)
        if not old.reset_index(drop=True).equals(new.reset_index(drop=True)):
            bad_values += 1
            print(f"  VALUES {output} {region} {scenario}")
        if list(old.index) != list(new.index):
            index_differs += 1
    print(f"  value mismatches {bad_values}, index mismatches {index_differs}")


# ---- benchmark --------------------------------------------------------------

def bench(label, fn, repeats=5):
    fn()  # warm
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000)
    return label, min(times), sum(times) / len(times)


def benchmark():
    rows = []
    for dbname, output, region, scenario, year, others in (
        # Publication-only output names cannot be used here: the old
        # MultiOutputRetrieval names its columns from Options().outputs, which
        # is the other dataset's list, and json.loads the name when it misses.
        (PUB, "GDP_billion_USD2007", "GLB", "2C", 2050,
         ["population_million_people", "sectoral_output_Electricity_billion_USD2007",
          "primary_energy_use_Oil_EJ"]),
        (FULL, "emissions_CO2eq_total_million_ton_CO2eq", "GLB", "15C_med", 2050,
         ["GDP_billion_USD2007", "consumption_billion_USD2007"]),
    ):
        repo = repository(dbname)
        engine = config.engine(dbname)
        multi = [output] + others
        cases = [
            ("one series-year",
             lambda: old_get_df(engine, output, region, scenario, year),
             lambda: repo.series(output, region, scenario, year)),
            ("full series",
             lambda: old_get_df(engine, output, region, scenario),
             lambda: repo.series(output, region, scenario)),
            ("choropleth, 18 regions",
             lambda: old_choropleth(engine, output, scenario, year, 5, 95),
             lambda: repo.choropleth(output, Options().region_names[1:], scenario,
                                     year, 5, 95)),
            (f"multi-output, {len(multi)} outputs",
             lambda: old_multi(engine, multi, region, scenario, year),
             lambda: repo.multi_output(multi, region, scenario, year)),
        ]
        for label, old_fn, new_fn in cases:
            _, old_min, old_avg = bench(label, old_fn)
            _, new_min, new_avg = bench(label, new_fn)
            rows.append((dbname, label, old_min, old_avg, new_min, new_avg))

    print("\n=== benchmark (ms, min of 5 / mean of 5) ===")
    print(f"{'dataset':18} {'case':26} {'old min':>9} {'old mean':>9} "
          f"{'new min':>9} {'new mean':>9} {'speedup':>8}")
    for dbname, label, om, oa, nm, na in rows:
        print(f"{dbname:18} {label:26} {om:9.2f} {oa:9.2f} {nm:9.2f} {na:9.2f} "
              f"{oa / na:7.1f}x")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "compare"):
        compare(PUB, 60)
        compare(FULL, 40)
        compare_year(PUB, 40)
        compare_year(FULL, 40)
    if what in ("all", "bench"):
        benchmark()
