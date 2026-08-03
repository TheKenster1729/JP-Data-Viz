"""Read one dataset out of the normalized series_values schema.

A repository is bound to exactly one database for its whole life. It resolves
every output, region and scenario through that database's catalog, so it can
neither read another dataset nor silently accept a name that only exists in
the other one.

Query shape matters more than usual here. series_values has no secondary
indexes and no foreign keys, and its primary key is
(output_id, region_id, scenario_id, year, run). Every query below binds
output_id, by equality or an IN list, so the leading key column is constrained
and the read stays a range scan over the clustered index. Leaving it open
forces a full scan: on publication, all 34 outputs for one region, scenario and
year take 7 ms with an IN list and 1,311 ms without. The same applies to
region_id when sweeping regions, 3.7 ms against 41.9 ms, which is why a
choropleth is one query rather than one per region.

Frames match what the old per-series tables produced: columns Run #, Year,
Value, year-major and run-minor, with NULLs dropped. A single-year fetch still
keeps the row's offset into the whole series as its index (year 2050 starts at
2400 when every earlier year is complete), because that is what filtering the
melted table produced and what MultiOutputRetrieval inherited for its frame.
"""

import threading

import numpy as np
import pandas as pd

from data import expressions
from datasets import NotInDataset, dataset as get_dataset

RUN = "Run #"
YEAR = "Year"
VALUE = "Value"
COLUMNS = [RUN, YEAR, VALUE]
DTYPES = {RUN: "int64", YEAR: "int64", VALUE: "float64"}

# Ingested as a fraction, presented as a percentage everywhere downstream.
SCALED_BY_100 = {"percapita_consumption_loss_percent"}

# Custom variables are transient. This only spares the JSON decoder work
# within a session, and is bounded so a caller that builds a fresh variable per
# request cannot grow it without limit.
_EXPRESSION_CACHE_LIMIT = 256


def ordinal(n):
    """1 -> '1st'. Reproduces DataRetrieval.number_to_ordinal over 1..100."""
    if 11 <= n <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return str(n) + suffix


class SeriesRepository:
    """Every read of one dataset goes through here."""

    def __init__(self, dataset_name):
        self.dataset = get_dataset(dataset_name)
        self.engine = self.dataset.engine
        self._expressions = {}
        self._lock = threading.Lock()

    @property
    def name(self):
        return self.dataset.name

    # ---- reads -----------------------------------------------------------

    def series(self, output, region, scenario, year=None):
        """One output for one region and scenario, all years or just one."""
        if self.dataset.has_output(output):
            return self._fetch(output, region, scenario, year)
        return self._evaluate(self.expression(output), region, scenario, year)

    def percentile_band(self, output, region, scenario, lower, upper, year=None):
        """Year-indexed lower percentile, median and upper percentile."""
        return _band(self.series(output, region, scenario, year), lower, upper)

    def choropleth(self, output, regions, scenario, year, lower, upper):
        """One row per region: lower percentile, median, upper percentile.

        Regions this dataset does not hold are dropped rather than raised on.
        The caller passes a fixed world map rather than a user request, and the
        old layer dropped them too, by way of a bare except.
        """
        columns = ["Region", ordinal(lower) + " Percentile", "Median",
                   ordinal(upper) + " Percentile"]
        if year is None:
            # The old implementation indexed the band by year and swallowed the
            # resulting KeyError, leaving an empty frame.
            return pd.DataFrame([], columns=columns)

        by_region = self._values_by_region(output, regions, scenario, year)
        rows = []
        for region in regions:
            values = by_region.get(region)
            if values is None or not len(values):
                continue
            rows.append({
                "Region": region,
                columns[1]: np.percentile(values, lower),
                "Median": np.median(values),
                columns[3]: np.percentile(values, upper),
            })
        return pd.DataFrame(rows, columns=columns)

    def multi_output(self, outputs, region, scenario, year):
        """Wide frame: Run # plus one column per output, named for display.

        Outputs are intersected on Run # as they are added, so a run that
        crashed for one of them is absent from all of them. year=None yields
        an empty frame: MultiOutputRetrieval historically went through
        mapping_df, which returned empty when no year was set.
        """
        if year is None:
            return pd.DataFrame({RUN: pd.Series(dtype="int64")})
        frames = self._by_output(outputs, region, scenario, year)
        result = pd.DataFrame()
        for output in outputs:
            column = self.display_name(output)
            frame = frames[output]
            if len(result) == 0:
                result = pd.DataFrame({RUN: frame[RUN], column: frame[VALUE]})
            else:
                result = result[result[RUN].isin(set(frame[RUN]))]
                result[column] = frame[VALUE]
        return result

    def display_name(self, output):
        if self.dataset.has_output(output):
            return self.dataset.display_name(output)
        return self.expression(output).display_name

    def expression(self, output):
        """Parse and validate a custom variable against this dataset."""
        key = output if isinstance(output, str) else repr(output)
        with self._lock:
            cached = self._expressions.get(key)
        if cached is not None:
            return cached

        try:
            parsed = expressions.parse(output)
        except (ValueError, TypeError):
            raise self.dataset.missing("output", output)
        if isinstance(parsed, expressions.OutputRef):
            # A bare name the catalog does not hold. Saying so here beats
            # failing later on an empty lookup.
            raise self.dataset.missing("output", output)
        for name in parsed.outputs():
            self.dataset.output_id(name)

        with self._lock:
            if len(self._expressions) >= _EXPRESSION_CACHE_LIMIT:
                self._expressions.clear()
            self._expressions[key] = parsed
        return parsed

    # ---- queries ---------------------------------------------------------

    def _rows(self, sql, params):
        """Raw driver cursor: these read up to 6,800 rows and are called per
        region and per output, so the result-wrapping SQLAlchemy would do on
        top is a measurable share of the query."""
        connection = self.engine.raw_connection()
        try:
            cursor = connection.cursor()
            try:
                cursor.execute(sql, params)
                return cursor.fetchall()
            finally:
                cursor.close()
        finally:
            connection.close()  # returns it to the pool

    def _fetch(self, output, region, scenario, year=None, scale=True):
        output_id = self.dataset.output_id(output)
        region_id = self.dataset.region_id(region)
        scenario_id = self.dataset.scenario_id(scenario)

        # Always read the whole series, then filter. Pushing year into the
        # WHERE clause would be a few milliseconds faster, but it restarts the
        # index at 0 and MultiOutputRetrieval's column assignment inherits that
        # index from the first output. Matching the old melt-then-filter index
        # is worth more than the round trip.
        rows = self._rows(
            "SELECT run, year, value FROM series_values "
            "WHERE output_id=%s AND region_id=%s AND scenario_id=%s "
            "ORDER BY year, run",
            (output_id, region_id, scenario_id))
        frame = _frame(rows, scale and output in SCALED_BY_100)
        if year is not None:
            frame = frame[frame[YEAR] == int(year)]
        return frame

    def _values_by_region(self, output, regions, scenario, year):
        """One year of one output across many regions, in a single query.

        Only the values come back. A choropleth reduces each region to three
        order-independent statistics, so neither the run nor the year is worth
        carrying, and no per-region frame has to be built.
        """
        if not self.dataset.has_output(output):
            expression = self.expression(output)
            return {region: self._evaluate(expression, region, scenario, year)[VALUE].to_numpy()
                    for region in regions if self.dataset.has_region(region)}

        wanted = [(self.dataset.region_id(region), region)
                  for region in regions if self.dataset.has_region(region)]
        if not wanted:
            return {}

        placeholders = ", ".join(["%s"] * len(wanted))
        params = ([self.dataset.output_id(output)]
                  + [region_id for region_id, _ in wanted]
                  + [self.dataset.scenario_id(scenario), int(year)])
        rows = self._rows(
            "SELECT region_id, value FROM series_values "
            "WHERE output_id=%s AND region_id IN ({}) AND scenario_id=%s AND year=%s".format(
                placeholders),
            tuple(params))

        grouped = {}
        for region_id, value in rows:
            if value is not None:
                grouped.setdefault(region_id, []).append(value)
        scale = 100 if output in SCALED_BY_100 else 1
        return {region: np.array(grouped.get(region_id, []), dtype="float64") * scale
                for region_id, region in wanted}

    def _by_output(self, outputs, region, scenario, year):
        """Many outputs for one region, scenario and year.

        The plain outputs come back in one query bound by an output_id IN list;
        custom variables are evaluated separately because each is its own tree.
        """
        frames = {}
        plain = {}
        for output in outputs:
            if self.dataset.has_output(output):
                plain[self.dataset.output_id(output)] = output

        if plain:
            region_id = self.dataset.region_id(region)
            scenario_id = self.dataset.scenario_id(scenario)
            placeholders = ", ".join(["%s"] * len(plain))
            # Same whole-series-then-filter rule as _fetch, so each column of
            # the wide frame carries the melt-then-filter index.
            rows = self._rows(
                "SELECT output_id, run, year, value FROM series_values "
                "WHERE output_id IN ({}) AND region_id=%s AND scenario_id=%s "
                "ORDER BY output_id, year, run".format(placeholders),
                tuple(list(plain) + [region_id, scenario_id]))

            grouped = {}
            for output_id, run, row_year, value in rows:
                grouped.setdefault(output_id, []).append((run, row_year, value))
            for output_id, output in plain.items():
                frame = _frame(grouped.get(output_id, []),
                               output in SCALED_BY_100)
                if year is not None:
                    frame = frame[frame[YEAR] == int(year)]
                frames[output] = frame

        for output in outputs:
            if output not in frames:
                frames[output] = self._evaluate(
                    self.expression(output), region, scenario, year)
        return frames

    def _evaluate(self, expression, region, scenario, year):
        # Region and scenario are checked once here rather than once per leaf.
        self.dataset.region_id(region)
        self.dataset.scenario_id(scenario)
        # scale=False: the old layer keyed the x100 off the requested output,
        # not the fetched one, so a custom variable referencing that output did
        # not scale it. Preserved rather than corrected.
        return expression.evaluate(
            lambda name: self._fetch(name, region, scenario, year, scale=False),
            region, scenario, year)


def _frame(rows, scale_by_100=False):
    """Rows in (year, run) order to a Run #/Year/Value frame.

    The index is the row's position before NULLs are dropped, which is what the
    old melt-then-insert order produced. Callers that then filter to one year
    keep those positions, matching DataRetrieval.get_df.
    """
    if not rows:
        frame = pd.DataFrame({column: pd.Series(dtype=dtype)
                              for column, dtype in DTYPES.items()})
    else:
        # Built column by column. Handing pandas 6,800 driver tuples costs
        # about a millisecond in type inference; numpy converts each column in
        # one pass, and NULL becomes NaN on the way through.
        runs, years, values = zip(*rows)
        frame = pd.DataFrame({
            RUN: np.array(runs, dtype="int64"),
            YEAR: np.array(years, dtype="int64"),
            VALUE: np.array(values, dtype="float64"),
        })
    frame = frame.dropna()
    if scale_by_100:
        frame[VALUE] = frame[VALUE] * 100
    return frame


def _band(frame, lower, upper):
    band = frame.groupby([YEAR])[VALUE].agg([
        lambda x: np.percentile(x, lower),
        np.median,
        lambda x: np.percentile(x, upper),
    ])
    band.columns = ["{} Percentile".format(ordinal(lower)), "Median",
                    "{} Percentile".format(ordinal(upper))]
    return band


_REPOSITORIES = {}
_REPOSITORY_LOCK = threading.Lock()


def repository(dataset_name):
    """The repository for one database, created once per process.

    Sharing it shares the catalog and the connection pool, and guarantees a
    repository handed out for one dataset is never reused for another.
    """
    with _REPOSITORY_LOCK:
        if dataset_name not in _REPOSITORIES:
            _REPOSITORIES[dataset_name] = SeriesRepository(dataset_name)
        return _REPOSITORIES[dataset_name]


__all__ = ["NotInDataset", "SeriesRepository", "ordinal", "repository"]
