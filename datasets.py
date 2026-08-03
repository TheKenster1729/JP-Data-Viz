"""Per-dataset registry: what each database contains, and its dimension ids.

There are two databases and they must never be mixed. 11 output names exist in
both holding different numbers, and the two are close enough that a chart built
from the wrong one looks plausible: GDP for GLB Ref 2050 differs by about half
a percent. This module is where that separation is enforced, by giving every
caller a catalog scoped to one database instead of a union of both.

That is the specific hazard in ``styling.Options.all_outputs``, which
concatenates the two output lists. Membership in it says an output exists
somewhere, not that it exists in the database being read.

The dimension tables are small (34 or 80 outputs, 19 regions, 2 or 12
scenarios) and static, so each catalog is loaded once per process and cached.
Every later query binds integer ids taken from here rather than joining the
dimension tables again.
"""

import threading

from sqlalchemy import text

import config
from styling import Options

PUBLICATION = "publication"
ALL_DATA = "all_data_aug_2024"

# The scenario vocabulary each database was built with. Kept here rather than
# derived from the database so a missing or extra scenario is a visible
# disagreement instead of silently becoming the new truth.
SCENARIOS = {
    PUBLICATION: ("Ref", "2C"),
    ALL_DATA: tuple(Options().scenarios),
}

DATASET_NAMES = tuple(sorted(SCENARIOS))


class UnknownDataset(LookupError):
    pass


class NotInDataset(LookupError):
    """A requested output, region or scenario is not in the bound dataset.

    Replaces the ``IndexError: list index out of range`` the old layer raised
    when a name_mappings lookup came back empty, which said nothing about
    which name or which database was at fault.
    """


class Dataset:
    """The name, scenario vocabulary and dimension catalog of one database."""

    def __init__(self, name):
        if name not in SCENARIOS:
            raise UnknownDataset(
                "unknown dataset {!r}; known datasets are {}".format(
                    name, ", ".join(DATASET_NAMES)))
        self.name = name
        self.scenario_vocabulary = SCENARIOS[name]
        self.engine = config.engine(name)
        self._load()

    def _load(self):
        with self.engine.connect() as conn:
            outputs = conn.execute(
                text("SELECT name, output_id, display_name FROM outputs")).fetchall()
            regions = conn.execute(
                text("SELECT code, region_id FROM regions")).fetchall()
            scenarios = conn.execute(
                text("SELECT code, scenario_id FROM scenarios")).fetchall()

        self._output_ids = {name: int(oid) for name, oid, _ in outputs}
        self._display_names = {name: display for name, _, display in outputs}
        self._region_ids = {code: int(rid) for code, rid in regions}
        self._scenario_ids = {code: int(sid) for code, sid in scenarios}

        self.outputs = tuple(sorted(self._output_ids))
        # Regions keep the app's display order, not the alphabetical order the
        # migration assigned ids in. Anything the database has and Options does
        # not is appended rather than dropped.
        known = [r for r in Options().region_names if r in self._region_ids]
        self.regions = tuple(known + sorted(set(self._region_ids) - set(known)))
        declared = [s for s in self.scenario_vocabulary if s in self._scenario_ids]
        self.scenarios = tuple(declared + sorted(set(self._scenario_ids) - set(declared)))

    def has_output(self, name):
        return isinstance(name, str) and name in self._output_ids

    def has_region(self, code):
        return code in self._region_ids

    def has_scenario(self, code):
        return code in self._scenario_ids

    def output_id(self, name):
        try:
            return self._output_ids[name]
        except (KeyError, TypeError):
            raise self.missing("output", name)

    def region_id(self, code):
        try:
            return self._region_ids[code]
        except (KeyError, TypeError):
            raise self.missing("region", code)

    def scenario_id(self, code):
        try:
            return self._scenario_ids[code]
        except (KeyError, TypeError):
            raise self.missing("scenario", code)

    def display_name(self, name):
        """The published name for an output, falling back to its id.

        outputs.display_name is NULL for anything absent from display_names.csv
        and publication_output_names.csv.
        """
        return self._display_names.get(name) or name

    def missing(self, kind, value):
        """The error to raise for a name this dataset does not hold.

        Naming the dataset is the point: the same output name can be valid
        against the other database, and that is the mistake worth catching.
        """
        available = {"output": self.outputs, "region": self.regions,
                     "scenario": self.scenarios}[kind]
        near = []
        if isinstance(value, str):
            near = [name for name in available if value.lower() in name.lower()][:5]
        if near:
            detail = "; closest in this dataset: {}".format(", ".join(near))
        elif len(available) <= 12:
            detail = "; this dataset has: {}".format(", ".join(map(str, available)))
        else:
            detail = "; this dataset has {} {}s, e.g. {}".format(
                len(available), kind, ", ".join(map(str, available[:5])))
        return NotInDataset("{} {!r} is not in dataset {!r}{}".format(
            kind, value, self.name, detail))

    def __repr__(self):
        return "<Dataset {} outputs={} regions={} scenarios={}>".format(
            self.name, len(self.outputs), len(self.regions), len(self.scenarios))


_DATASETS = {}
_LOCK = threading.Lock()


def dataset(name):
    """The catalog for one database, loaded once per process."""
    with _LOCK:
        if name not in _DATASETS:
            _DATASETS[name] = Dataset(name)
        return _DATASETS[name]
