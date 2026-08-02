"""Assert the codebase still produces the numbers recorded in tests/golden/.

Run with: pytest tests/test_golden.py

Needs a running MySQL server with the publication and all_data_aug_2024
databases. Cases with no snapshot on disk are skipped, so the suite stays
usable while the baseline is being filled in.
"""

import json
import os

import pytest

from tests.cases import cases
from tests.harness import run_case

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

ALL_CASES = cases()


def _expected(name):
    path = os.path.join(GOLDEN_DIR, name.replace("/", "__") + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _differences(expected, actual, path=""):
    """Walk two summaries together and report the leaves that differ.

    A plain equality assertion on these nested dicts produces an unreadable
    diff, and the digests in particular tell you nothing on their own. This
    reports the specific statistic that moved.
    """
    diffs = []
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float)) and isinstance(actual, (int, float))
    ):
        return [f"{path or '<root>'}: type {type(expected).__name__} -> {type(actual).__name__}"]

    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}" if path else key
            if key not in expected:
                diffs.append(f"{sub}: added ({actual[key]!r:.80})")
            elif key not in actual:
                diffs.append(f"{sub}: removed")
            else:
                diffs += _differences(expected[key], actual[key], sub)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(expected)} -> {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual)):
            diffs += _differences(e, a, f"{path}[{i}]")
    elif expected != actual:
        diffs.append(f"{path}: {expected!r} -> {actual!r}")
    return diffs


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_golden(name):
    expected = _expected(name)
    if expected is None:
        pytest.skip(f"no snapshot recorded for {name}; run scripts/capture_golden.py")

    actual = run_case(name, ALL_CASES[name])

    if expected["status"] == "error":
        assert actual["status"] == "error", (
            f"{name} used to raise {expected['error_type']} and now succeeds. "
            "If that is the intended fix, re-capture this case."
        )
        assert actual["error_type"] == expected["error_type"], (
            f"{name} raises {actual['error_type']}, expected {expected['error_type']}")
        return

    assert actual["status"] == "ok", (
        f"{name} used to succeed and now raises "
        f"{actual.get('error_type')} at {actual.get('error_site')}: "
        f"{actual.get('error_message')}")

    diffs = _differences(expected["result"], actual["result"])
    assert not diffs, "{} changed in {} place(s):\n  {}".format(
        name, len(diffs), "\n  ".join(diffs[:25]))


def test_colliding_keys_stay_separate():
    """The two databases must never converge on the same numbers.

    209 output keys exist in both databases holding different data. If a
    refactor ever points both datasets at one source these snapshots would
    still match individually only if both were recaptured, so this asserts the
    relationship between them directly.
    """
    pairs = [
        ("retrieval/pub/gdp_GLB_Ref_2050_COLLIDING",
         "retrieval/full/gdp_GLB_Ref_2050_COLLIDING"),
        ("retrieval/pub/population_USA_Ref_2050_COLLIDING",
         "retrieval/full/population_USA_Ref_2050_COLLIDING"),
    ]
    for pub_name, full_name in pairs:
        pub, full = _expected(pub_name), _expected(full_name)
        if pub is None or full is None:
            pytest.skip("snapshots not captured yet")
        assert pub["result"]["values_digest"] != full["result"]["values_digest"], (
            f"{pub_name} and {full_name} hold identical data. The publication and "
            "all_data_aug_2024 databases have been crossed.")
