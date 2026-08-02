"""Record the current behavior of the codebase as the golden baseline.

    python scripts/capture_golden.py                 # capture everything
    python scripts/capture_golden.py --only retrieval # capture a subset
    python scripts/capture_golden.py --list           # show case names

Writes one JSON file per case under tests/golden/ plus an index summarizing
the run. Requires a running MySQL server with the publication and
all_data_aug_2024 databases.

Only re-run this when a change to the numbers is intended. Recapturing to make
a failing test pass defeats the point of having it.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.cases import cases  # noqa: E402
from tests.harness import run_case  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tests", "golden")


def path_for(name):
    return os.path.join(GOLDEN_DIR, name.replace("/", "__") + ".json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="substring filter on case names")
    parser.add_argument("--list", action="store_true", help="list case names and exit")
    args = parser.parse_args()

    all_cases = cases()
    if args.list:
        for name in sorted(all_cases):
            print(name)
        return 0

    selected = {k: v for k, v in all_cases.items() if not args.only or args.only in k}
    if not selected:
        print(f"no cases match {args.only!r}")
        return 1

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    index, failures = {}, []

    for i, name in enumerate(sorted(selected), 1):
        started = time.time()
        record = run_case(name, selected[name])
        elapsed = time.time() - started

        with open(path_for(name), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")

        index[name] = {"status": record["status"], "seconds": round(elapsed, 2)}
        if record["status"] == "error":
            failures.append((name, record["error_type"], record["error_message"],
                             record.get("error_site")))
            flag = f"ERROR {record['error_type']}"
        else:
            flag = "ok"
        print(f"[{i:2d}/{len(selected)}] {elapsed:6.2f}s  {flag:24s} {name}")

    with open(os.path.join(GOLDEN_DIR, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")

    ok = len(selected) - len(failures)
    print(f"\ncaptured {len(selected)} cases: {ok} ok, {len(failures)} recorded as errors")
    if failures:
        print("\ncases that currently raise (their failure is now the baseline):")
        for name, etype, msg, site in failures:
            print(f"  {name}\n      {etype} at {site}: {msg.splitlines()[0][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
