"""Shared machinery for the golden-output snapshots.

The refactor has to prove it did not change any numbers. Every case defined
here is run against the current code, reduced to a JSON-safe summary, and
stored under tests/golden/. scripts/capture_golden.py writes those files and
tests/test_golden.py asserts the code still reproduces them.

Two things this is specifically built to catch:

1. Dataset crossover. "publication" and "all_data_aug_2024" share 209 identical
   output keys that hold different numbers, and the values are close enough
   (GDP GLB Ref 2050 differs by ~0.5%) that a chart built from the wrong
   database looks completely plausible. Cases marked with the same output name
   under both datasets pin that difference.

2. Silent numeric drift. Every summary carries both readable statistics and a
   digest of the full payload, so a failure says which column moved rather
   than just reporting a hash mismatch.

A case that currently raises is recorded as an error rather than skipped. The
baseline is what the code does today, bugs included; a refactor that changes a
failure into a success should surface for review, not pass silently.
"""

import hashlib
import json
import math
import traceback

import numpy as np
import pandas as pd

# Rounding applied to every float before it is hashed or compared. MySQL FLOAT
# columns carry ~7 significant digits, so 10 is well inside the noise floor
# while still catching a genuine change in a computation.
SIG_FIGS = 10


def _round(x, sig=SIG_FIGS):
    """Round to a number of significant figures, not decimal places.

    Values here span carbon prices near 1 and GDP near 1e5, so decimal-place
    rounding would be meaningless at one end of that range.
    """
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x if math.isfinite(x) else str(x)
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def canonical(value):
    """Reduce arbitrary values to ordered, JSON-serializable structures."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _round(value)
    if isinstance(value, (np.ndarray,)):
        return [canonical(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(v) for v in value)
    if isinstance(value, pd.Series):
        return {"index": canonical(list(value.index)), "values": canonical(value.to_numpy())}
    if isinstance(value, pd.DataFrame):
        return summarize_frame(value)
    return str(value)


def digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def summarize_frame(df):
    """Readable statistics plus a digest of every cell."""
    numeric = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric[str(col)] = {
                "mean": _round(s.mean()) if len(s) else None,
                "std": _round(s.std()) if len(s) > 1 else None,
                "min": _round(s.min()) if len(s) else None,
                "max": _round(s.max()) if len(s) else None,
                "sum": _round(s.sum()) if len(s) else None,
                "nulls": int(s.isna().sum()),
            }
    return {
        "kind": "dataframe",
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "dtypes": [str(t) for t in df.dtypes],
        "index_name": str(df.index.name),
        "stats": numeric,
        "head": canonical(df.head(3).to_dict(orient="records")),
        "tail": canonical(df.tail(3).to_dict(orient="records")),
        # Digest over every value, so drift outside the summary still fails.
        # The index is digested separately because it is incidental to most of
        # these frames: today it is whatever row offsets survived a dropna and
        # a year filter. A new data layer can produce identical numbers on a
        # different index, and that should read as one obvious difference
        # rather than as every frame in the suite changing at once. It is not
        # ignored, because VariableOutput aligns its operands on the index, so
        # a change there is worth seeing.
        "values_digest": digest(canonical(df.to_dict(orient="records"))),
        "index_digest": digest(canonical(list(df.index))),
    }


def _strip_volatile(node):
    """Remove plotly's per-trace uid, which is regenerated on every build."""
    if isinstance(node, dict):
        return {k: _strip_volatile(v) for k, v in node.items() if k != "uid"}
    if isinstance(node, (list, tuple)):
        return [_strip_volatile(v) for v in node]
    return node


def summarize_figure(fig):
    """Structure and content of a plotly figure.

    layout.template is hashed on its own. It is large and mostly static, so
    folding it into the main digest would bury a real change to a title or an
    axis label under thousands of lines of theme defaults.
    """
    raw = _strip_volatile(fig.to_plotly_json())
    layout = dict(raw.get("layout", {}))
    template = layout.pop("template", None)

    traces = []
    for tr in raw.get("data", []):
        entry = {"type": tr.get("type"), "name": tr.get("name")}
        for axis in ("x", "y", "z", "values", "labels", "locations"):
            if isinstance(tr.get(axis), (list, tuple, np.ndarray)):
                entry[f"n_{axis}"] = len(tr[axis])
        if "dimensions" in tr:
            entry["dimensions"] = [d.get("label") for d in tr["dimensions"]]
        traces.append(entry)

    title = layout.get("title")
    return {
        "kind": "figure",
        "n_traces": len(raw.get("data", [])),
        "traces": canonical(traces),
        "title": canonical(title.get("text") if isinstance(title, dict) else title),
        "layout_keys": sorted(layout),
        "annotations": canonical([a.get("text") for a in layout.get("annotations", []) or []]),
        "data_digest": digest(canonical(raw.get("data", []))),
        "layout_digest": digest(canonical(layout)),
        "template_digest": digest(canonical(template)) if template is not None else None,
    }


def summarize_tree(model):
    """The fitted structure of a decision tree, not just its predictions."""
    t = model.tree_
    return {
        "kind": "decision_tree",
        "n_nodes": int(t.node_count),
        "max_depth": int(t.max_depth),
        "classes": canonical(getattr(model, "classes_", [])),
        "features": canonical(t.feature),
        "thresholds": canonical(t.threshold),
        "children_left": canonical(t.children_left),
        "children_right": canonical(t.children_right),
        "value_digest": digest(canonical(t.value)),
        "feature_names": canonical(list(getattr(model, "feature_names_in_", []))),
    }


def run_case(name, fn):
    """Run one case, capturing either its summary or the way it fails."""
    try:
        return {"case": name, "status": "ok", "result": canonical(fn())}
    except Exception as exc:  # noqa: BLE001 - the failure mode is the record
        return {
            "case": name,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:400],
            "error_site": _origin(exc),
        }


def _origin(exc):
    """Innermost frame inside this project, so the record survives refactors
    of unrelated library code."""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if "site-packages" not in frame.filename:
            return f"{frame.filename.split('/')[-1]}:{frame.lineno}"
    return None
