"""Persist time-series cluster assignments per run (avoid re-fitting k-means)."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLUSTER_LABELS_ROOT = Path(__file__).resolve().parents[2] / "data" / "cluster_labels"


class _FittedClusters:
  def __init__(self, labels_, cluster_centers_, inertia_=None):
    self.labels_ = labels_
    self.cluster_centers_ = cluster_centers_
    if inertia_ is not None:
      self.inertia_ = inertia_


def _safe_output_slug(output):
  text = output if isinstance(output, str) else str(output)
  if text.startswith("{") or text.startswith("["):
    return "custom_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
  return re.sub(r"[^\w.-]+", "_", text)


def cluster_labels_csv_path(
    dataset_name, output, region, scenario, n_clusters, metric, year=None,
):
  parts = [
      _safe_output_slug(output),
      region,
      scenario,
      f"k{n_clusters}",
      metric,
  ]
  if year is not None:
    parts.append(str(int(year)))
  filename = "_".join(parts) + ".csv"
  return CLUSTER_LABELS_ROOT / dataset_name / filename


def load_cluster_labels(path, run_index):
  """Return label array aligned to run_index, or None if missing or stale."""
  if not path.is_file():
    return None
  frame = pd.read_csv(path)
  if "Run #" not in frame.columns or "cluster" not in frame.columns:
    return None
  by_run = frame.set_index("Run #")["cluster"]
  try:
    labels = by_run.reindex(run_index).to_numpy()
  except Exception:
    return None
  if np.any(pd.isna(labels)):
    return None
  return labels.astype(int)


def save_cluster_labels(path, run_index, labels):
  path.parent.mkdir(parents=True, exist_ok=True)
  frame = pd.DataFrame({"Run #": run_index, "cluster": labels})
  frame.to_csv(path, index=False)


def fitted_from_labels(pivot_df, labels):
  """Build labels_ and cluster_centers_ like TimeSeriesKMeans after fit."""
  labels = np.asarray(labels, dtype=int)
  n_clusters = int(labels.max()) + 1
  centers = []
  for k in range(n_clusters):
    mask = labels == k
    if not np.any(mask):
      centers.append(np.zeros(pivot_df.shape[1], dtype=float))
    else:
      centers.append(pivot_df.iloc[mask].mean(axis=0).to_numpy())
  return _FittedClusters(labels, np.array(centers))


def reorder_clusters_by_terminal_year(fitted, pivot_df):
  """
  Relabel clusters so Cluster 1 is lowest at the final year, then ascending
  (e.g. low / mid / high renewable share at 2100).
  """
  labels = np.asarray(fitted.labels_, dtype=int)
  centers = np.asarray(fitted.cluster_centers_)
  n_clusters = int(labels.max()) + 1
  if n_clusters <= 1:
    return fitted

  terminal = []
  for k in range(n_clusters):
    if centers.ndim >= 2:
      terminal.append(float(centers[k].ravel()[-1]))
    else:
      mask = labels == k
      terminal.append(float(pivot_df.iloc[mask].iloc[:, -1].mean()))

  order = sorted(range(n_clusters), key=lambda k: terminal[k])
  old_to_new = {old: new for new, old in enumerate(order)}
  new_labels = np.array([old_to_new[lab] for lab in labels], dtype=int)
  new_centers = np.array([centers[order[new]] for new in range(n_clusters)])
  inertia = getattr(fitted, "inertia_", None)
  return _FittedClusters(new_labels, new_centers, inertia_=inertia)


class ClusterLabelFileCache:
    """Load cluster labels from CSV when present; otherwise fit and write."""

    dataset_name = "publication"

    def generate_clusters(self):
        from tslearn.clustering import TimeSeriesKMeans

        year_tag = getattr(self, "year", None)
        path = cluster_labels_csv_path(
            self.dataset_name,
            self.output,
            self.region,
            self.scenario,
            self.n_clusters,
            self.metric,
            year_tag,
        )
        runs = self.df_for_clustering.index
        cached = load_cluster_labels(path, runs)
        if cached is not None:
            fitted = fitted_from_labels(self.df_for_clustering, cached)
            return reorder_clusters_by_terminal_year(fitted, self.df_for_clustering)

        fitted = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric=self.metric,
            random_state=self.random_state,
        ).fit(self.df_for_clustering)
        fitted = reorder_clusters_by_terminal_year(
            _FittedClusters(fitted.labels_, fitted.cluster_centers_, inertia_=fitted.inertia_),
            self.df_for_clustering,
        )
        save_cluster_labels(path, runs, fitted.labels_)
        return fitted
