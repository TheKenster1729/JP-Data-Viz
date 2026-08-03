"""Turn a continuous output vector into a binary classification target."""

import numpy as np


def discrete_from_percentile(values, threshold, gt=True):
    """Label runs above (gt=True) or below (gt=False) the percentile threshold."""
    percentile = np.percentile(values, threshold)
    if gt:
        return np.where(values > percentile, 1, 0)
    return np.where(values < percentile, 1, 0)
