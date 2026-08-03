"""Shared constants for sklearn estimators in this package."""

# Without this, trees and forests are not reproducible run-to-run.
RANDOM_STATE = 0

RUNS_TO_DROP_BY_OUTPUT = {
    "percapita_consumption_loss_percent": {
        "About15C_pes": [82, 98, 283, 305, 338, 373],
        "15C_med": [184, 221, 314, 374, 383],
    },
}
