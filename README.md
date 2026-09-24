# MIT EPPA Model Data Visualization Dashboard

Local Dash application for exploring MIT EPPA model ensembles: time series, input/output mappings, choropleths, clustering, and custom derived outputs.

## Datasets

Two MySQL databases must stay separate; they share output names but not the same series:

| Database | Source folder | Role |
|----------|---------------|------|
| `publication` | `Raw Data/Archive` | Publication ensemble |
| `all_data_aug_2024` | `Raw Data/New_Ensembles` | Full ensemble |

The UI “Overview” tab switches between them. Queries always bind `output_id` on the normalized `series_values` schema.

## Setup

1. Python 3.9+ recommended (project uses a `.venv` at the repo root).
2. Install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. MySQL with both databases loaded. Connection defaults (override with env):

   - `JP_DB_HOST`, `JP_DB_USER`, `JP_DB_PASSWORD` (or `MYSQL_PWD`)
   - See `config.py` for database names and ports.

4. Migrate Excel archives into MySQL (one-time or after data updates):

   ```bash
   python scripts/migrate_schema.py --dataset publication
   python scripts/migrate_schema.py --dataset all_data_aug_2024
   python scripts/validate_against_excel.py --dataset publication
   python scripts/validate_against_excel.py --dataset all_data_aug_2024
   ```

## Run the dashboard

```bash
source .venv/bin/activate
python app.py
```

Open `http://0.0.0.0:8050` (Dash default). For production-style serving, use gunicorn as in `startup.txt` (`app:server`).

## Tests

Golden tests freeze retrieval, analysis, and figure outputs:

```bash
pytest tests/test_golden.py
```

Recapture baselines after intentional behavior changes:

```bash
python scripts/capture_golden.py
```

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Entry point (`create_app`) |
| `eppa_viz/webapp/` | Dash layout and callbacks |
| `eppa_viz/figures/` | Plotly figure builders and styling pipeline |
| `eppa_viz/analysis/` | Input/output mapping and ML helpers |
| `data/` | `SeriesRepository`, custom variable expressions |
| `sql_utils.py` | `SQLConnection`, `DataRetrieval` (compat layer over repository) |
| `styling.py` | Colors, labels, `FinishedFigure` |
| `global_classes.py` | `VariableOutput` arithmetic for custom variables |
| `scripts/` | Migration, validation, legacy Excel tooling (`processing.py`) |

Backward-compatible re-exports: `analysis.py`, `figure.py`.

## Legacy / out of scope

- STRESS platform integration has been removed.
- `scripts/processing.py` is legacy CSV/Excel tooling; the live app reads from MySQL only.
- `all_data_jan_2024` and old `archive.py` publication figure workflows are not maintained here.
