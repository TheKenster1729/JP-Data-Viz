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

4. Both databases are already loaded. To add data, see [Adding raw Excel data](#adding-raw-excel-data).

## Adding raw Excel data

`scripts/ingest_excel.py` goes from workbooks to what the app reads in one step. It checks the format, writes `series_values` and the dimension tables, reads every series back to verify it before committing, and adds new outputs to the dropdown CSV.

```bash
# 1. put the workbooks under the dataset's raw folder, one subfolder per scenario
#    e.g. Raw Data/New_Ensembles/2C_med/12_my_new_output_units_2C_med.xlsx
# 2. check them without writing anything
python scripts/ingest_excel.py --dataset all_data_aug_2024 "Raw Data/New_Ensembles/2C_med" --dry-run
# 3. load them
python scripts/ingest_excel.py --dataset all_data_aug_2024 "Raw Data/New_Ensembles/2C_med"
# 4. restart the app
```

Series already in the database are skipped unless you pass `--replace`. Each workbook is loaded in a single transaction, so a workbook that fails loads nothing. Use `--create-database` to start a new database.

**Workbook format** (both existing layouts are accepted):

| | Rule |
|---|---|
| File name | `{n}_{output}_{scenario}.xlsx`. The number is ignored. The scenario is the database code with no periods (`About15C_med`), matching its folder or an existing scenario. |
| Sheets | One per region code (`GLB`, `USA`, …). `Data Note` and unknown names are skipped. |
| Row 1 | Year headers (`2020` … `2100`), as numbers or text. Years off the app's 5-year grid are reported and skipped unless you pass `--all-years`. |
| Run column | The column immediately left of the first year. Columns further left (units, labels) are ignored. |
| Cells | Numbers. `Eps` → 0. An empty cell → NULL. Any other text rejects the workbook. |

Further steps for things the app hard-codes:

- **New outputs** are appended to `display_names.csv` (full dataset) or `publication_output_names.csv`, with the raw name as the display name. Edit that label to taste.
- **New scenarios** load fine but appear in dropdowns only once they're added to `Options.scenarios` and `scenario_display_names` in `styling.py`, and to `SCENARIOS` in `datasets.py`.
- **A new database** must be added to `SCENARIOS` in `datasets.py` before the app will open it.

Do not re-run `scripts/migrate_schema.py` after ingesting. It rebuilds `series_values` from the legacy per-series tables, and it refuses to run if that would delete ingested series. To audit a whole dataset against its workbooks, use `python scripts/validate_against_excel.py --dataset <name>`.

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
| `scripts/` | Excel ingest (`ingest_excel.py`), migration, validation, legacy tooling (`processing.py`) |

Backward-compatible re-exports: `analysis.py`, `figure.py`.

## Legacy / out of scope

- STRESS platform integration has been removed.
- `scripts/processing.py` is legacy CSV/Excel tooling; the live app reads from MySQL only.
- `all_data_jan_2024` and old `archive.py` publication figure workflows are not maintained here.
