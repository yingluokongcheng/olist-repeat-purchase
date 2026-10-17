# Reproduction and verification

## Inspect the recorded result without rerunning anything

```bash
python scripts/verify_package.py
```

The command uses only Python's standard library. It checks evidence hashes, recorded headline counts/metrics, internal links, source syntax and accidental inclusion of raw/customer-level data. It never imports the modeling scripts or fits a model.

Open `reports/audit/index.html`, `reports/v1/index.html`, `reports/v2/index.html` and `reports/v3/index.html` in a browser after downloading the repository. Their CSV/JSON files and PNG figures are directly inspectable on GitHub. These are recorded historical results, not live dashboards.

## Explicit numerical replay

**Not performed during packaging.** Historical stage scripts ran in the original workspace. The new isolation wrapper has passed static checks only; an end-to-end clean-directory replay has not been run because experiments were explicitly stopped. Do not describe this package as newly end-to-end reproduced.

If an explicit replay is wanted later, use Python 3.12 and a fresh virtual environment:

```bash
python -m venv .venv
# Activate the environment for your OS, then:
python -m pip install -r requirements.txt
python scripts/replay.py --data data/raw/olist.zip --run-dir runs/replay-001 --execute
```

Download the dataset from [the Olist publisher page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) under its terms. The ZIP must match:

```text
967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784
```

The wrapper refuses an existing run directory or a mismatching archive. It stages the unchanged historical scripts and protocols into a new workspace, creates new preservation records, and executes audit, features, V1, V2 and V3 numerical stages. It writes only under the selected directory inside `runs/`. It never overwrites `reports/` or claims new registration timestamps predate the historical analyses.

Historical scripts retain their original `work/` and `outputs/` relative paths. The wrapper intentionally provides that directory structure rather than rewriting statistical logic. Use the wrapper, not direct execution of stage scripts from the repository root. Only run scripts and load local pickle/joblib intermediates you trust.

The numerical replay produces new tables, JSON, cached data and selected models, not rebuilt historical HTML narratives. The archived report-builder source is included for provenance; some narrative counts and statements are historical snapshots rather than generic reporting templates.

## Boundaries

- No raw ZIP, extracted source tables, customer-level model frame, predictions or binary checkpoints are committed.
- Recorded LightGBM text models contain the selected trees and feature names; their hashes are checked against records of models locked before evaluation.
- Exact values can differ across platforms/library builds despite fixed seeds. Pinned versions and implementation checks make differences auditable, not impossible.
- Replay is reproduction of a fixed protocol, not permission to tune against the reused evaluation set.
