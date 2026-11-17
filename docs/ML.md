# ML pipeline (Phase 2)

**Current shipped model: trained on SYNTHETIC rule-generated labels. It is a demo component, not clinical evidence.**
The pipeline below is complete and tested; what is missing is a real dataset to run it on. Until one is used,
`promote --channel production` is refused by the gate.

## Data
| Dataset | Access | Role | Notes |
|---|---|---|---|
| **MIMIC-IV-ED** (nurse-assigned ESI, ~400k stays) | credentialed PhysioNet (CITI training + DUA) | the real target | Adapter `load_mimic_ed` is written against the real file schema and **verified on the open MIMIC-IV-ED demo** (100 patients, ~220 stays: schema check only, far too small to train). |
| **KTAS Korea** (expert 5-level labels, ~1.3k visits) | public download (Kaggle, free account) | external check / `research` channel | KTAS is **not** ESI. Adapter matches columns case-insensitively and errors with the column list if the file differs from my expectation (I wrote it from the published schema and could not download the file). |
| synthetic | none | plumbing / `demo` channel only | labels come from if/else rules, so metrics are circular. Every report says so. |

Put MIMIC under `backend/data/external/<name>/{ed,hosp}/` (git-ignored; the DUA forbids sharing it).

## Run it
```
python -m backend.training.train --dataset mimic --path backend/data/external/mimic-iv-ed --external-dataset ktas --external-path ktas.csv
python -m backend.training.train --dataset mimic --path ... --inspect        # data-quality report only
python -m backend.training.promote --run <run_id> --channel research
python -m backend.training.promote --run <run_id> --channel production --signoff "Dr A. Patel, ED physician"
```
Outputs land in `backend/ml_runs/<run_id>/` (model, ONNX, meta, `model_card.json`, `report.md`, parity vectors); MLflow
(`backend/ml_runs/mlflow.db`; `mlflow ui --backend-store-uri sqlite:///backend/ml_runs/mlflow.db`) indexes them.

## What the pipeline guarantees
* **Locked test set** carved off first, grouped by patient (one patient never straddles train/test or CV folds).
  Every extra evaluation of the same test set is counted in the card (`test_set.n_evaluations`).
* **Grouped stratified 5-fold CV** on development data compares rules-only, logistic regression and XGBoost.
* **Features**: respiratory rate included; explicit `<vital>_missing` indicators; NaN stays NaN (no imputing to "normal").
  Defined once in `app/ml/feature_schema.py`, used by training, server and browser.
* **Calibration**: one temperature fitted on out-of-fold predictions; applied identically on server and browser.
* **Metrics** that matter: ESI 1-2 sensitivity, under/severe-under/over-triage, QWK, macro-F1, NLL/Brier/ECE,
  bootstrap CIs, subgroups (infant, pediatric, adult, geriatric, 85+, missing critical vitals, sex).
* **Explanations**: exact TreeSHAP from XGBoost (`pred_contribs`), shown to the nurse as "what drove the ML model".
* **The shipped decision is model + deterministic rules** (`min(model ESI, rule floor)`); the rules-only baseline
  and the model alone are reported next to it.
* **Gate** (`gate_config.json`): parity, golden scenarios, regression vs the shipped model (same dataset only),
  absolute thresholds, calibration, pediatric/geriatric gaps, baselines. Channels: `demo` (any label source, only
  parity/golden/regression/baselines block), `research` (real labels, everything blocks), `production`
  (MIMIC-IV-ED labels + named clinical sign-off + everything blocks). Thresholds are an engineer's starting point;
  a clinician must set the real ones.
* **Browser == server**: ONNX exported from the same fitted model in the same step; `promote` copies both bundles;
  CI runs `frontend/scripts/verify-onnx-parity.mjs` (onnxruntime-web + the app's own `edge-math.ts`) against vectors
  computed by the server's joblib model.

## Known limits (read before trusting a number)
* `has_prior_history` means "earlier ED stay in this dataset" in MIMIC but "history documents on file" in production.
  The report includes an ablation without it; if performance depends on it, do not ship it.
* ESI labels are noisy nurse judgements; calibration measures agreement with them, not with outcomes.
* No prospective, multi-site or outcome-based validation. Not a medical device.
