# Predicting Next-Day Hourly Bike-Sharing Demand

ISOM 835 Predictive Analytics and Machine Learning · **TeamName** · **[NAMES TO FILL IN]**

At 11:59 PM, forecast each of the next calendar day's 24 system-wide hourly rental counts. A hypothetical operations manager uses the forecasts to review aggregate service capacity and staffing. This is regression/forecasting of completed rentals, not classification, latent unmet demand, unique bicycles, or individual station demand.

## Verified results

Real uploaded UCI data: **17,379 observed hours**, January 1, 2011–December 31, 2012. A full hourly grid exposes **165 missing hours**; no target observations are invented.

Test window: **November 5–December 30, 2012**, the last eight complete Monday–Sunday weeks. All methods share **1,302 scored hours**. Of 1,344 scheduled hours, four lack actuals and 38 additional hours lack an observed same-hour-last-week baseline. December 31 is a partial terminal week and is excluded. Model fitting uses 15,195 observed rows through October 7; 636 observed rows from October 8–November 4 are reserved for conformal calibration. These date blocks are disjoint.

| Prespecified method | MAE (rentals/hour) | RMSE | MAE improvement vs baseline |
|---|---:|---:|---:|
| Same hour last week | 71.95 | 119.50 | — |
| Ridge regression | **64.10** | **96.10** | **10.90%** |
| Tuned histogram gradient boosting | 69.30 | 108.42 | 3.67% |

**The simpler Ridge model performs best on held-out MAE.** No test-driven retuning was performed. All three comparisons were specified before evaluation; a new operational choice needs prospective validation.

For the boosting model, MAPIE's nominal 90% intervals achieve **85.64% empirical coverage** with **251.42 rentals/hour average width**. Time dependence and temporal shift prevent an automatic exchangeability-based coverage guarantee. The executed notebooks show the actual shortfall.

With a fixed capacity rule (ceiling of nonnegative point forecasts), assumed costs of $5 per underplanned rental-unit and $1 per excess unit yield $230,005 baseline proxy cost versus $133,389 boosting proxy cost: a **$96,616 hypothetical difference across the common test sample**. These are classroom assumptions, not actual expenses, causal savings, or an optimized staffing policy. Nine cost combinations are reported in `results/costs.csv`. Ridge is the accuracy leader; the requested baseline-versus-boosting cost comparison is not an assertion that boosting is the best operational policy.

## Methodology and leakage controls

- Build the full hourly index with `asfreq('h')`; preserve missing targets and exact timestamp lags.
- Exclude `casual` and `registered`, the components of `cnt`. Exclude contemporaneous `cnt`, row identifier, and all future observed weather.
- Use known calendar features, exact 24/48/168-hour demand lags, and seven-day rolling mean/max/count over [target−191h, target−24h]. Missing historical values are handled by training-only pipeline imputation, not target interpolation.
- Add original lag_48, rolling max, observed-history count, and workday/commute interaction features beyond the teaching template.
- Ridge uses a ColumnTransformer/Pipeline with train-only median imputation, scaling, and one-hot calendar encoding. Boosting uses its own pipeline; eight random configurations tune learning rate, leaves, and L2 regularization over three chronological training-only folds with a 24-row gap. `random_state=835` applies wherever randomness occurs; randomized early stopping is disabled.
- Fit both models only before calibration. Keep parameters frozen during calibration and testing. Historical test days can refresh tomorrow's lags only after their counts have been observed at issuance.
- Score all models on the same rows. Excluding missing baseline hours limits the evaluated population; report this rather than inventing data.
- Generate genuine SHAP waterfalls for low, moderate, and high *predicted* demand, verify additivity with transformed feature names, and report hour/day-type/season slices. SHAP is not causal.

The model assumes recorded hourly counts are finalized before the 11:59 PM issue time and calendar/holiday information is available. Reporting delays require larger lags. No station inventories, unmet demand, or archived forecast weather are available. Test seasons cover autumn/winter, not a full year.

## Files and deliverables

| File | Purpose |
|---|---|
| `milestone1_canvas.txt` | Exactly one team/problem/data paragraph, due November 2 |
| `milestone1_discussion_guidance.md` | Participation checklist; no Canvas action taken |
| `TeamName_milestone2.ipynb` | Executed data audit, EDA, leakage controls, seasonal baseline, first regression |
| `TeamName_milestone3.ipynb` | Executed tuning, MAPIE intervals, SHAP, slices, cost sensitivity, model card |
| `BikeSharing_Final.ipynb` | Executed integrated final workflow |
| `TeamName_dryrun.pdf` | Ten-slide business presentation, for an eight-minute talk |
| `TeamName_memo.pdf` | Two-page memo with all five required sections |
| `model_card.md` | Concise intended-use, evaluation, limitations, monitoring card |
| `src/forecasting.py` | Readable audited feature/model/evaluation code and 24-hour forecast API |
| `src/plots.py`, `src/documents.py` | Real-result figures, SHAP, slides and memo generation |
| `scripts/download_data.py` | Verified local loading or TLS-verified UCI download, with no synthetic fallback |
| `scripts/build_notebooks.py` | Notebook source generator (regenerating clears saved outputs) |
| `scripts/execute_notebooks.py` | Explicit end-to-end execution; stops on errors |
| `scripts/build_documents.py` | Regenerate PDFs from executed results |
| `tests/test_forecast_safety.py` | Timestamp, mutation, target exclusion, split, and preprocessing tests |
| `results/` | Actual metrics, CV scores, predictions, slices, intervals and SHAP contributions |
| `visualizations/` | EDA, held-out forecasts/residuals, intervals, sliced errors, cost sensitivity, three SHAP waterfalls |
| `references/course_resources.md` | Course-source provenance and differences from the lesson example |

## Run locally or in the cloud

Tested on **Python 3.12**. CPU only; no external services, API keys, or GPU required. Exact tested direct dependencies are in `requirements.txt`; `requirements-lock.txt` also pins their transitive dependencies.

```bash
cd /workspace/ISOM_835_Project  # or your cloned repository root
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/download_data.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/execute_notebooks.py
.venv/bin/python scripts/build_documents.py
```

The uploaded data are already in `data/`; normal runs are offline after dependencies are installed. The download script checks the audited SHA-256 before use and refuses a changed source. To regenerate notebook structure after editing the generator, run `scripts/build_notebooks.py`, then reexecute all notebooks to restore visible outputs. Reexecution recomputes tuning; it does not load cached model scores. The three notebook workflows intentionally share transparent helper code to avoid divergent implementations. Read those helpers alongside the explanations.

The tests verify exact timestamp lags, rolling windows, feature availability, target/weather exclusion, chronological splitting, missing-target preservation, train-only imputer statistics, rolling-origin updates, and the daily forecast API's independence from target-day observations. `results/verification.json` records the final executed artifact checks. After regenerating deliverables, `python scripts/verify_artifacts.py` can repeat these checks; that optional PDF-page check requires the system `pdfinfo` utility (Poppler). No result is a zero-test or syntax-only substitute for execution.

### GitHub and Colab

Files are prepared locally; **nothing has been pushed, published, or submitted**. The configured GitHub remote was empty when inspected. Review names, artifacts, data attribution and course requirements before choosing to commit/push.

After *you* publish the repository to `main`, open:

- https://colab.research.google.com/github/agrawalanshuman/ISOM_835_Project/blob/main/BikeSharing_Final.ipynb
- https://colab.research.google.com/github/agrawalanshuman/ISOM_835_Project/blob/main/TeamName_milestone2.ipynb
- https://colab.research.google.com/github/agrawalanshuman/ISOM_835_Project/blob/main/TeamName_milestone3.ipynb

These are prospective links and will not work until you publish. For Colab, first clone the published repository into the runtime, change into its root, install requirements, and restart the runtime if it already imported conflicting package versions:

```python
!git clone https://github.com/agrawalanshuman/ISOM_835_Project.git /content/ISOM_835_Project
%cd /content/ISOM_835_Project
%pip install -r requirements-lock.txt
```

Then run the opened notebook from the repository root. Local cloud execution is verified; Colab execution has not been independently tested. Avoid repeating `git clone` into an existing directory.

## Course dates, sources and honesty

Milestone 2: November 23; Milestone 3: November 30; dry-run slides: December 7; final deliverables: December 14, 2026. The course project source lists the final deadline at 5 PM; a Session 11 note conflicts, so verify Canvas. TeamName and member names remain intentional placeholders. Each teammate must review and explain the work, complete discussion participation, and submit personally.

OpenAI Codex assisted with implementation, tests, analysis and drafting. All scores, intervals, SHAP plots and figures come from real executed code. The direct course website/Colab requests were blocked by the egress policy; the corresponding public course GitHub source was read instead. See `references/course_resources.md`.

Data: Fanaee-T, Hadi, and João Gama (2013), *Event labeling combining ensemble detectors and background knowledge*, Progress in Artificial Intelligence, DOI [10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3). [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). The uploaded source readme and provenance are retained in `data/`. No new license is assigned to source data or course references.
