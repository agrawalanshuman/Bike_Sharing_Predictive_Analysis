# Bike Sharing Predictive Analysis — assessment map

**Author and Canvas project name: Anshuman · ISOM 835**

This directory holds Anshuman's original milestone materials and a concise mapping from the course requirements to the portfolio deliverables. The repository's main entry point is the [project README](../README.md). Lecture notes, instructor notebooks and copied course-site source files are not distributed here.

## Deliverables

| Course item | Project evidence | Course date |
|---|---|---|
| Milestone 1: problem and data | [`milestone1_canvas.txt`](milestone1_canvas.txt): one paragraph naming author, unit, target, horizon, decision owner, current process, public source and leakage column | November 2, 2026 |
| Milestone 2: baseline with a metric | [`milestone2.ipynb`](milestone2.ipynb): real-data audit, EDA, forecast-time features, chronological split, seasonal baseline, Ridge pipeline, MAE/RMSE and held-out plots | November 23, 2026 |
| Milestone 3: tuned model and error analysis | [`milestone3.ipynb`](milestone3.ipynb): training-only tuning, regression intervals, three SHAP cases, sliced errors and model card | November 30, 2026 |
| Milestone 4: presentation dry run | [`reports/presentation.pdf`](../reports/presentation.pdf): ten slides for an eight-minute business presentation | December 7, 2026 |
| Final integrated analysis | [`analysis.ipynb`](../analysis.ipynb): executed complete workflow and AI-assistance disclosure | December 14, 2026 |
| Final decision memo | [`reports/memo.pdf`](../reports/memo.pdf): two pages with Question, Data and Model, Honest Evaluation, Decision and Money, and Trust Story | December 14, 2026 |
| Public repository | README, source data citation, one Python implementation, notebook, figures and reports in [Bike_Sharing_Predictive_Analysis](https://github.com/agrawalanshuman/Bike_Sharing_Predictive_Analysis) | December 14, 2026 |

The course project page lists Milestones 1–3 at 11:59 PM, dry-run slides at 5 PM, and final deliverables at 5 PM. A Session 11 notebook contains a conflicting December 13 note; confirm the applicable deadline in Canvas.

## Evidence against the assessment criteria

| Criterion | Where it is addressed |
|---|---|
| Prediction framing and real data | Unit = one system-wide hour; target = completed rentals; next-day 24-hour horizon issued at 11:59 PM; hypothetical operations manager; 17,379 original UCI observations |
| Leakage prevention | No `casual`, `registered`, current `cnt` or future observed weather inputs; missing targets retained; timestamp-based lag and rolling-window checks |
| Validation and baseline | Nonoverlapping proper-training, calibration and eight-complete-week test periods; seasonal naive and fitted models scored on the same observed hours; daily rolling-origin feature updates |
| First model and preprocessing | Ridge inside ColumnTransformer/Pipeline; training-only imputation, scaling and encoding; MAE and RMSE reported in rentals/hour |
| Tuning | Eight seeded random configurations over learning rate, leaf count and regularization; expanding chronological folds strictly inside proper training |
| Original forecasting features | `lag_48`, rolling maximum, observed-history count, and working-day/commute interaction extend the basic lag/calendar example |
| Trust and explanations | Three genuine SHAP waterfalls for low, moderate and high predictions, with business explanations and additivity checks |
| Uncertainty | MAPIE regression intervals calibrated before testing; empirical coverage and width reported; temporal-shift limitations discussed |
| Error analysis | MAE/RMSE by hour, commute period, weekday/weekend, season and observed weather situation; group counts retained. Weather describes errors after prediction and never enters the forecast features |
| Model card | [`reports/model_card.md`](../reports/model_card.md): owner, intended use, data, metrics, weak slices, uncertainty, failure modes, operating limits and monitoring |
| Business value | Fixed aggregate-capacity planning rule, explicit asymmetric cost assumptions, nine-scenario sensitivity table, and limits on causal savings and unmet-demand interpretation |
| Business communication | Executive narrative in the ten-slide presentation; five required sections in the two-page memo; a reproducible public README |
| Reproducibility and safety | One Python file, pinned dependencies, visible notebook outputs, source checksum, 17 forecasting and optimization checks and artifact verification |
| Academic honesty | Anshuman credited as author; AI assistance disclosed; source data and learning references cited; no invented scores, participation or deployment claims |

The Ridge model has the lowest held-out MAE. The advanced model is reported honestly even though it trails Ridge. Nominal 90% intervals attain 85.64% coverage, and this shortfall is part of the evaluation rather than hidden or retuned away.

## Course actions still performed by the author

Read [`milestone1_discussion_guidance.md`](milestone1_discussion_guidance.md), complete the required Canvas discussion participation, review all deliverables and submit them through the course's approved process. Rehearse the presentation and be prepared to explain the split, leakage controls, baseline, feature availability, SHAP limitations and operational assumptions. Repository files do not establish that any submission or discussion action occurred.

## Learning references and assistance

The project was informed by the publicly available ISOM 835 learning materials:

- [Project brief](https://isom-835.vercel.app/project/)
- [Dataset catalogue](https://isom-835.vercel.app/datasets/)
- [Course cheatsheet](https://isom-835.vercel.app/cheatsheet/)
- [Session 3: feature pipelines](https://isom-835.vercel.app/sessions/session-03/)
- [Session 9: tuning, explanations and uncertainty](https://isom-835.vercel.app/sessions/session-09/)
- [Session 11: forecasting](https://isom-835.vercel.app/sessions/session-11/)
- [Public course-source repository](https://github.com/harslan/isom-835), inspected at commit `a3dd676287524f6fbcf7c7db43a7f31c54dd1047` when the course website was unavailable from the development environment.

The implementation adapts the concepts to a stricter forecast-time contract: it avoids the lesson example's synthetic fallback, target interpolation, future observed weather inputs and overlapping split boundary. These differences are documented in the analysis. OpenAI Codex assisted with code, checks, analysis presentation and drafting. Anshuman must verify and understand the work; attribution does not transfer that responsibility.

## Additional operations-research extension

The integrated analysis and Milestone 3 now include a Gurobi mixed-integer shift-assignment model,
complete-day comparisons, solver certificates and fixed-policy cost sensitivity. Plotly adds an
interactive daily-cost report. This extends the business-decision evidence while retaining the
required regression models, training-only tuning, SHAP cases and MAPIE intervals. The new policy
analysis is explicitly retrospective; operating assumptions and licensing requirements are documented
in the main README. All implementation and 17 checks remain in `bike_sharing.py`.
