"""Generate readable notebooks; execution is a separate explicit step."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]

def md(s):return nbf.v4.new_markdown_cell(s.strip())
def code(s):return nbf.v4.new_code_cell(s.strip())

intro='''# Predicting Next-Day Hourly Bike-Sharing Demand Using Python and Machine Learning
**ISOM 835 · TeamName · [NAMES TO FILL IN]**

One row is one system-wide hour; the regression target is `cnt`, observed rental trips.
At **11:59 PM on the previous calendar day**, forecast all **24 hourly counts for the next day**
(a lead time of about 1 minute to 23 hours 1 minute to hourly slot starts).
A hypothetical operations manager currently uses last week's same-hour rentals to plan aggregate
service capacity and staffing; we compare that rule with machine learning. These data cannot allocate
bikes to individual stations, and observed trips do not measure unmet demand.

Primary metric: **MAE in rentals/hour**; secondary: RMSE. No random time-series splitting.
This is a historical rolling-origin evaluation, not a claim of deployment.
'''
setup='''from pathlib import Path
import os, sys
# Limit nested numeric parallelism for reproducible, modest CPU use.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bike-mpl')
ROOT = Path.cwd()
if not (ROOT / 'src/forecasting.py').exists():
    raise RuntimeError('Run this notebook from the repository root; see README for Colab checkout steps.')
sys.path.insert(0, str(ROOT))
import json, numpy as np, pandas as pd
from IPython.display import display, Markdown, Image
from src.forecasting import (prepare, fit_first, run_advanced, save_results,
                             score, FEATURES, NUMERIC, CATEGORICAL, issue_times)
from src.plots import eda, evaluation_plots, explain_cases, business_plot
pd.set_option('display.max_columns', 20)
d = prepare()
print(json.dumps(d['audit'], indent=2))
'''
audit='''## 1. Real data and audit
The uploaded UCI `hour.csv` is the source, never a synthetic fallback. The data-loading script verifies
its SHA-256 checksum. `day.csv` is retained as source context, not used as an additional target or future
aggregate. The full hourly grid exposes missing records; **missing target values stay NaN**.
The audit covers the whole file to define coverage and split dates. EDA used to understand modeling
patterns is restricted to the proper fitting period, before calibration and testing.
'''
leakage='''## 2. Leakage audit and forecast-time availability
| Raw columns | Treatment and reason |
|---|---|
| `cnt` | Target only. Never a contemporaneous feature. |
| `casual`, `registered` | Excluded: components whose sum is `cnt`. |
| `temp`, `atemp`, `hum`, `windspeed`, `weathersit` | Excluded: observed at the future target hour; no historical forecast archive exists here. No future-aware interpolation. |
| `instant` | Excluded: arbitrary row identifier. |
| `dteday` | Index construction only, not a raw predictor. |
| `hr`, `weekday`, `mnth`, `yr` | Replaced by explicitly derived hour, weekday, month, and elapsed calendar time. |
| `season` | Excluded: month captures known annual calendar information. Error slices use explicit meteorological seasons. |
| `holiday`, `workingday` | Calendar information assumed published before issuance; holiday is mapped by date, workingday recomputed from weekday and holiday. No target-dependent filling. |

`lag_24`, `lag_48`, `lag_168` reference the **exact timestamp**, including missing hours.
For a target at time t, seven-day mean, maximum, and observed count use the 168 slots
**[t − 191 hours, t − 24 hours]**. Means/maxima ignore missing historical observations; the count exposes
how much history was observed. Missing lag features are median-imputed inside each training pipeline.
No target is imputed for fitting or scoring.

All historical sources end no later than the previous day's same hour, hence before the common
11:59 PM issuance. Assuming that an hour's count is finalized by 11:59 PM that day is an explicit
operational assumption; a reporting delay would require an additional lag.
The calendar itself is known in advance. Tests perturb all target-day and later counts/weather and
check that all 24 forecast rows remain unchanged.

Original extensions beyond Session 11 include **lag_48**, **seven-day rolling maximum**,
**workday × commute-hour interaction**, and observed-history count. Cyclic hour features and a calendar
trend are also included. These choices were fixed before viewing test metrics; no feature was kept or
removed based on test performance.
'''
split='''## 3. Chronological splitting and evaluation contract
The last **eight complete Monday–Sunday calendar weeks** are the held-out test window.
December 31 is a partial terminal week and is not used for model selection or testing.
Four preceding weeks are reserved exclusively for conformal calibration, including in Milestone 2 so
that the first model stays comparable in later notebooks. All earlier observed rows after the fixed
191-hour history warm-up train both models. Dates and row counts are printed below.

All methods are scored on **identical observed target hours with an observed 168-hour lag**.
Missing baseline references cause exclusion from the common metric sample; they are never fabricated.
This can bias evaluation toward hours with good history and is a stated limitation. Models can still
forecast all 24 hours using train-only imputation; only scoring requires observed targets and baseline.

The fitted model is frozen through calibration and testing. For each next day, past test-day observations
may refresh lag features after they have actually occurred. This is rolling-origin **daily feature
updating**, not a single 56-day forecast and not test-target fitting. No current-day count is used in
any of that day's 24 forecasts.
'''
first='''## 4. Seasonal baseline and first regression
The baseline is the observed count exactly 168 hours earlier. Ridge regression (`alpha=10`, fixed in
advance) uses a `ColumnTransformer` plus `Pipeline`: median imputation and scaling for numeric features,
one-hot calendar encoding with unknown-category handling, then regression. Every fitted transformation
sees proper training only. Negative predictions are clipped to zero as a prespecified physical constraint.
'''
advanced='''## 6. Advanced model: tuning entirely inside proper training
HistGradientBoostingRegressor is tuned on three meaningful knobs: learning rate, leaf count, and L2
regularization. Eight seeded random configurations use three expanding chronological TimeSeriesSplit
folds. Each validation block has 672 observed rows; a 24-row gap conservatively keeps training labels
before the first validation day's issuance, even with missing hours. Folds may begin within a calendar
day because this is a row-based splitter; the availability assertion below ensures this is still safe.
No randomized early-stopping split is used (`early_stopping=False`, 220 iterations fixed in advance).
The winning configuration is chosen by mean validation MAE, refit on proper training, and frozen.
Calibration and test outcomes never choose hyperparameters. More complex does not guarantee better.
'''
uncertainty='''## 7. Regression prediction intervals with MAPIE
`SplitConformalRegressor(prefit=True, confidence_level=0.9)` calibrates the frozen tuned model on a
separate chronological four-week block. Nonnegative bounds are prespecified because rental counts
cannot be negative. No model refit occurs after calibration. Coverage is the fraction of held-out
actuals inside the interval, and width is measured in rentals/hour on the same common test sample.

Split conformal's usual marginal guarantee requires exchangeability. Serial dependence, changing
rental volumes, weather, and temporal distribution shift violate that assumption here. **90% is the
nominal target, not a guaranteed time-series or per-hour coverage claim.** Report the measured value,
including any shortfall; intervals are not a promise of station-level service capacity.
'''
business='''## 10. Business decision and transparent money assumptions
Hypothetical daily decision: plan aggregate hourly rental-service capacity as `ceil(max(0, forecast))`.
This is a rental-throughput planning unit, not unique bicycles, a station allocation, or an optimized
staffing roster. Convert to staffing only after validating operational throughput constraints.
The same fixed rule is applied to both forecasts, without selecting an offset on test data.

Assume **$5 for each rental-unit of underplanned capacity** and **$1 for each excess planned unit**.
Proxy cost = 5 × max(actual − planned, 0) + 1 × max(planned − actual, 0), summed across identical test
hours. These are classroom assumptions, not verified operating costs. The sensitivity grid varies
shortage cost across $2/$5/$10 and excess cost across $0.50/$1/$2. It evaluates the fixed decisions;
it does not optimize an asymmetric-loss policy on the test set.

Counts record completed rentals. Stockouts, unavailable docks, weather cancellations, and unobserved
customers mean completed trips are an imperfect proxy for latent demand. Cost differences are
conditional historical scenarios, **not causal savings or evidence of deployment**. Point forecasts
optimized for MAE need not minimize asymmetric cost; a future cost-aware policy must be selected using
new training/calibration data, then evaluated on a new holdout.
'''
ai='''## Academic honesty and reproducibility
OpenAI Codex assisted with code, documentation, tests, charts, and drafting the memo/slides. All reported
results come from executed code on the uploaded real data. Team members must review and understand the
splits, leakage audit, baseline, SHAP, interval assumptions, and cost proxy; names remain placeholders.
No GitHub publication or Canvas submission was performed. Course resources were read through the
public `harslan/isom-835` GitHub source because the website and Colab domains returned proxy 403 errors.
See `references/course_resources.md` for provenance and differences from the teaching example.

The example's target interpolation, synthetic fallback, future observed weather, and inclusive split
boundary were deliberately replaced to satisfy this project's stricter forecast-time rules. SHAP
explains predictions, not causality. No station-level claim or verified savings claim is made.
'''

def common():
    return [md(intro),md(audit),code(setup),code('''display(d['raw'].head())
display(d['raw'].dtypes.rename('dtype').to_frame())
display(d['raw']['cnt'].describe().to_frame('Full-source target distribution'))
print('Duplicate timestamps:', d['raw'].index.duplicated().sum())
print('Original missing cells:', int(d['raw'].isna().sum().sum()))
ed a = None
'''.replace('ed a = None',"display(Image(filename=str(eda(d))))")),md(leakage),code('''display(d['X'].loc[d['fit'][:5]])
print('Features:', FEATURES)
for lag in (24,48,168):
    assert ((d['X'].index-pd.Timedelta(hours=lag)) < issue_times(d['X'].index)).all()
assert ((d['X'].index-pd.Timedelta(hours=24)) < issue_times(d['X'].index)).all()
print('All lag and rolling-window end timestamps precede daily issuance.')'''),md(split),code('''display(pd.Series(d['audit']).to_frame('Value'))
print('No overlap:', d['fit'].max() < d['cal'].min() < d['test'].min())
print('Scheduled test hours:', d['audit']['test_scheduled_hours'])
print('Missing actuals excluded:', d['audit']['test_missing_actuals'])
print('Missing baseline with observed target excluded:', d['audit']['test_missing_week_lag_with_actual'])'''),md(first),code('''ridge_model, ridge_pred = fit_first(d)
first_predictions = pd.DataFrame({'actual': d['y'].loc[d['test']],
    'seasonal_naive': d['X'].loc[d['test'], 'lag_168'], 'ridge': ridge_pred})
first_metrics = pd.DataFrame({name: score(first_predictions.actual, first_predictions[name])
                              for name in ['seasonal_naive','ridge']}).T
gain = first_metrics.loc['seasonal_naive','MAE']-first_metrics.loc['ridge','MAE']
display(first_metrics)
print(f'Ridge absolute MAE improvement: {gain:.3f} rentals/hour; percentage: {100*gain/first_metrics.iloc[0].MAE:.2f}%')
print(ridge_model)
# A genuine daily API issues 24 forecasts from strictly prior-day history.
from src.forecasting import forecast_day
first_day = d['test'][0].normalize()
daily_forecast = forecast_day(ridge_model, d['grid'], first_day, holiday=int(d['grid'].loc[first_day, 'holiday']))
assert len(daily_forecast) == 24
display(daily_forecast)
first_metrics.to_csv(ROOT/'results/milestone2_metrics.csv')'''),md('## 5. Held-out week and residuals\nThe plotted week is the first test week, specified in advance. All lines use the common scoring rows.'),code('''evaluation_plots(first_predictions)
display(Image(filename=str(ROOT/'visualizations/milestone2_week.png')))
display(Image(filename=str(ROOT/'visualizations/milestone2_residuals.png')))''')]

def extended():
    return [md(advanced),code('''a = run_advanced(d, ridge_pred)
summary = save_results(d, a)
print('Selected parameters:', summary['best_params'])
print('Mean proper-training CV MAE:', summary['best_cv_MAE'])
display(pd.DataFrame(a['cv_boundaries']))
display(pd.DataFrame(a['search'].cv_results_)[['params','mean_test_score','std_test_score','rank_test_score']].sort_values('rank_test_score'))
display(a['metrics'])
best = a['metrics'].MAE.idxmin()
display(Markdown(f"**Lowest test MAE: {best}.** These test results compare prespecified workflows; they were not used to retune any model. A deployment choice still requires a new pilot or holdout."))
evaluation_plots(a['predictions'], a['slices'])
display(Image(filename=str(ROOT/'visualizations/actual_vs_predicted.png')))
display(Image(filename=str(ROOT/'visualizations/residuals.png')))'''),md(uncertainty),code('''display(pd.Series(a['intervals']).to_frame('Result'))
display(Image(filename=str(ROOT/'visualizations/intervals.png')))
coverage = a['intervals']['empirical_coverage']
print(f'Observed coverage: {coverage:.2%}; nominal target: 90%; shortfall: {max(0,.9-coverage):.2%} points.')'''),md('''## 8. Three genuine SHAP waterfall explanations
Cases are selected by the lowest, median, and highest tuned-model forecast on common test hours,
not chosen for attractive outcomes. The fitted pipeline supplies the transformed feature names.
Tree SHAP values add to the **raw** model prediction; clipping any negative count to zero is a separate
physical rule. The additivity assertion checks each explanation. Correlated lags may share attribution;
these charts are model explanations, not causal analyses.'''),code('''cases = explain_cases(a['model'], d, a['predictions'])
(ROOT/'results/shap_cases.json').write_text(json.dumps(cases, indent=2))
for case in cases:
    display(Markdown('### '+case['case'].title()+' demand: '+case['explanation']))
    display(Image(filename=str(ROOT/f"visualizations/shap_{case['case']}.png")))'''),md('''## 9. Error slices
MAE and RMSE are computed by hour, commute/noncommute, weekday/weekend, and meteorological season.
These test weeks contain autumn and winter only; no spring/summer generalization claim follows.
Small groups are noisy, and the worst subgroup is descriptive, not a new tuning target.'''),code('''display(a['slices'])
boost_slices = a['slices'].query("model == 'boosting'")
worst = boost_slices.loc[boost_slices.MAE.idxmax()]
display(Markdown(f"Worst boosting slice by MAE: **{worst['dimension']} = {worst['slice']}**, n={int(worst['n'])}, MAE={worst['MAE']:.2f}, RMSE={worst['RMSE']:.2f}."))
display(Image(filename=str(ROOT/'visualizations/sliced_errors.png')))'''),md(business),code('''display(a['costs'])
business_plot(a['costs'])
base_cost = a['costs'].query('under_cost == 5 and excess_cost == 1').iloc[0]
print(f"Assumed-cost reduction versus seasonal baseline: ${base_cost.proxy_cost_reduction:,.2f} across {len(d['test'])} common test hours.")
display(Image(filename=str(ROOT/'visualizations/business_costs.png')))'''),md('## 11. Model card\nThe following concise card is generated from this execution, including the measured limitations.'),code('''from src.documents import model_card
card = model_card(d, a)
(ROOT/'model_card.md').write_text(card)
display(Markdown(card))''')]

for filename,advanced_flag in [('TeamName_milestone2.ipynb',False),('TeamName_milestone3.ipynb',True),('BikeSharing_Final.ipynb',True)]:
    cells=common()+(extended() if advanced_flag else [])+[md(ai)]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,ROOT/filename)
    print('Created',filename)
