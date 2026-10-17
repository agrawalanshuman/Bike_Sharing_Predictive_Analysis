"""Leakage-safe next-calendar-day forecasting. All fitted transforms live in pipelines."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
SEED = 835
CATEGORICAL = ['hour', 'dow', 'month', 'holiday', 'workingday']
NUMERIC = ['lag_24', 'lag_48', 'lag_168', 'roll_mean_7d', 'roll_max_7d',
           'roll_count_7d', 'commute_workday', 'hour_sin', 'hour_cos', 'trend_days']
FEATURES = CATEGORICAL + NUMERIC
EXCLUDED = ['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum', 'windspeed',
            'weathersit', 'instant', 'dteday', 'yr', 'mnth', 'hr', 'weekday', 'season']


def load_data(path=None):
    """Keep missing target hours as NaN, never interpolate rental observations."""
    path = Path(path or ROOT / 'data/hour.csv')
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw['dteday']) + pd.to_timedelta(raw['hr'], unit='h')
    if ts.duplicated().any():
        raise ValueError('Duplicate hourly timestamps')
    if raw['cnt'].isna().any() or not (raw['casual'] + raw['registered']).equals(raw['cnt']):
        raise ValueError('Invalid original target data')
    raw.index = pd.DatetimeIndex(ts, name='timestamp')
    return raw.sort_index(), raw.sort_index().asfreq('h')


def issue_times(index):
    """One common issuance for every hour of a target calendar day."""
    return index.normalize() - pd.Timedelta(minutes=1)


def make_features(grid):
    idx, y = grid.index, grid['cnt']
    # Holiday metadata are treated as a published calendar, not estimated from outcomes.
    holiday_by_date = grid.groupby(idx.normalize())['holiday'].first()
    holiday = holiday_by_date.reindex(idx.normalize()).to_numpy()
    work = ((idx.dayofweek < 5) & (holiday == 0)).astype(int)
    X = pd.DataFrame({'hour': idx.hour, 'dow': idx.dayofweek, 'month': idx.month,
                      'holiday': holiday, 'workingday': work}, index=idx)
    for lag in [24, 48, 168]:
        X[f'lag_{lag}'] = y.shift(lag)
    # The window is [target-191h, target-24h], inclusive: all 168 hourly slots.
    history = y.shift(24).rolling(168, min_periods=1)
    X['roll_mean_7d'] = history.mean()
    X['roll_max_7d'] = history.max()
    X['roll_count_7d'] = history.count()
    X['commute_workday'] = (np.isin(idx.hour, [7, 8, 9, 16, 17, 18]) & (work == 1)).astype(int)
    X['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
    X['trend_days'] = (idx - pd.Timestamp('2011-01-01')) / pd.Timedelta(days=1)
    return X[FEATURES]


def prepare():
    raw, grid = load_data()
    X = make_features(grid)
    # Full Monday-Sunday weeks; Dec 31, 2012 is an incomplete terminal week.
    exclusive_end = (grid.index.max() + pd.Timedelta(hours=1)).normalize()
    test_end = exclusive_end - pd.Timedelta(days=exclusive_end.dayofweek)
    test_start = test_end - pd.Timedelta(weeks=8)
    cal_start = test_start - pd.Timedelta(weeks=4)
    enough_history = X.index >= grid.index.min() + pd.Timedelta(hours=191)
    # All fitted models can impute missing lags; baseline cannot invent a lag_168.
    observed = grid['cnt'].notna() & enough_history
    fit_idx = X.index[observed & (X.index < cal_start)]
    cal_idx = X.index[observed & (X.index >= cal_start) & (X.index < test_start)]
    scheduled_test = (X.index >= test_start) & (X.index < test_end)
    test_idx = X.index[observed & scheduled_test & X['lag_168'].notna()]
    audit = {'source_rows': len(raw), 'source_columns': len(raw.columns),
             'source_start': str(raw.index.min()), 'source_end': str(raw.index.max()),
             'full_grid_hours': len(grid), 'missing_hours': int(grid['cnt'].isna().sum()),
             'sha256_hour_csv': hashlib.sha256((ROOT/'data/hour.csv').read_bytes()).hexdigest(),
             'fit_start': str(fit_idx.min()), 'fit_end': str(fit_idx.max()), 'fit_rows': len(fit_idx),
             'calibration_start': str(cal_start), 'calibration_end': str(test_start-pd.Timedelta(hours=1)),
             'calibration_rows': len(cal_idx), 'test_start': str(test_start),
             'test_end': str(test_end-pd.Timedelta(hours=1)), 'test_rows': len(test_idx),
             'test_scheduled_hours': int(scheduled_test.sum()),
             'test_missing_actuals': int((scheduled_test & grid['cnt'].isna()).sum()),
             'test_missing_week_lag_with_actual': int((scheduled_test & grid['cnt'].notna() & X['lag_168'].isna()).sum()),
             'terminal_hours_not_tested': int((X.index >= test_end).sum())}
    return {'raw': raw, 'grid': grid, 'X': X, 'y': grid['cnt'], 'fit': fit_idx,
            'cal': cal_idx, 'test': test_idx, 'audit': audit}


def build_pipeline(kind):
    steps = [('impute', SimpleImputer(strategy='median', add_indicator=True))]
    if kind == 'ridge':
        steps.append(('scale', StandardScaler()))
    prep = ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC),
        ('calendar', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)
    ], sparse_threshold=0)
    model = Ridge(alpha=10) if kind == 'ridge' else HistGradientBoostingRegressor(
        max_iter=220, early_stopping=False, random_state=SEED)
    return Pipeline([('prep', prep), ('model', model)])


def score(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)), 'RMSE': float(root_mean_squared_error(y, pred))}


def fit_first(d):
    model = build_pipeline('ridge').fit(d['X'].loc[d['fit']], d['y'].loc[d['fit']])
    # Nonnegative clipping is a prespecified physical constraint for rental counts.
    pred = np.maximum(0, model.predict(d['X'].loc[d['test']]))
    return model, pred


def tune_boosting(d):
    X, y = d['X'].loc[d['fit']], d['y'].loc[d['fit']]
    cv = TimeSeriesSplit(n_splits=3, test_size=24*28, gap=24)
    boundaries = []
    for tr, va in cv.split(X):
        assert X.index[tr[-1]] < issue_times(X.index[va[:1]])[0]
        boundaries.append({'train_end': str(X.index[tr[-1]]), 'validation_start': str(X.index[va[0]]),
                           'validation_end': str(X.index[va[-1]])})
    search = RandomizedSearchCV(build_pipeline('boost'),
        {'model__learning_rate': [0.04, 0.08, 0.12],
         'model__max_leaf_nodes': [15, 31, 47],
         'model__l2_regularization': [0., 1., 10.]},
        n_iter=8, cv=cv, scoring='neg_mean_absolute_error', random_state=SEED,
        n_jobs=1, refit=True, error_score='raise')
    search.fit(X, y)
    return search, boundaries


def compute_intervals(model, d):
    from mapie.regression import SplitConformalRegressor
    # No refitting on calibration or test targets.
    conformal = SplitConformalRegressor(estimator=model, confidence_level=0.9, prefit=True)
    conformal.conformalize(d['X'].loc[d['cal']], d['y'].loc[d['cal']])
    point, bounds = conformal.predict_interval(d['X'].loc[d['test']])
    lower, upper = np.maximum(0, bounds[:, 0, 0]), np.maximum(0, bounds[:, 1, 0])
    truth = d['y'].loc[d['test']].to_numpy()
    return point, lower, upper, {'nominal_coverage': .9,
        'empirical_coverage': float(np.mean((truth >= lower) & (truth <= upper))),
        'average_width': float(np.mean(upper-lower)), 'calibration_rows': len(d['cal'])}


def error_slices(predictions):
    rows = []
    idx = predictions.index
    groups = {'hour': pd.Series(idx.hour.astype(str), index=idx),
              'day_type': pd.Series(np.where(idx.dayofweek < 5, 'weekday', 'weekend'), index=idx),
              'season': pd.Series(np.select([idx.month.isin([12,1,2]), idx.month.isin([3,4,5]),
                    idx.month.isin([6,7,8])], ['winter','spring','summer'], default='autumn'),index=idx),
              'period': pd.Series(np.where(np.isin(idx.hour,[7,8,9,16,17,18]),'commute','other'),index=idx)}
    for dimension, labels in groups.items():
        for value, ix in labels.groupby(labels).groups.items():
            for name in ['seasonal_naive','ridge','boosting']:
                rows.append({'dimension':dimension,'slice':value,'model':name,'n':len(ix),
                             **score(predictions.loc[ix,'actual'],predictions.loc[ix,name])})
    return pd.DataFrame(rows)


def cost_analysis(predictions):
    """Capacity = ceiling(max(0, point forecast)), fixed before viewing costs/test."""
    rows=[]
    for under in [2., 5., 10.]:
        for excess in [.5, 1., 2.]:
            totals={}
            for model in ['seasonal_naive','boosting']:
                plan=np.ceil(np.maximum(0, predictions[model]))
                shortage=np.maximum(predictions.actual-plan,0).sum()
                surplus=np.maximum(plan-predictions.actual,0).sum()
                totals[model] = float(under*shortage + excess*surplus)
            rows.append({'under_cost':under,'excess_cost':excess,**totals,
                         'proxy_cost_reduction': totals['seasonal_naive']-totals['boosting']})
    return pd.DataFrame(rows)


def run_advanced(d, ridge_pred):
    search, boundaries = tune_boosting(d)
    model = search.best_estimator_
    raw_pred, lower, upper, intervals = compute_intervals(model,d)
    predictions=pd.DataFrame({'actual':d['y'].loc[d['test']],
        'seasonal_naive':d['X'].loc[d['test'],'lag_168'], 'ridge':ridge_pred,
        'boosting':np.maximum(0,raw_pred), 'lower_90':lower,'upper_90':upper})
    metrics=pd.DataFrame({n:score(predictions.actual,predictions[n])
                         for n in ['seasonal_naive','ridge','boosting']}).T
    metrics.index.name='model'
    metrics['MAE_improvement']=metrics.loc['seasonal_naive','MAE']-metrics.MAE
    metrics['MAE_improvement_pct']=100*metrics.MAE_improvement/metrics.loc['seasonal_naive','MAE']
    return {'model':model,'search':search,'cv_boundaries':boundaries,'predictions':predictions,
            'metrics':metrics,'intervals':intervals,'slices':error_slices(predictions),
            'costs':cost_analysis(predictions)}


def save_results(d, a):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    for k in ['predictions','metrics','slices','costs']:
        a[k].to_csv(out/f'{k}.csv',index=k in ['predictions','metrics'])
    pd.DataFrame(a['search'].cv_results_).to_csv(out/'cv_results.csv',index=False)
    summary={'audit':d['audit'],'best_params':a['search'].best_params_,
             'best_cv_MAE':-a['search'].best_score_,'cv_boundaries':a['cv_boundaries'],
             'intervals':a['intervals']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def forecast_day(model, history_grid, target_date, holiday):
    """Issue 24 forecasts using only history before the requested calendar day.

    `holiday` must come from an available calendar, not inferred from future demand.
    The caller must supply a model fitted before issuance. Partial/missing historical
    hours are retained and handled by the fitted training-only imputer.
    """
    day = pd.Timestamp(target_date).normalize()
    past = history_grid.loc[history_grid.index < day].copy()
    if past.empty:
        raise ValueError('Historical observations are required')
    slots = pd.date_range(day, periods=24, freq='h', name='timestamp')
    grid = past.reindex(pd.date_range(past.index.min(), slots[-1], freq='h', name='timestamp'))
    grid.loc[slots, 'holiday'] = int(holiday)
    X = make_features(grid).loc[slots]
    return pd.DataFrame({'forecast': np.maximum(0, model.predict(X)),
                         'seasonal_naive': X['lag_168']}, index=slots)
