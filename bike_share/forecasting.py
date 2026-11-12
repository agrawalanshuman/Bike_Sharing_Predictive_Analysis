"""Train-only pipelines, temporal tuning, calibration and daily prediction."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from .config import NUMERIC, CATEGORICAL, SEED
from .features import make_features, issue_times
from .evaluation import score, error_slices, cost_analysis
from .validation import require

def build_pipeline(kind):
    require(kind in {'ridge','boost'}, 'Unknown pipeline kind; choose ridge or boost')
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
            'metrics':metrics,'intervals':intervals,'slices':error_slices(predictions, d['grid']),
            'costs':cost_analysis(predictions)}


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
