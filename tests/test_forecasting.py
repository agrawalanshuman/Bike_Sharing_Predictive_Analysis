import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from bike_share.config import PROJECT_ROOT as ROOT, FEATURES, NUMERIC, STAFFING
from bike_share.forecasting import build_pipeline, fit_first, forecast_day, run_advanced
from bike_share.evaluation import score


def test_imputer_statistics_from_proper_train_only(d):
    train=d['X'].loc[d['fit']]; model=build_pipeline('ridge').fit(train,d['y'].loc[d['fit']])
    imputer=model.named_steps['prep'].named_transformers_['numeric'].named_steps['impute']
    assert np.allclose(imputer.statistics_,train[NUMERIC].median().to_numpy())
    old=imputer.statistics_.copy()
    adversarial=d['X'].loc[d['test']].copy();adversarial[NUMERIC]=1e12
    model.predict(adversarial)
    assert np.array_equal(old,imputer.statistics_)


def test_daily_api_produces_24_forecasts_without_accessing_target_day(d):
    model=build_pipeline('ridge').fit(d['X'].loc[d['fit']],d['y'].loc[d['fit']])
    day=d['test'][0].normalize()
    holiday=int(d['grid'].loc[day,'holiday'])
    first=forecast_day(model,d['grid'],day,holiday)
    poisoned=d['grid'].copy();poisoned.loc[day:,:]=999999
    second=forecast_day(model,poisoned,day,holiday)
    pd.testing.assert_frame_equal(first,second)
    assert len(first)==24 and first.forecast.notna().all()
    expected=np.maximum(0,model.predict(d['X'].loc[first.index]))
    assert np.allclose(first.forecast,expected)


def test_unknown_pipeline_rejected():
    import unittest
    with unittest.TestCase().assertRaises(ValueError):
        build_pipeline('typo')


def test_ridge_matches_saved_predictions(d,ridge):
    _, predictions=ridge
    saved=pd.read_csv(ROOT/'results/predictions.csv',index_col=0,parse_dates=True)
    assert saved.index.equals(d['test'])
    # Numerical comparison tolerates CSV parsing at machine precision, not changed forecasts.
    np.testing.assert_allclose(predictions,saved.ridge,rtol=1e-12,atol=1e-12)


def test_ridge_fit_is_reproducible(d,ridge):
    _, repeated=fit_first(d)
    np.testing.assert_array_equal(ridge[1],repeated)


@pytest.mark.parametrize('date', ['2012-11-05','2012-11-19','2012-12-25'])
def test_daily_prediction_shape_and_clipping(d,ridge,date):
    day=pd.Timestamp(date)
    prediction=forecast_day(ridge[0],d['grid'],day,int(d['grid'].loc[day,'holiday']))
    assert prediction.index.equals(pd.date_range(day,periods=24,freq='h',name='timestamp'))
    assert prediction.forecast.shape==(24,)
    assert np.isfinite(prediction.forecast).all() and prediction.forecast.ge(0).all()


@pytest.mark.slow
def test_boosting_and_intervals_reproduce_saved_experiment(d,ridge):
    result=run_advanced(d,ridge[1])
    saved=pd.read_csv(ROOT/'results/predictions.csv',index_col=0,parse_dates=True)
    pd.testing.assert_frame_equal(result['predictions'],saved,check_freq=False,rtol=1e-12,atol=1e-12)
    metrics=pd.read_csv(ROOT/'results/metrics.csv',index_col=0)
    pd.testing.assert_frame_equal(result['metrics'],metrics,rtol=1e-12,atol=1e-12)
