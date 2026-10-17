"""Tests target availability rules and preprocessing, not merely code syntax."""
import numpy as np
import pandas as pd
import pytest
from src.forecasting import (prepare, make_features, issue_times, FEATURES,
                             NUMERIC, build_pipeline)

@pytest.fixture(scope='module')
def d():return prepare()

@pytest.mark.parametrize('lag',[24,48,168])
def test_lags_equal_exact_timestamp_observation(d,lag):
    pd.testing.assert_series_equal(d['X'][f'lag_{lag}'],d['y'].shift(lag),check_names=False)
    assert ((d['X'].index-pd.Timedelta(hours=lag)) < issue_times(d['X'].index)).all()

@pytest.mark.parametrize('operation,column',[('mean','roll_mean_7d'),('max','roll_max_7d'),('count','roll_count_7d')])
def test_rolling_window_has_only_available_observations(d,operation,column):
    # Independently inspect three complete 168-slot windows, including missing data.
    for t in pd.to_datetime(['2011-01-10 23:00','2012-10-30 12:00','2012-12-25 00:00']):
        history=d['y'].loc[t-pd.Timedelta(hours=191):t-pd.Timedelta(hours=24)]
        assert len(history)==168
        assert history.index.max()<issue_times(pd.DatetimeIndex([t]))[0]
        expected=getattr(history,operation)()
        assert np.isclose(d['X'].loc[t,column],expected,equal_nan=True)


def test_no_target_day_or_future_observation_changes_forecast_features(d):
    target=pd.Timestamp('2012-11-19')
    changed=d['grid'].copy()
    cols=['cnt','casual','registered','temp','atemp','hum','windspeed','weathersit']
    changed.loc[target:,cols]=999999
    day=pd.date_range(target,periods=24,freq='h')
    pd.testing.assert_frame_equal(make_features(changed).loc[day],d['X'].loc[day])


def test_features_exclude_target_components_and_future_weather():
    forbidden={'casual','registered','cnt','temp','atemp','hum','windspeed','weathersit','instant'}
    assert not forbidden.intersection(FEATURES)


def test_chronology_and_complete_test_week_bounds(d):
    assert d['fit'].max()<issue_times(d['cal'][:1])[0]
    assert d['cal'].max()<issue_times(d['test'][:1])[0]
    start=pd.Timestamp(d['audit']['test_start']);end=pd.Timestamp(d['audit']['test_end'])
    assert start.dayofweek==0 and start.hour==0
    assert end.dayofweek==6 and end.hour==23
    assert end+pd.Timedelta(hours=1)-start==pd.Timedelta(weeks=8)


def test_missing_targets_never_filled_and_test_rows_comparable(d):
    assert d['grid']['cnt'].isna().sum()==165
    assert d['y'].loc[d['test']].notna().all()
    assert d['X'].loc[d['test'],'lag_168'].notna().all()
    missing=d['grid'].index.difference(d['raw'].index)
    assert d['y'].loc[missing].isna().all()


def test_imputer_statistics_from_proper_train_only(d):
    train=d['X'].loc[d['fit']]; model=build_pipeline('ridge').fit(train,d['y'].loc[d['fit']])
    imputer=model.named_steps['prep'].named_transformers_['numeric'].named_steps['impute']
    assert np.allclose(imputer.statistics_,train[NUMERIC].median().to_numpy())
    old=imputer.statistics_.copy()
    adversarial=d['X'].loc[d['test']].copy();adversarial[NUMERIC]=1e12
    model.predict(adversarial)
    assert np.array_equal(old,imputer.statistics_)


def test_rolling_origin_can_use_only_already_observed_test_days(d):
    t=pd.Timestamp('2012-12-05 18:00')
    assert t-pd.Timedelta(hours=24) in d['test']
    assert t-pd.Timedelta(hours=24)<issue_times(pd.DatetimeIndex([t]))[0]


def test_daily_api_produces_24_forecasts_without_accessing_target_day(d):
    from src.forecasting import forecast_day
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
