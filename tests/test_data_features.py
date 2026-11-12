import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from bike_share.config import PROJECT_ROOT as ROOT, FEATURES, NUMERIC, STAFFING
from bike_share.data import load_data, ensure_source_data
from bike_share.features import make_features, issue_times


@pytest.mark.parametrize("lag", [24,48,168])
def test_lags_equal_exact_timestamp_observation(d,lag):
    pd.testing.assert_series_equal(d['X'][f'lag_{lag}'],d['y'].shift(lag),check_names=False)
    assert ((d['X'].index-pd.Timedelta(hours=lag)) < issue_times(d['X'].index)).all()


@pytest.mark.parametrize("operation,column", [("mean","roll_mean_7d"),("max","roll_max_7d"),("count","roll_count_7d")])
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


def test_rolling_origin_can_use_only_already_observed_test_days(d):
    t=pd.Timestamp('2012-12-05 18:00')
    assert t-pd.Timedelta(hours=24) in d['test']
    assert t-pd.Timedelta(hours=24)<issue_times(pd.DatetimeIndex([t]))[0]


def test_feature_grid_rejects_ambiguous_timestamps():
    import unittest
    _, grid = load_data()
    for invalid in [grid.iloc[::-1], grid.drop(grid.index[1]),
                    pd.concat([grid.iloc[:1],grid]), grid.tz_localize('UTC')]:
        with unittest.TestCase().assertRaisesRegex(ValueError, 'hourly grid'):
            make_features(invalid)


@pytest.mark.parametrize('cached', [False, True])
def test_explicit_source_errors_never_download(tmp_path, monkeypatch, cached):
    def unexpected(*args, **kwargs):
        pytest.fail('An explicit source error must not trigger a download')
    monkeypatch.setattr('urllib.request.urlopen', unexpected)
    if cached:
        (tmp_path/'data').mkdir()
        (tmp_path/'data/hour.csv').write_bytes((ROOT/'data/hour.csv').read_bytes())
    with pytest.raises(FileNotFoundError):
        ensure_source_data(tmp_path/'missing.csv',root=tmp_path)
    bad=tmp_path/'bad.csv';bad.write_text('invalid source')
    with pytest.raises(ValueError,match='Explicit'):
        ensure_source_data(bad,root=tmp_path)


def test_no_implicit_target_interpolation_with_missing_history(d):
    missing=d['grid'].index.difference(d['raw'].index)[0]
    assert pd.isna(d['X'].loc[missing+pd.Timedelta(hours=24),'lag_24'])
    assert pd.isna(d['X'].loc[missing+pd.Timedelta(hours=168),'lag_168'])
