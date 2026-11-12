"""Forecast-time features and fixed chronological evaluation boundaries."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .config import FEATURES, PROJECT_ROOT
from .validation import require
from .data import load_data

def issue_times(index):
    """One common issuance for every hour of a target calendar day."""
    return index.normalize() - pd.Timedelta(minutes=1)


def make_features(grid):
    idx, y = grid.index, grid['cnt']
    require(isinstance(idx, pd.DatetimeIndex) and idx.tz is None and idx.is_unique
            and idx.is_monotonic_increasing and len(idx) > 0
            and idx.equals(pd.date_range(idx[0], idx[-1], freq='h')),
            'Features require a sorted, unique, timezone-naive full hourly grid; retain missing rows')
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


def prepare(data_file=None):
    data_file = Path(data_file or PROJECT_ROOT/'data/hour.csv')
    raw, grid = load_data(data_file)
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
             'sha256_hour_csv': hashlib.sha256(data_file.read_bytes()).hexdigest(),
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
