"""Pure scoring, error slices and fixed-capacity cost scenarios."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def score(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)), 'RMSE': float(root_mean_squared_error(y, pred))}


def error_slices(predictions, observed_grid=None):
    rows = []
    idx = predictions.index
    groups = {'hour': pd.Series(idx.hour.astype(str), index=idx),
              'day_type': pd.Series(np.where(idx.dayofweek < 5, 'weekday', 'weekend'), index=idx),
              'season': pd.Series(np.select([idx.month.isin([12,1,2]), idx.month.isin([3,4,5]),
                    idx.month.isin([6,7,8])], ['winter','spring','summer'], default='autumn'),index=idx),
              'period': pd.Series(np.where(np.isin(idx.hour,[7,8,9,16,17,18]),'commute','other'),index=idx)}
    if observed_grid is not None:
        # Post-prediction diagnostic only: actual weather never enters model features.
        weather = observed_grid.loc[idx, 'weathersit'].map({
            1: 'clear', 2: 'mist/cloudy', 3: 'light rain/snow', 4: 'severe weather'})
        groups['observed_weather'] = weather.fillna('unknown')
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
