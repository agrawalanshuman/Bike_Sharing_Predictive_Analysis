"""Fixed experiment constants and repository location."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 835
DATA_SHA256 = 'e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f'
CATEGORICAL = ['hour', 'dow', 'month', 'holiday', 'workingday']
NUMERIC = ['lag_24', 'lag_48', 'lag_168', 'roll_mean_7d', 'roll_max_7d',
           'roll_count_7d', 'commute_workday', 'hour_sin', 'hour_cos', 'trend_days']
FEATURES = CATEGORICAL + NUMERIC
EXCLUDED = ['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum', 'windspeed',
            'weathersit', 'instant', 'dteday', 'yr', 'mnth', 'hr', 'weekday', 'season']


STAFFING = {'throughput':40., 'hourly_wage':20., 'under_cost':5., 'excess_cost':1.,
            'max_per_shift':20, 'assignment_budget':48, 'shift_hours':4}
