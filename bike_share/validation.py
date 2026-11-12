"""Explicit input contracts; no solver or file I/O."""
import numpy as np
from .config import STAFFING

def require(condition, message):
    """An input/artifact contract that remains active under Python optimization."""
    if not condition:
        raise ValueError(message)


def validate_hourly_values(values, label):
    values = np.asarray(values, dtype=float)
    require(values.shape == (24,) and np.isfinite(values).all() and (values >= 0).all(),
            f'{label} must contain 24 finite nonnegative hourly values')
    return values


def validate_staffing_assumptions(assumptions=None):
    p = dict(STAFFING if assumptions is None else assumptions)
    require(set(p) == set(STAFFING), 'Supply exactly the documented staffing assumptions')
    require(all(isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) and v > 0
                for v in p.values()), 'Staffing assumptions must be finite positive numbers')
    require(p['shift_hours'] == 4, 'This model requires four-hour shifts')
    require(all(float(p[k]).is_integer() for k in ['max_per_shift', 'assignment_budget']),
            'Assignment limits must be integers')
    return p
