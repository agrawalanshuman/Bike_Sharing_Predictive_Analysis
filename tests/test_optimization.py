import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from bike_share.config import PROJECT_ROOT as ROOT, FEATURES, NUMERIC, STAFFING
from bike_share.optimization import staffing_cost, optimize_staffing


def test_cost_rejects_invalid_actuals_and_plans():
    import unittest
    for actual in [10, np.zeros((24,1)), np.full(24,np.nan), np.full(24,-1.)]:
        with unittest.TestCase().assertRaises(ValueError):
            staffing_cost(actual, np.zeros(6))
    for plan in [np.zeros(5), np.ones(6)*.5, np.full(6,9), np.full(6,-1), np.full(6,np.inf)]:
        with unittest.TestCase().assertRaises(ValueError):
            staffing_cost(np.zeros(24),plan)


def test_assumptions_rejected_before_solver():
    import unittest
    from unittest.mock import patch
    invalid = [{**STAFFING,'under_cost':v} for v in [np.nan,np.inf,0,-1,True,'5']]
    invalid += [{**STAFFING,'assignment_budget':1.5}, {**STAFFING,'shift_hours':8}, {'under_cost':5}]
    with patch.dict('sys.modules', {'gurobipy':None}):
        for assumptions in invalid:
            with unittest.TestCase().assertRaises(ValueError):
                optimize_staffing(np.zeros(24),assumptions)


@pytest.mark.gurobi
def test_optimizer_matches_enumeration():
    import itertools
    p={**STAFFING,'max_per_shift':2,'assignment_budget':3}
    f=np.repeat([5,90,50,0,80,20],4)
    workers,certificate=optimize_staffing(f,p)
    best=min(staffing_cost(f,x,p)['total_cost'] for x in itertools.product(range(3),repeat=6) if sum(x)<=3)
    assert np.isclose(certificate['objective'],best)
    assert np.isclose(staffing_cost(f,workers,p)['total_cost'],best)


@pytest.mark.gurobi
def test_optimizer_feasibility_and_objective():
    f=np.arange(24)*40.
    x,c=optimize_staffing(f)
    assert np.equal(x,np.floor(x)).all() and (x>=0).all() and (x<=20).all() and x.sum()<=48
    assert np.isclose(c['objective'],staffing_cost(f,x)['total_cost']) and c['mip_gap']==0


@pytest.mark.gurobi
def test_optimizer_no_actual_input_and_zero_demand():
    import inspect
    assert list(inspect.signature(optimize_staffing).parameters)==['forecast','assumptions']
    x,c=optimize_staffing(np.zeros(24))
    assert x.sum()==0 and c['objective']==0


def test_optimizer_rejects_invalid_forecast():
    for bad in [np.zeros(23),np.full(24,np.nan),np.full(24,-1.)]:
        try:optimize_staffing(bad)
        except ValueError:continue
        raise AssertionError('Invalid forecast accepted')


@pytest.mark.gurobi
def test_solver_error_classification():
    import unittest, gurobipy as gp
    from unittest.mock import patch
    for code, message in [(gp.GRB.Error.NO_LICENSE,'license unavailable'),
                          (gp.GRB.Error.OUT_OF_MEMORY,'solver error')]:
        with patch('gurobipy.Env',side_effect=gp.GurobiError(code,'test failure')):
            with unittest.TestCase().assertRaisesRegex(RuntimeError,message):
                optimize_staffing(np.zeros(24))
