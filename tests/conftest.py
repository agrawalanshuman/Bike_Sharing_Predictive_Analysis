"""Shared audited data and a deterministic Ridge fit, with no solver dependency."""
import pytest
from bike_share.features import prepare
from bike_share.forecasting import fit_first

@pytest.fixture(scope='session')
def d():
    return prepare()

@pytest.fixture(scope='session')
def ridge(d):
    return fit_first(d)
