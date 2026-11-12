"""Historical user-flow tests; deliberately run without importing Gurobi."""
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest
from bike_share.config import PROJECT_ROOT


def app():
    return AppTest.from_file(str(PROJECT_ROOT/'bike_sharing.py'),default_timeout=45).run()


def test_forecast_controls_and_missing_hours():
    page=app()
    assert not page.exception
    assert [m.value for m in page.metric[:3]]==['64.10','10.90%','1,302']
    assert any('24 of 24' in c.value for c in page.caption)
    page.selectbox(key='forecast_model').set_value('boosting').run()
    assert not page.exception and any('85.64%' in c.value for c in page.caption)
    page.checkbox[0].uncheck().run()
    page.selectbox(key='forecast_day').set_value('2012-11-05').run()
    assert not page.exception and any('1 of 24' in c.value for c in page.caption)
    page.selectbox(key='forecast_model').set_value('seasonal_naive').run()
    assert not page.exception and not page.checkbox
    assert not any(block.value.strip()=='None' for block in page.code)


def test_staffing_scenario_controls():
    page=app()
    page.selectbox(key='plan_model').set_value('ridge').run()
    expected=pd.read_csv(PROJECT_ROOT/'results/staffing_daily.csv').query("date=='2012-11-07' and method=='ridge'").iloc[0]
    assert not page.exception and page.metric[-1].value==f'${expected.total_cost:,.0f}'
    assert len(page.dataframe[-1].value)==6


@pytest.mark.parametrize('error',[FileNotFoundError('/private/path'),ValueError('invalid saved results')])
def test_missing_or_invalid_artifacts_fail_closed(monkeypatch,error):
    from bike_share import dashboard
    def fail():raise error
    monkeypatch.setattr(dashboard,'load_saved_analysis',fail)
    page=app()
    assert not page.exception and len(page.error)==1 and not page.metric
    assert '/private/path' not in page.error[0].value


def test_viewer_requires_no_gurobi_package(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'gurobipy',None)
    page=app()
    assert not page.exception and len(page.metric)==4
