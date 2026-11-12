import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from bike_share.config import PROJECT_ROOT as ROOT, FEATURES, NUMERIC, STAFFING
from bike_share.artifacts import load_saved_analysis, validate_saved_analysis, verify_artifacts
from bike_share.quality import write_check_report


def test_prediction_artifacts_reject_tampering():
    import copy, unittest
    pristine = load_saved_analysis()
    for change in ['timestamp','actual','baseline','metric','bounds']:
        invalid = copy.deepcopy(pristine)
        if change=='timestamp':invalid['predictions'].index += pd.Timedelta(hours=1)
        elif change=='actual':invalid['predictions'].iloc[0,0] += 1
        elif change=='baseline':invalid['predictions'].loc[:, 'seasonal_naive'] += 1
        elif change=='metric':invalid['metrics'].loc['ridge','MAE'] += 1
        else:invalid['predictions'].loc[:, 'lower_90'] = invalid['predictions'].upper_90+1
        with unittest.TestCase().assertRaises(ValueError):
            validate_saved_analysis(invalid)


def test_staffing_artifacts_reject_missing_duplicate_and_fractional_plans():
    import copy, unittest
    pristine = load_saved_analysis()
    for change in ['missing_day','duplicate_day','missing_shift','fractional_shift','component','assumptions']:
        invalid = copy.deepcopy(pristine)
        if change=='missing_day':invalid['daily'] = invalid['daily'].iloc[1:]
        elif change=='duplicate_day':invalid['daily'] = pd.concat([invalid['daily'],invalid['daily'].iloc[:1]])
        elif change=='missing_shift':invalid['plans'] = invalid['plans'].iloc[1:]
        elif change=='fractional_shift':
            invalid['plans']['assignments'] = invalid['plans'].assignments.astype(float)
            invalid['plans'].loc[0,'assignments'] += .5
        elif change=='component':invalid['daily'].loc[0,'wages'] += 1
        else:invalid['operations']['assumptions']['throughput'] += 1
        with unittest.TestCase().assertRaises(ValueError):
            validate_saved_analysis(invalid)


def test_check_reports_preserve_historical_evidence():
    import tempfile, unittest
    before = (ROOT/'results/safety_tests.json').read_bytes()
    with unittest.TestCase().assertRaises(ValueError):
        write_check_report({'tests':0},ROOT/'results/safety_tests.json')
    assert before == (ROOT/'results/safety_tests.json').read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'new-report.json'
        write_check_report({'tests':1},path)
        assert json.loads(path.read_text()) == {'tests':1}
        with unittest.TestCase().assertRaises(ValueError):
            write_check_report({'tests':2},path)


def test_optimized_python_cannot_report_false_success():
    import subprocess, sys
    for action in ['--self-test','--verify']:
        run = subprocess.run([sys.executable,'-O',str(ROOT/'bike_sharing.py'),action],
                             capture_output=True,text=True,timeout=45)
        assert run.returncode != 0 and 'without Python -O' in run.stderr


def test_verification_is_read_only():
    import hashlib
    protected=[*ROOT.glob('results/*'),ROOT/'analysis.ipynb',ROOT/'data/hour.csv']
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    report=verify_artifacts()
    assert report['results']['metrics_recomputed_from_predictions']
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}==before
