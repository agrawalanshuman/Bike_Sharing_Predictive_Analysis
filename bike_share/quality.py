"""Pytest command adapters and optional reports; no writes to historical evidence."""
from pathlib import Path
import json
from .config import PROJECT_ROOT
from .validation import require

def write_check_report(report, report_file):
    if report_file is not None:
        path = Path(report_file).expanduser().resolve()
        repository = PROJECT_ROOT
        require(path != repository and not any(path.is_relative_to(repository/folder)
                for folder in ['results','data','visualizations']),
                'Write new check reports outside the saved data/results/visualizations folders')
        require(not path.exists(), 'Check report path already exists; use a new path')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2))


def run_pytest(paths, report_file=None, with_gurobi=False):
    if not __debug__:
        raise RuntimeError('Run tests without Python -O; assertions must remain enabled')
    import pytest
    class Recorder:
        def __init__(self):
            self.passed = self.failed = self.errors = self.skipped = 0
        def pytest_runtest_logreport(self, report):
            if report.skipped:
                self.skipped += 1
            elif report.when == 'call':
                if report.passed: self.passed += 1
                else: self.failed += 1
            elif report.failed:
                self.errors += 1
    recorder = Recorder()
    args = ['-q', *[str(PROJECT_ROOT/path) for path in paths]]
    if not with_gurobi: args += ['-m', 'not gurobi']
    status = pytest.main(args, plugins=[recorder])
    report = {'tests':recorder.passed+recorder.failed+recorder.errors+recorder.skipped,
              'passed':recorder.passed,'failures':recorder.failed,'errors':recorder.errors,
              'skipped':recorder.skipped,'exit_code':int(status)}
    write_check_report(report, report_file)
    if status: raise SystemExit(int(status))
    return report


def run_self_tests(report_file=None, with_gurobi=False):
    return run_pytest(['tests/test_data_features.py','tests/test_forecasting.py',
                      'tests/test_artifacts.py','tests/test_optimization.py'],report_file,with_gurobi)


def run_demo_tests(report_file=None):
    return run_pytest(['tests/test_dashboard.py'],report_file)
