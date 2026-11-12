"""Command-line orchestration; modeling functions take explicit inputs and paths."""
from pathlib import Path
import json
import pandas as pd
from .config import PROJECT_ROOT
from .data import ensure_source_data
from .features import prepare
from .forecasting import fit_first, run_advanced, forecast_day
from .evaluation import score
from .artifacts import save_results, verify_artifacts
from .visualization import eda, evaluation_plots, explain_cases, business_plot
from .optimization import run_optimization
from .documents import model_card, generate_pdfs
from .notebooks import execute_notebooks
from .quality import run_self_tests, run_demo_tests

def main(argv=None):
    """Execute the complete, unchanged analysis and regenerate all business outputs."""
    import argparse
    parser = argparse.ArgumentParser(description='Bike Sharing Predictive Analysis: historical forecasting and staffing scenarios')
    parser.add_argument('--output-dir', type=Path, help='Destination for a run, or project directory for an action')
    parser.add_argument('--with-gurobi', action='store_true', help='Include optional solver tests with --self-test')
    parser.add_argument('--report-file', type=Path, help='Save a new test/verification report without changing historical evidence')
    parser.add_argument('--data-file', type=Path, help='Optional path to the original hour.csv')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--self-test', action='store_true', help='Run pytest core checks (no solver license by default)')
    actions.add_argument('--verify', action='store_true', help='Verify saved notebook, report and result artifacts')
    actions.add_argument('--execute-notebooks', action='store_true', help='Execute the integrated analysis notebook')
    actions.add_argument('--build-documents', action='store_true', help='Build PDFs from retained results')
    actions.add_argument('--test-demo', action='store_true', help='Run automated Streamlit regression tests')
    actions.add_argument('--demo', action='store_true', help='Launch the interactive Streamlit demo')
    args = parser.parse_args(argv)
    if args.report_file and not (args.self_test or args.verify or args.test_demo):
        parser.error('--report-file requires --self-test, --test-demo, or --verify')
    if args.with_gurobi and not args.self_test:
        parser.error('--with-gurobi requires --self-test')
    if args.demo:
        import subprocess, sys
        return subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(PROJECT_ROOT/'bike_sharing.py'),
                               '--browser.gatherUsageStats=false'], check=True)
    action_root = (args.output_dir or PROJECT_ROOT).expanduser().resolve()
    if args.self_test:
        return run_self_tests(args.report_file,args.with_gurobi)
    if args.verify:
        return verify_artifacts(args.report_file,action_root)
    if args.test_demo:
        return run_demo_tests(args.report_file)
    if args.execute_notebooks:
        return execute_notebooks(action_root)
    if args.build_documents:
        return generate_pdfs(action_root)
    root = (args.output_dir or PROJECT_ROOT/'run_outputs').expanduser().resolve()
    out = root/'visualizations'
    for folder in [root, out, root/'results', root/'data', root/'reports']:
        folder.mkdir(parents=True, exist_ok=True)
    ensure_source_data(args.data_file,root)
    d = prepare(root/'data/hour.csv')
    print('DATA AUDIT', json.dumps(d['audit'], indent=2), flush=True)
    print(d['raw'].head().to_string())
    print(d['raw'].dtypes.to_string())
    print(d['raw']['cnt'].describe().to_string())
    eda(d,out)
    ridge, ridge_pred = fit_first(d)
    first = pd.DataFrame({'actual': d['y'].loc[d['test']],
        'seasonal_naive': d['X'].loc[d['test'], 'lag_168'], 'ridge': ridge_pred})
    first_metrics = pd.DataFrame({n:score(first.actual,first[n]) for n in ['seasonal_naive','ridge']}).T
    first_metrics.to_csv(root/'results/baseline_metrics.csv')
    evaluation_plots(first,output_dir=out)
    print('Tuning boosting on proper-training time folds only...', flush=True)
    a = run_advanced(d, ridge_pred)
    save_results(d, a, root)
    evaluation_plots(a['predictions'], a['slices'],out)
    cases = explain_cases(a['model'], d, a['predictions'],out)
    (root/'results/shap_cases.json').write_text(json.dumps(cases, indent=2))
    for case in cases:
        print(case['explanation'])
    business_plot(a['costs'],out)
    operations = run_optimization(a['predictions'],root)
    print('GUROBI STAFFING COSTS\n'+operations['totals'].to_string())
    (root/'reports/model_card.md').write_text(model_card(d,a))
    day = d['test'][0].normalize()
    forecast_day(ridge,d['grid'],day,int(d['grid'].loc[day,'holiday'])).to_csv(root/'results/example_24_hour_forecast.csv')
    generate_pdfs(root)
    print('\nVERIFIED TEST METRICS\n'+a['metrics'].to_string())
    print('\nINTERVALS', json.dumps(a['intervals'],indent=2))
    print('\nCOST SENSITIVITY\n'+a['costs'].to_string(index=False))
    print('\nOutputs saved to:', root)
    print('AI assistance: OpenAI Codex helped debug and resolve issues, implement code, and draft documentation. Author review required.')
    print('Analysis outputs saved locally; this script does not deploy a service.')
    return a
