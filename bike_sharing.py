"""Compatibility entry point for the CLI, Streamlit and existing analysis notebook.

Run `python bike_sharing.py --help` or `python -m streamlit run bike_sharing.py`.
Implementation and tests live in bike_share/ and tests/.
"""
from bike_share.config import FEATURES, NUMERIC, CATEGORICAL, SEED, STAFFING
from bike_share.data import load_data
from bike_share.features import prepare, make_features, issue_times
from bike_share.forecasting import build_pipeline, fit_first, run_advanced, forecast_day
from bike_share.evaluation import score, error_slices, cost_analysis
from bike_share.artifacts import save_results, verify_artifacts
from bike_share.visualization import eda, evaluation_plots, explain_cases, business_plot
from bike_share.optimization import optimize_staffing, staffing_cost, run_optimization
from bike_share.documents import model_card

if __name__ == '__main__':
    import sys
    if 'streamlit' in sys.modules:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_dashboard = get_script_run_ctx(suppress_warning=True) is not None
    else:
        in_dashboard = False
    if in_dashboard:
        from bike_share.dashboard import render_dashboard
        render_dashboard()
    else:
        from bike_share.cli import main
        main()
