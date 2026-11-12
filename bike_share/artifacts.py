"""Persist and reconcile the audited historical experiment without model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import PROJECT_ROOT, STAFFING
from .data import load_data
from .features import prepare
from .evaluation import score, error_slices, cost_analysis
from .optimization import staffing_cost
from .validation import require
from .quality import write_check_report

def save_results(d, a, root=PROJECT_ROOT):
    root = Path(root)
    out=root/'results';out.mkdir(exist_ok=True)
    for k in ['predictions','metrics','slices','costs']:
        a[k].to_csv(out/f'{k}.csv',index=k in ['predictions','metrics'])
    pd.DataFrame(a['search'].cv_results_).to_csv(out/'cv_results.csv',index=False)
    summary={'audit':d['audit'],'best_params':a['search'].best_params_,
             'best_cv_MAE':-a['search'].best_score_,'cv_boundaries':a['cv_boundaries'],
             'intervals':a['intervals']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def validate_saved_analysis(bundle,data_file=None):
    """Validate historical result relationships against the audited source, without fitting."""
    p, m, plans, daily, totals, summary, ops = (bundle[k] for k in
        ['predictions', 'metrics', 'plans', 'daily', 'totals', 'summary', 'operations'])
    d = prepare(data_file)
    models = ['seasonal_naive', 'ridge', 'boosting']
    methods = ['flat_staffing', *models]
    require(isinstance(p.index, pd.DatetimeIndex) and p.index.equals(d['test']),
            'Prediction timestamps must exactly match the chronological common test sample')
    columns = ['actual', *models, 'lower_90', 'upper_90']
    require(set(columns).issubset(p.columns), 'Missing prediction columns')
    values = p[columns].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all(), 'Invalid prediction values')
    require(np.allclose(p.actual, d['y'].loc[d['test']], rtol=0, atol=0),
            'Saved actuals differ from the audited source')
    require(np.allclose(p.seasonal_naive, d['X'].loc[d['test'], 'lag_168'], rtol=0, atol=0),
            'Saved weekly baseline differs from the audited source')
    require(m.index.is_unique and set(m.index) == set(models), 'Invalid model metric keys')
    for model in models:
        for metric, value in score(p.actual, p[model]).items():
            require(np.isclose(m.loc[model, metric], value, rtol=1e-9), 'Model metrics do not match predictions')
        gain = m.loc['seasonal_naive', 'MAE'] - m.loc[model, 'MAE']
        require(np.isclose(m.loc[model, 'MAE_improvement'], gain), 'Invalid absolute MAE improvement')
        require(np.isclose(m.loc[model, 'MAE_improvement_pct'], 100*gain/m.loc['seasonal_naive','MAE']),
                'Invalid percentage MAE improvement')
    require((p.lower_90 <= p.upper_90).all(), 'Prediction interval bounds are reversed')
    require(summary['audit'] == d['audit'], 'Saved data/split audit differs from the source')
    coverage = ((p.actual >= p.lower_90) & (p.actual <= p.upper_90)).mean()
    require(np.isclose(coverage, summary['intervals']['empirical_coverage']), 'Interval coverage is inconsistent')
    require(np.isclose((p.upper_90-p.lower_90).mean(), summary['intervals']['average_width']),
            'Interval width is inconsistent')
    require(summary['intervals']['nominal_coverage'] == .9 and
            summary['intervals']['calibration_rows'] == len(d['cal']), 'Invalid calibration metadata')
    require(ops['assumptions'] == STAFFING, 'Saved staffing assumptions differ from the recorded experiment')
    dates = [str(date.date()) for date, frame in p.groupby(p.index.normalize())
             if len(frame) == 24 and list(frame.index.hour) == list(range(24))]
    expected_days = {(date, method) for date in dates for method in methods}
    require(not daily.duplicated(['date','method']).any() and
            set(zip(daily.date, daily.method)) == expected_days, 'Missing or duplicate policy-days')
    expected_shifts = {(date, method, hour) for date, method in expected_days for hour in range(0,24,4)}
    require(not plans.duplicated(['date','method','start_hour']).any() and
            set(zip(plans.date,plans.method,plans.start_hour)) == expected_shifts,
            'Missing or duplicate policy-shift assignments')
    require(ops['complete_days'] == len(dates) and ops['evaluated_hours'] == 24*len(dates)
            and ops['optimal_solves'] == 3*len(dates)
            and ops['excluded_partial_days'] == p.index.normalize().nunique()-len(dates),
            'Staffing sample counts are inconsistent')
    for row in daily.itertuples():
        day = p.loc[row.date]
        x = plans[(plans.date == row.date) & (plans.method == row.method)].sort_values('start_hour').assignments.to_numpy()
        actual_cost = staffing_cost(day.actual, x)
        for field, value in actual_cost.items():
            require(np.isclose(getattr(row, field), value, rtol=1e-9), 'Staffing cost components are inconsistent')
        require(row.assignments == x.sum(), 'Daily assignment total is inconsistent')
        if row.method == 'flat_staffing':
            require(np.array_equal(x, np.full(6,8)) and row.status == 'FIXED', 'Invalid fixed staffing baseline')
        else:
            require(row.status == 'OPTIMAL' and row.mip_gap == 0 and row.variables == 54 and row.constraints == 25,
                    'Invalid saved solver certificate')
            require(np.isclose(staffing_cost(day[row.method],x)['total_cost'],row.objective,rtol=1e-9)
                    and np.isclose(row.objective,row.bound,rtol=1e-9), 'Solver objective/bound mismatch')
    expected_totals = daily.groupby('method')[['wages','shortage_units','excess_units','total_cost']].sum()
    require(totals.index.is_unique and set(totals.index) == set(methods), 'Invalid aggregate policy keys')
    require(np.allclose(totals.loc[expected_totals.index,expected_totals.columns], expected_totals, rtol=1e-9),
            'Aggregate staffing costs are inconsistent')
    return bundle


def load_saved_analysis(root=PROJECT_ROOT):
    """Read-only boundary shared by verification and the historical demo."""
    root = Path(root)
    result = root/'results'
    bundle = {
        'predictions': pd.read_csv(result/'predictions.csv', index_col='timestamp', parse_dates=True),
        'metrics': pd.read_csv(result/'metrics.csv', index_col='model'),
        'plans': pd.read_csv(result/'staffing_plans.csv'),
        'daily': pd.read_csv(result/'staffing_daily.csv'),
        'totals': pd.read_csv(result/'staffing_metrics.csv', index_col='method'),
        'summary': json.loads((result/'summary.json').read_text()),
        'operations': json.loads((result/'staffing_summary.json').read_text()),
    }
    return validate_saved_analysis(bundle,root/'data/hour.csv')


def verify_artifacts(report_file=None,root=PROJECT_ROOT):
    """Check results and portfolio artifacts without retraining or selecting models."""
    root = Path(root)
    if not __debug__:
        raise RuntimeError("Run verification without Python -O; assertions must remain enabled")
    validated = load_saved_analysis(root)
    import nbformat
    from pypdf import PdfReader
    report={'notebooks':{},'pdfs':{},'results':{},'limitations':[
        'Google-hosted Colab execution has not been independently verified.',
        'Nominal 90% temporal prediction intervals under-cover.',
        'Personal research prototype; not deployed in production.']}
    for name in ['analysis.ipynb']:
        nb=nbformat.read(root/name,as_version=4);nbformat.validate(nb)
        cells=[c for c in nb.cells if c.cell_type=='code']
        assert cells and all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        images=sum('image/png' in o.get('data',{}) for c in cells for o in c.outputs)
        assert images>=3
        report['notebooks'][name]={'executed_code_cells':len(cells),'embedded_png_outputs':images,'errors':0}
    for name,limit in [('reports/presentation.pdf',10),('reports/memo.pdf',2)]:
        if not (root/name).exists():
            report['pdfs'][name]={'status':'not generated; optional export'}
            continue
        pdf=PdfReader(root/name)
        assert 0<len(pdf.pages)<=limit
        text='\n'.join(page.extract_text() for page in pdf.pages)
        assert 'Anshuman' in text or 'ANSHUMAN' in text
        assert '[NAMES TO FILL IN]' not in text
        report['pdfs'][name]={'pages':len(pdf.pages),'page_limit':limit}
    p=validated['predictions']
    m=validated['metrics']
    assert len(p)==1302 and not p.isna().any().any()
    for model in ['seasonal_naive','ridge','boosting']:
        actual=score(p.actual,p[model])
        assert np.isclose(actual['MAE'],m.loc[model,'MAE'])
        assert np.isclose(actual['RMSE'],m.loc[model,'RMSE'])
    summary=validated['summary']
    coverage=((p.actual>=p.lower_90)&(p.actual<=p.upper_90)).mean()
    assert np.isclose(coverage,summary['intervals']['empirical_coverage'])
    assert np.isclose((p.upper_90-p.lower_90).mean(),summary['intervals']['average_width'])
    report['results']={'common_test_hours':len(p),'metrics_recomputed_from_predictions':True,
                       'coverage':float(coverage),'best_MAE_model':m.MAE.idxmin()}
    cases=json.loads((root/'results/shap_cases.json').read_text())
    assert {c['case'] for c in cases}=={'low','moderate','high'}
    for case in cases:
        assert np.isclose(case['base_value']+sum(f['SHAP'] for f in case['features']),case['raw_prediction'])
        assert (root/f"visualizations/shap_{case['case']}.png").stat().st_size>10000
    slices=pd.read_csv(root/'results/slices.csv')
    assert 'observed_weather' in set(slices.dimension)
    _,grid=load_data(root/'data/hour.csv')
    expected=error_slices(p,grid)
    pd.testing.assert_frame_equal(slices,expected,check_dtype=False,rtol=1e-10,atol=1e-10)
    tests=json.loads((root/'results/safety_tests.json').read_text())
    assert tests=={'tests':17,'failures':0,'errors':0,'skipped':0}
    report['historical_test_evidence']=tests
    ops=validated['operations']
    daily=validated['daily']
    plans=validated['plans']
    assert ops['complete_days']==48 and ops['evaluated_hours']==1152 and ops['optimal_solves']==144
    assert daily.query("method != 'flat_staffing'").status.eq('OPTIMAL').all()
    for row in daily.itertuples():
        day=p.loc[row.date]
        assert len(day)==24
        x=plans[(plans.date==row.date)&(plans.method==row.method)].sort_values('start_hour').assignments.to_numpy()
        assert len(x)==6 and (x>=0).all() and (x<=20).all() and x.sum()<=48
        assert np.isclose(staffing_cost(day.actual,x)['total_cost'],row.total_cost)
        if row.method!='flat_staffing':
            assert np.isclose(staffing_cost(day[row.method],x)['total_cost'],row.objective)
            assert np.isclose(row.objective,row.bound) and row.mip_gap==0
    expected_totals=daily.groupby('method')[['wages','shortage_units','excess_units','total_cost']].sum()
    pd.testing.assert_frame_equal(validated['totals'],expected_totals)
    pd.testing.assert_frame_equal(pd.read_csv(root/'results/costs.csv'),cost_analysis(p))
    expected_sensitivity = []
    for under in [2.,5.,10.]:
        for excess in [.5,1.,2.]:
            for method,row in expected_totals.iterrows():
                expected_sensitivity.append({'under_cost':under,'excess_cost':excess,'method':method,
                    'fixed_plan_cost':row.wages+under*row.shortage_units+excess*row.excess_units})
    pd.testing.assert_frame_equal(pd.read_csv(root/'results/staffing_sensitivity.csv'),pd.DataFrame(expected_sensitivity))
    hourly = pd.read_csv(root/'results/staffing_hourly.csv')
    expected_hourly = []
    for row in daily.itertuples():
        x=plans[(plans.date==row.date)&(plans.method==row.method)].sort_values('start_hour').assignments.to_numpy()
        for hour,(timestamp,record) in enumerate(p.loc[row.date].iterrows()):
            expected_hourly.append({'timestamp':str(timestamp),'method':row.method,
                'actual':float(record.actual),'capacity':float(x[hour//4]*STAFFING['throughput'])})
    pd.testing.assert_frame_equal(hourly,pd.DataFrame(expected_hourly))
    assert (root/'visualizations/staffing_interactive.html').stat().st_size>10000
    report['optimization']={'complete_days':48,'certified_optimal_solves':144,'costs_recomputed':True}

    # Source inventory is explicit; do not walk virtual environments or unrelated caches.
    source_root = PROJECT_ROOT
    sources = [source_root/'bike_sharing.py', *sorted((source_root/'bike_share').glob('*.py')),
               *sorted((source_root/'tests').glob('*.py')), *sorted((source_root/'benchmarks').glob('*.py'))]
    report['python_files'] = [str(path.relative_to(source_root)) for path in sources if path.is_file()]
    write_check_report(report, report_file)
    print(json.dumps({k:v for k,v in report.items() if k!='generated_files'},indent=2))
    return report
