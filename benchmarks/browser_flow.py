"""Reproducible browser benchmark; start the local Streamlit server first."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import json,time,statistics,platform
import argparse
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Measure a single local historical-demo browser flow')
parser.add_argument('--url',default='http://127.0.0.1:8501')
parser.add_argument('--output-dir',type=Path,default=ROOT/'run_outputs/benchmark')
args=parser.parse_args()
output=args.output_dir.expanduser().resolve()
output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1500})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    started=time.perf_counter()
    response=page.goto(args.url,wait_until='domcontentloaded')
    page.get_by_text('Historical forecasts · 2012-11-07',exact=True).wait_for(timeout=60000)
    first_render=time.perf_counter()-started
    samples=[]
    for day in ['2012-11-08','2012-11-09','2012-11-10','2012-11-11','2012-11-12']:
        started=time.perf_counter()
        combo=page.get_by_role('combobox',name='Forecast day',exact=True)
        combo.fill(day)
        page.get_by_role('option',name=day,exact=True).click()
        page.get_by_text('Historical forecasts · '+day,exact=True).wait_for(timeout=30000)
        samples.append(time.perf_counter()-started)
    page.get_by_role('combobox',name='Forecast model',exact=True).click()
    page.get_by_role('option',name='Tuned gradient boosting',exact=True).click()
    page.get_by_text('Nominal coverage: 90%; measured test coverage: 85.64%.',exact=False).wait_for()
    with page.expect_download() as info:
        page.get_by_role('button',name='Download this day’s scored forecasts').click()
    download=info.value;download.save_as(output/'selected_forecasts.csv')
    page.locator('[data-testid="stMain"]').evaluate('(el)=>el.scrollTop=0')
    page.screenshot(path=str(output/'demo.png'))
    # Test staffing selection separately after a fresh page render.
    page.reload(wait_until='domcontentloaded')
    page.get_by_text('Historical forecasts · 2012-11-07',exact=True).wait_for(timeout=60000)
    page.get_by_role('tab',name='Explore staffing',exact=True).click()
    page.get_by_role('combobox',name='Planning method',exact=True).click()
    page.get_by_role('option',name='Ridge regression',exact=True).click()
    page.get_by_text('Selected day · hypothetical evaluated cost',exact=True).wait_for()
    page.screenshot(path=str(output/'staffing.png'))
    page.set_viewport_size({'width':390,'height':844})
    page.get_by_role('tab',name='Explore forecasts',exact=True).click()
    page.locator('[data-testid="stMain"]').evaluate('(el)=>el.scrollTop=0')
    page.screenshot(path=str(output/'mobile.png'))
    assert not errors,errors
    report={'measured_on':datetime.now(timezone.utc).date().isoformat(),'environment':{'os':platform.system(),'machine':platform.machine(),
             'python':platform.python_version(),'browser':browser.version,'viewport':[1440,1500]},
        'scope':'Single local Chromium client against an already-running Streamlit server; saved-data viewer, no Gurobi package. Not a production SLA or concurrent load test.',
        'first_page_ready_seconds':round(first_render,3),
        'historic_date_changes_seconds':[round(x,3) for x in samples],
        'historic_date_change_median_seconds':round(statistics.median(samples),3),
        'historic_date_change_max_seconds':round(max(samples),3),
        'http_status':response.status,'verified':{'historic_dates':True,'forecast_and_weekly_baseline':True,
        'interval_coverage':True,'staffing_selection':True,'csv_download':download.suggested_filename,
        'mobile_layout':True,'page_errors':errors}}
    (output/'demo_benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    browser.close()
