"""Check saved notebook execution, results consistency, and PDF page limits."""
from pathlib import Path
import sys, json, subprocess, re, xml.etree.ElementTree as ET
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
ROOT=Path(__file__).resolve().parents[1]
report={'notebooks':{},'pdfs':{},'results':{},'tests':{},'limitations':[
    'TeamName and team member placeholders require user edits and review.',
    'Colab execution is documented but not independently tested.',
    'Nominal 90% temporal intervals under-cover; no guaranteed future coverage.',
    'No publication, push, deployment, or Canvas submission has been performed.']}
for name in ['TeamName_milestone2.ipynb','TeamName_milestone3.ipynb','BikeSharing_Final.ipynb']:
    nb=nbformat.read(ROOT/name,as_version=4);nbformat.validate(nb)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert cells and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    images=sum('image/png' in o.get('data',{}) for c in cells for o in c.outputs)
    assert images>=3
    report['notebooks'][name]={'executed_code_cells':len(cells),'embedded_png_outputs':images,'errors':0}
for name,limit in [('TeamName_dryrun.pdf',10),('TeamName_memo.pdf',2)]:
    info=subprocess.check_output(['pdfinfo',str(ROOT/name)],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    assert 0<pages<=limit
    report['pdfs'][name]={'pages':pages,'page_limit':limit}
p=pd.read_csv(ROOT/'results/predictions.csv',index_col=0,parse_dates=True)
m=pd.read_csv(ROOT/'results/metrics.csv',index_col=0)
assert len(p)==1302 and not p.isna().any().any()
for model in ['seasonal_naive','ridge','boosting']:
    assert np.isclose(mean_absolute_error(p.actual,p[model]),m.loc[model,'MAE'])
    assert np.isclose(root_mean_squared_error(p.actual,p[model]),m.loc[model,'RMSE'])
summary=json.loads((ROOT/'results/summary.json').read_text())
coverage=((p.actual>=p.lower_90)&(p.actual<=p.upper_90)).mean()
assert np.isclose(coverage,summary['intervals']['empirical_coverage'])
assert np.isclose((p.upper_90-p.lower_90).mean(),summary['intervals']['average_width'])
report['results']={'common_test_hours':len(p),'metrics_recomputed_from_exported_predictions':True,
                   'coverage_recomputed':float(coverage),'best_MAE_model':m.MAE.idxmin()}
cases=json.loads((ROOT/'results/shap_cases.json').read_text())
assert {c['case'] for c in cases}=={'low','moderate','high'}
for c in cases:
    assert np.isclose(c['base_value']+sum(f['SHAP'] for f in c['features']),c['raw_prediction'])
    assert (ROOT/f"visualizations/shap_{c['case']}.png").stat().st_size>10000
suite=ET.parse(ROOT/'results/test_results.xml').getroot().find('testsuite')
assert int(suite.attrib['tests'])==13 and int(suite.attrib['failures'])==0 and int(suite.attrib['errors'])==0
report['tests']={k:int(suite.attrib.get(k,0)) for k in ['tests','failures','errors','skipped']}
text=(ROOT/'milestone1_canvas.txt').read_text().strip()
assert len(text.split('\n\n'))==1
paths=[]
for path in sorted(ROOT.rglob('*')):
    rel=path.relative_to(ROOT)
    if any(part in {'.venv','.git','.jupyter','.pytest_cache','__pycache__','.ipynb_checkpoints'} for part in rel.parts):continue
    if path.is_file():paths.append({'file':str(rel),'bytes':path.stat().st_size})
report['generated_files']=paths
(ROOT/'results/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='generated_files'},indent=2))
