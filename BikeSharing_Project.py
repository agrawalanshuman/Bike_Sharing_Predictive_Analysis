"""Anshuman's complete ISOM 835 workflow in one runnable Python file.

Install requirements.txt, then run:
    python BikeSharing_Project.py
Outputs go to run_outputs/, leaving the submitted results untouched.
Use --data-file /path/to/hour.csv to supply a local source file.
See Open_in_Colab.ipynb for a one-click Google Colab launcher.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bike-mpl')


# ==================== FORECASTING ====================
"""Leakage-safe next-calendar-day forecasting. All fitted transforms live in pipelines."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parent
SEED = 835
CATEGORICAL = ['hour', 'dow', 'month', 'holiday', 'workingday']
NUMERIC = ['lag_24', 'lag_48', 'lag_168', 'roll_mean_7d', 'roll_max_7d',
           'roll_count_7d', 'commute_workday', 'hour_sin', 'hour_cos', 'trend_days']
FEATURES = CATEGORICAL + NUMERIC
EXCLUDED = ['casual', 'registered', 'cnt', 'temp', 'atemp', 'hum', 'windspeed',
            'weathersit', 'instant', 'dteday', 'yr', 'mnth', 'hr', 'weekday', 'season']


def load_data(path=None):
    """Keep missing target hours as NaN, never interpolate rental observations."""
    path = Path(path or ROOT / 'data/hour.csv')
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw['dteday']) + pd.to_timedelta(raw['hr'], unit='h')
    if ts.duplicated().any():
        raise ValueError('Duplicate hourly timestamps')
    if raw['cnt'].isna().any() or not (raw['casual'] + raw['registered']).equals(raw['cnt']):
        raise ValueError('Invalid original target data')
    raw.index = pd.DatetimeIndex(ts, name='timestamp')
    return raw.sort_index(), raw.sort_index().asfreq('h')


def issue_times(index):
    """One common issuance for every hour of a target calendar day."""
    return index.normalize() - pd.Timedelta(minutes=1)


def make_features(grid):
    idx, y = grid.index, grid['cnt']
    # Holiday metadata are treated as a published calendar, not estimated from outcomes.
    holiday_by_date = grid.groupby(idx.normalize())['holiday'].first()
    holiday = holiday_by_date.reindex(idx.normalize()).to_numpy()
    work = ((idx.dayofweek < 5) & (holiday == 0)).astype(int)
    X = pd.DataFrame({'hour': idx.hour, 'dow': idx.dayofweek, 'month': idx.month,
                      'holiday': holiday, 'workingday': work}, index=idx)
    for lag in [24, 48, 168]:
        X[f'lag_{lag}'] = y.shift(lag)
    # The window is [target-191h, target-24h], inclusive: all 168 hourly slots.
    history = y.shift(24).rolling(168, min_periods=1)
    X['roll_mean_7d'] = history.mean()
    X['roll_max_7d'] = history.max()
    X['roll_count_7d'] = history.count()
    X['commute_workday'] = (np.isin(idx.hour, [7, 8, 9, 16, 17, 18]) & (work == 1)).astype(int)
    X['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
    X['trend_days'] = (idx - pd.Timestamp('2011-01-01')) / pd.Timedelta(days=1)
    return X[FEATURES]


def prepare():
    raw, grid = load_data()
    X = make_features(grid)
    # Full Monday-Sunday weeks; Dec 31, 2012 is an incomplete terminal week.
    exclusive_end = (grid.index.max() + pd.Timedelta(hours=1)).normalize()
    test_end = exclusive_end - pd.Timedelta(days=exclusive_end.dayofweek)
    test_start = test_end - pd.Timedelta(weeks=8)
    cal_start = test_start - pd.Timedelta(weeks=4)
    enough_history = X.index >= grid.index.min() + pd.Timedelta(hours=191)
    # All fitted models can impute missing lags; baseline cannot invent a lag_168.
    observed = grid['cnt'].notna() & enough_history
    fit_idx = X.index[observed & (X.index < cal_start)]
    cal_idx = X.index[observed & (X.index >= cal_start) & (X.index < test_start)]
    scheduled_test = (X.index >= test_start) & (X.index < test_end)
    test_idx = X.index[observed & scheduled_test & X['lag_168'].notna()]
    audit = {'source_rows': len(raw), 'source_columns': len(raw.columns),
             'source_start': str(raw.index.min()), 'source_end': str(raw.index.max()),
             'full_grid_hours': len(grid), 'missing_hours': int(grid['cnt'].isna().sum()),
             'sha256_hour_csv': hashlib.sha256((ROOT/'data/hour.csv').read_bytes()).hexdigest(),
             'fit_start': str(fit_idx.min()), 'fit_end': str(fit_idx.max()), 'fit_rows': len(fit_idx),
             'calibration_start': str(cal_start), 'calibration_end': str(test_start-pd.Timedelta(hours=1)),
             'calibration_rows': len(cal_idx), 'test_start': str(test_start),
             'test_end': str(test_end-pd.Timedelta(hours=1)), 'test_rows': len(test_idx),
             'test_scheduled_hours': int(scheduled_test.sum()),
             'test_missing_actuals': int((scheduled_test & grid['cnt'].isna()).sum()),
             'test_missing_week_lag_with_actual': int((scheduled_test & grid['cnt'].notna() & X['lag_168'].isna()).sum()),
             'terminal_hours_not_tested': int((X.index >= test_end).sum())}
    return {'raw': raw, 'grid': grid, 'X': X, 'y': grid['cnt'], 'fit': fit_idx,
            'cal': cal_idx, 'test': test_idx, 'audit': audit}


def build_pipeline(kind):
    steps = [('impute', SimpleImputer(strategy='median', add_indicator=True))]
    if kind == 'ridge':
        steps.append(('scale', StandardScaler()))
    prep = ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC),
        ('calendar', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)
    ], sparse_threshold=0)
    model = Ridge(alpha=10) if kind == 'ridge' else HistGradientBoostingRegressor(
        max_iter=220, early_stopping=False, random_state=SEED)
    return Pipeline([('prep', prep), ('model', model)])


def score(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)), 'RMSE': float(root_mean_squared_error(y, pred))}


def fit_first(d):
    model = build_pipeline('ridge').fit(d['X'].loc[d['fit']], d['y'].loc[d['fit']])
    # Nonnegative clipping is a prespecified physical constraint for rental counts.
    pred = np.maximum(0, model.predict(d['X'].loc[d['test']]))
    return model, pred


def tune_boosting(d):
    X, y = d['X'].loc[d['fit']], d['y'].loc[d['fit']]
    cv = TimeSeriesSplit(n_splits=3, test_size=24*28, gap=24)
    boundaries = []
    for tr, va in cv.split(X):
        assert X.index[tr[-1]] < issue_times(X.index[va[:1]])[0]
        boundaries.append({'train_end': str(X.index[tr[-1]]), 'validation_start': str(X.index[va[0]]),
                           'validation_end': str(X.index[va[-1]])})
    search = RandomizedSearchCV(build_pipeline('boost'),
        {'model__learning_rate': [0.04, 0.08, 0.12],
         'model__max_leaf_nodes': [15, 31, 47],
         'model__l2_regularization': [0., 1., 10.]},
        n_iter=8, cv=cv, scoring='neg_mean_absolute_error', random_state=SEED,
        n_jobs=1, refit=True, error_score='raise')
    search.fit(X, y)
    return search, boundaries


def compute_intervals(model, d):
    from mapie.regression import SplitConformalRegressor
    # No refitting on calibration or test targets.
    conformal = SplitConformalRegressor(estimator=model, confidence_level=0.9, prefit=True)
    conformal.conformalize(d['X'].loc[d['cal']], d['y'].loc[d['cal']])
    point, bounds = conformal.predict_interval(d['X'].loc[d['test']])
    lower, upper = np.maximum(0, bounds[:, 0, 0]), np.maximum(0, bounds[:, 1, 0])
    truth = d['y'].loc[d['test']].to_numpy()
    return point, lower, upper, {'nominal_coverage': .9,
        'empirical_coverage': float(np.mean((truth >= lower) & (truth <= upper))),
        'average_width': float(np.mean(upper-lower)), 'calibration_rows': len(d['cal'])}


def error_slices(predictions):
    rows = []
    idx = predictions.index
    groups = {'hour': pd.Series(idx.hour.astype(str), index=idx),
              'day_type': pd.Series(np.where(idx.dayofweek < 5, 'weekday', 'weekend'), index=idx),
              'season': pd.Series(np.select([idx.month.isin([12,1,2]), idx.month.isin([3,4,5]),
                    idx.month.isin([6,7,8])], ['winter','spring','summer'], default='autumn'),index=idx),
              'period': pd.Series(np.where(np.isin(idx.hour,[7,8,9,16,17,18]),'commute','other'),index=idx)}
    for dimension, labels in groups.items():
        for value, ix in labels.groupby(labels).groups.items():
            for name in ['seasonal_naive','ridge','boosting']:
                rows.append({'dimension':dimension,'slice':value,'model':name,'n':len(ix),
                             **score(predictions.loc[ix,'actual'],predictions.loc[ix,name])})
    return pd.DataFrame(rows)


def cost_analysis(predictions):
    """Capacity = ceiling(max(0, point forecast)), fixed before viewing costs/test."""
    rows=[]
    for under in [2., 5., 10.]:
        for excess in [.5, 1., 2.]:
            totals={}
            for model in ['seasonal_naive','boosting']:
                plan=np.ceil(np.maximum(0, predictions[model]))
                shortage=np.maximum(predictions.actual-plan,0).sum()
                surplus=np.maximum(plan-predictions.actual,0).sum()
                totals[model] = float(under*shortage + excess*surplus)
            rows.append({'under_cost':under,'excess_cost':excess,**totals,
                         'proxy_cost_reduction': totals['seasonal_naive']-totals['boosting']})
    return pd.DataFrame(rows)


def run_advanced(d, ridge_pred):
    search, boundaries = tune_boosting(d)
    model = search.best_estimator_
    raw_pred, lower, upper, intervals = compute_intervals(model,d)
    predictions=pd.DataFrame({'actual':d['y'].loc[d['test']],
        'seasonal_naive':d['X'].loc[d['test'],'lag_168'], 'ridge':ridge_pred,
        'boosting':np.maximum(0,raw_pred), 'lower_90':lower,'upper_90':upper})
    metrics=pd.DataFrame({n:score(predictions.actual,predictions[n])
                         for n in ['seasonal_naive','ridge','boosting']}).T
    metrics.index.name='model'
    metrics['MAE_improvement']=metrics.loc['seasonal_naive','MAE']-metrics.MAE
    metrics['MAE_improvement_pct']=100*metrics.MAE_improvement/metrics.loc['seasonal_naive','MAE']
    return {'model':model,'search':search,'cv_boundaries':boundaries,'predictions':predictions,
            'metrics':metrics,'intervals':intervals,'slices':error_slices(predictions),
            'costs':cost_analysis(predictions)}


def save_results(d, a):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    for k in ['predictions','metrics','slices','costs']:
        a[k].to_csv(out/f'{k}.csv',index=k in ['predictions','metrics'])
    pd.DataFrame(a['search'].cv_results_).to_csv(out/'cv_results.csv',index=False)
    summary={'audit':d['audit'],'best_params':a['search'].best_params_,
             'best_cv_MAE':-a['search'].best_score_,'cv_boundaries':a['cv_boundaries'],
             'intervals':a['intervals']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def forecast_day(model, history_grid, target_date, holiday):
    """Issue 24 forecasts using only history before the requested calendar day.

    `holiday` must come from an available calendar, not inferred from future demand.
    The caller must supply a model fitted before issuance. Partial/missing historical
    hours are retained and handled by the fitted training-only imputer.
    """
    day = pd.Timestamp(target_date).normalize()
    past = history_grid.loc[history_grid.index < day].copy()
    if past.empty:
        raise ValueError('Historical observations are required')
    slots = pd.date_range(day, periods=24, freq='h', name='timestamp')
    grid = past.reindex(pd.date_range(past.index.min(), slots[-1], freq='h', name='timestamp'))
    grid.loc[slots, 'holiday'] = int(holiday)
    X = make_features(grid).loc[slots]
    return pd.DataFrame({'forecast': np.maximum(0, model.predict(X)),
                         'seasonal_naive': X['lag_168']}, index=slots)


# ==================== PLOTS ====================
"""Figures and faithful local explanations from fitted model outputs."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':110,'savefig.dpi':150,'axes.spines.top':False,
                     'axes.spines.right':False,'font.size':10})
OUT=ROOT/'visualizations'


def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/f'{name}.png',bbox_inches='tight');plt.close(fig)
    return OUT/f'{name}.png'


def eda(d):
    # Descriptive EDA is restricted to the proper training period.
    raw=d['raw'].loc[:d['fit'].max()]
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    raw.groupby(raw.index.hour)['cnt'].mean().plot(ax=axs[0,0],color='#127d92')
    axs[0,0].set(title='Hourly seasonality (training data)',xlabel='Hour',ylabel='Mean rentals')
    raw.groupby(raw.index.dayofweek)['cnt'].mean().plot.bar(ax=axs[0,1],color='#127d92')
    axs[0,1].set(title='Weekly seasonality',xlabel='Weekday: Monday=0',ylabel='Mean rentals')
    raw['cnt'].resample('MS').mean().plot(ax=axs[1,0],color='#127d92')
    axs[1,0].set(title='Monthly mean hourly rentals',xlabel='Month',ylabel='Mean rentals')
    raw['cnt'].hist(bins=45,ax=axs[1,1],color='#127d92')
    axs[1,1].set(title='Training target distribution',xlabel='Rentals per hour',ylabel='Observed hours')
    return finish(fig,'eda')


def evaluation_plots(p,slices=None):
    week=p.loc[p.index.min():p.index.min()+pd.Timedelta(days=7)-pd.Timedelta(hours=1)]
    fig,ax=plt.subplots(figsize=(12,4))
    for n,color in [('actual','#182e45'),('seasonal_naive','#d68b25'),('ridge','#8c6bb1'),('boosting','#127d92')]:
        if n in week:ax.plot(week.index,week[n],label=n,color=color,lw=1.2)
    ax.set(title='First held-out week: same observed hours',ylabel='Rentals per hour',xlabel='Target time')
    ax.legend(ncol=4);finish(fig,'actual_vs_predicted' if 'boosting' in p else 'milestone2_week')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    name='boosting' if 'boosting' in p else 'ridge'
    res=p.actual-p[name]
    axs[0].scatter(p[name],res,alpha=.35,s=9);axs[0].axhline(0,color='black',lw=1)
    axs[0].set(xlabel='Forecast rentals',ylabel='Actual - forecast',title=f'{name}: residual pattern')
    axs[1].hist(res,bins=45,color='#127d92');axs[1].set(xlabel='Residual rentals',ylabel='Hours',title='Held-out residual distribution')
    finish(fig,'residuals' if name=='boosting' else 'milestone2_residuals')
    if 'lower_90' in p:
        fig,ax=plt.subplots(figsize=(12,4))
        ax.fill_between(week.index,week.lower_90,week.upper_90,alpha=.25,color='#127d92',label='90% nominal interval')
        ax.plot(week.index,week.actual,color='#182e45',label='Actual');ax.plot(week.index,week.boosting,color='#127d92',label='Boosting')
        ax.set(title='Calibration-based uncertainty on held-out hours',ylabel='Rentals per hour');ax.legend()
        finish(fig,'intervals')
    if slices is not None:
        h=slices[slices.dimension=='hour'].copy();h['hour']=h['slice'].astype(int)
        fig,ax=plt.subplots(figsize=(10,4))
        h.pivot(index='hour',columns='model',values='MAE').sort_index().plot(ax=ax)
        ax.set(title='Held-out MAE by hour',ylabel='MAE (rentals/hour)',xlabel='Hour');finish(fig,'sliced_errors')


def explain_cases(model,d,predictions):
    import shap
    X=d['X'].loc[d['test']]
    # Choose cases by forecast only, not outcome or explanation attractiveness.
    order=np.argsort(predictions.boosting.to_numpy())
    positions={'low':int(order[0]),'moderate':int(order[len(order)//2]),'high':int(order[-1])}
    prep=model.named_steps['prep']; tree=model.named_steps['model']
    transformed=prep.transform(X)
    names=prep.get_feature_names_out()
    explainer=shap.TreeExplainer(tree)
    values=explainer(transformed[list(positions.values())])
    records=[]
    for j,(label,pos) in enumerate(positions.items()):
        explanation=shap.Explanation(values=values.values[j],base_values=float(np.asarray(values.base_values[j]).reshape(-1)[0]),
            data=transformed[pos],feature_names=names)
        raw=float(tree.predict(transformed[pos:pos+1])[0])
        assert np.isclose(explanation.base_values+explanation.values.sum(),raw,atol=1e-5)
        shap.plots.waterfall(explanation,max_display=9,show=False)
        fig=plt.gcf();fig.set_size_inches(10,6);finish(fig,f'shap_{label}')
        top=np.argsort(np.abs(explanation.values))[-2:][::-1]
        terms=[f"{names[k].replace('numeric__','').replace('calendar__','')} ({explanation.values[k]:+.1f} rentals)" for k in top]
        sentence1=f"For the {label}-demand case at {X.index[pos]}, the model forecasts {max(0,raw):.1f} rentals; the largest local contributions are {' and '.join(terms)}."
        sentence2=f"These inputs move this prediction relative to the model reference of {explanation.base_values:.1f} rentals; they describe the model's reasoning, not the causal effect of changing capacity or the calendar."
        records.append({'case':label,'timestamp':str(X.index[pos]),'raw_prediction':raw,
                        'prediction':max(0,raw),'actual':float(predictions.actual.iloc[pos]),
                        'base_value':float(explanation.base_values), 'explanation':sentence1+' '+sentence2,
                        'features':[{'name':str(names[k]),'value':float(transformed[pos,k]),'SHAP':float(explanation.values[k])} for k in range(len(names))]})
    return records


def business_plot(costs):
    base=costs[(costs.under_cost==5)&(costs.excess_cost==1)].iloc[0]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].bar(['Same hour last week','Tuned boosting'],[base.seasonal_naive,base.boosting],color=['#d68b25','#127d92'])
    axs[0].set(title='Assumed costs: USD 5 shortage / USD 1 excess',ylabel='Total hypothetical cost ($)')
    table=costs.pivot(index='under_cost',columns='excess_cost',values='proxy_cost_reduction')
    im=axs[1].imshow(table.to_numpy(),cmap='YlGnBu',aspect='auto')
    axs[1].set_xticks(range(len(table.columns)),[f'${v:g}' for v in table.columns]);axs[1].set_yticks(range(len(table.index)),[f'${v:g}' for v in table.index])
    axs[1].set(title='Proxy cost reduction: baseline minus boosting',xlabel='Assumed excess-unit cost',ylabel='Assumed shortage-unit cost')
    for i in range(len(table)):
        for j in range(len(table.columns)):axs[1].text(j,i,f'${table.iloc[i,j]:,.0f}',ha='center',va='center',fontsize=9)
    finish(fig,'business_costs')


# ==================== DOCUMENTS ====================
"""Business-facing documents use persisted, executed results only."""
from pathlib import Path
import json
import pandas as pd


def model_card(d,a):
    m=a['metrics']; it=a['intervals']; s=a['slices'].query("model == 'boosting'")
    w=s.loc[s.MAE.idxmax()]
    return f'''# Model card — next-day hourly system-wide rentals
**Owner:** Anshuman, [NAMES TO FILL IN]; hypothetical operations manager is the decision owner.
**Intended use:** research prototype for aggregate hourly service-capacity and staffing planning at 11:59 PM for the next 24 hourly slots. Not station allocation or autonomous deployment.
**Data and target:** UCI Capital Bikeshare 2011–2012, {d['audit']['source_rows']:,} observed hours; target `cnt`, completed rentals. Fit {d['audit']['fit_start']}–{d['audit']['fit_end']} ({len(d['fit']):,} rows); calibration {d['audit']['calibration_start']}–{d['audit']['calibration_end']}; test {d['audit']['test_start']}–{d['audit']['test_end']} ({len(d['test']):,} comparable hours).
**Methods:** seasonal naive; fixed-alpha Ridge pipeline; HistGradientBoosting pipeline tuned on three training-only chronological folds. Calendar and pre-issuance demand features only; no target components or future observed weather. Daily lags update, model parameters stay frozen.
**Metrics:** baseline MAE/RMSE {m.loc['seasonal_naive','MAE']:.2f}/{m.loc['seasonal_naive','RMSE']:.2f}; Ridge {m.loc['ridge','MAE']:.2f}/{m.loc['ridge','RMSE']:.2f}; boosting {m.loc['boosting','MAE']:.2f}/{m.loc['boosting','RMSE']:.2f} rentals/hour. Ridge has the lowest held-out MAE; complexity did not win.
**Error slices:** worst boosting slice: {w['dimension']}={w['slice']}, MAE {w.MAE:.2f}, RMSE {w.RMSE:.2f}, n={int(w['n'])}. Test seasons are autumn/winter only; group errors are not individual fairness guarantees.
**Uncertainty:** MAPIE chronological calibration, nominal 90%, empirical {it['empirical_coverage']:.2%}, mean width {it['average_width']:.2f} rentals. Serial dependence and temporal shift undermine exchangeability guarantees; no conditional or station coverage claim.
**Known failure modes:** weather changes and exceptional events, trend shifts, commute peaks, missing history, recording delay, rental counts censoring unmet demand; no station inventories. SHAP is descriptive, not causal. Missing baseline hours narrow the scored population.
**Deployment limits and monitoring:** do not deploy directly. Pilot against Ridge and seasonal naive; verify calendar and count latency. Monitor daily missingness, rolling weekly MAE/RMSE, commute errors, interval coverage/width, and proxy costs. Investigate sustained coverage below 90% or MAE above the concurrent baseline across two weeks; review drift and recalibrate/retrain chronologically before reconsidering deployment. Thresholds are proposed monitoring rules, not validated alarms.
'''


def generate_pdfs():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    font_root = Path(matplotlib.get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('BodyFont', str(font_root/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('BodyBold', str(font_root/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('BodyFont', normal='BodyFont', bold='BodyBold', italic='BodyFont', boldItalic='BodyBold')
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    summary=json.loads((ROOT/'results/summary.json').read_text())
    audit=summary['audit']; interval=summary['intervals']
    m=pd.read_csv(ROOT/'results/metrics.csv',index_col='model')
    costs=pd.read_csv(ROOT/'results/costs.csv');c=costs.query('under_cost == 5 and excess_cost == 1').iloc[0]
    slices=pd.read_csv(ROOT/'results/slices.csv');sl=slices.query("model == 'boosting'");w=sl.loc[sl.MAE.idxmax()]
    cases=json.loads((ROOT/'results/shap_cases.json').read_text())
    navy=HexColor('#182e45');teal=HexColor('#127d92');gray=HexColor('#536474')
    W,H=960,540
    cv=canvas.Canvas(str(ROOT/'Anshuman_dryrun.pdf'),pagesize=(W,H))
    cv.setTitle('Next-Day Hourly Bike-Sharing Demand — Anshuman')
    def wrap(text,size=19,width=860,color=navy,x=50,y=400):
        import textwrap
        cv.setFillColor(color);cv.setFont('BodyFont',size)
        # Use measured word widths rather than a character-count approximation.
        lines=[];line=''
        for word in text.split():
            trial=(line+' '+word).strip()
            if cv.stringWidth(trial,'BodyFont',size)>width and line:
                lines.append(line);line=word
            else:line=trial
        if line:lines.append(line)
        for line in lines:cv.drawString(x,y,line);y-=size*1.45
        return y
    def begin(number,title,kicker):
        cv.setFillColor(white);cv.rect(0,0,W,H,fill=1,stroke=0)
        cv.setFillColor(teal);cv.rect(0,H-12,W,12,fill=1,stroke=0)
        cv.setFont('BodyBold',12);cv.drawString(50,H-44,'ISOM 835  /  ANSHUMAN')
        cv.setFillColor(navy);cv.setFont('BodyBold',27);cv.drawString(50,H-91,title)
        wrap(kicker,14,860,gray,y=H-122)
        cv.setFillColor(gray);cv.setFont('BodyFont',10)
        cv.drawString(50,22,'[NAMES TO FILL IN]  |  Historical classroom prototype; no deployment or verified savings')
        cv.drawRightString(W-40,22,f'{number:02d} / 10')
    def bullets(items,y=355,size=21,width=850):
        for item in items:y=wrap('- '+item,size,width,y=y)-20
    def image(name,x=50,y=65,width=860,height=300):
        cv.drawImage(ImageReader(str(ROOT/f'visualizations/{name}.png')),x,y,width=width,height=height,preserveAspectRatio=True,anchor='c')
    begin(1,'Can we improve tomorrow\'s hourly capacity plan?','Recommendation: pilot Ridge; retain seasonal naive as the fallback.')
    bullets([f'Ridge MAE: {m.loc["ridge","MAE"]:.1f} rentals/hour versus {m.loc["seasonal_naive","MAE"]:.1f} for last week.',
        f'Tuned boosting improves the baseline by {m.loc["boosting","MAE_improvement_pct"]:.1f}%, but trails Ridge.',
        'Use forecasts to inform aggregate planning; require a prospective pilot before operations use.'])
    cv.showPage()
    begin(2,'One issuance. Twenty-four hourly forecasts.','Decision owner: a hypothetical bike-sharing operations manager.')
    bullets(['At 11:59 PM, predict the next calendar day\'s 24 system-wide rental counts.',
        'Current hypothetical process: repeat observed rentals from the same hour last week.',
        'Improved process: review hourly rental-throughput plans and staffing needs with uncertainty.'])
    cv.showPage()
    begin(3,'Real trips, with real limitations','Source: UCI Bike Sharing, Capital Bikeshare, Washington DC, 2011-2012.')
    bullets([f'{audit["source_rows"]:,} observed hours; {audit["missing_hours"]} missing slots stay missing.',
        'Counts measure completed rentals, not unserved customers or station inventories.',
        'Weather observations are excluded; no archived forecasts were supplied.'])
    cv.showPage()
    begin(4,'The forecast cannot see tomorrow','Availability checks and a chronological holdout protect the evaluation.')
    bullets(['Calendar plus lags 24 / 48 / 168 hours; rolling windows end 24 hours before each target.',
        'Fit through Oct 7, 2012; calibrate Oct 8-Nov 4; test Nov 5-Dec 30.',
        f'{audit["test_rows"]:,} common scored hours; missing targets or baseline references excluded.',
        'Tune only on earlier time folds. Past test days refresh lags after observation.'],y=365,size=18)
    cv.showPage()
    begin(5,'The simpler regression wins on MAE','All three methods use the same held-out observations; lower is better.')
    y=345
    for name,label in [('seasonal_naive','Same hour last week'),('ridge','Ridge regression'),('boosting','Tuned boosting')]:
        y=wrap(f'{label}: MAE {m.loc[name,"MAE"]:.2f} | RMSE {m.loc[name,"RMSE"]:.2f}',23,y=y)-30
    wrap('Rental counts per hour. Model complexity did not produce the strongest result.',16,850,gray,y=125)
    cv.showPage()
    begin(6,'Where forecasts follow and miss the hourly pattern','First held-out week, selected in advance; gaps are not invented observations.')
    image('actual_vs_predicted',height=310)
    cv.showPage()
    begin(7,'A transparent capacity-cost scenario','Fixed plan: ceiling of the nonnegative forecast. Assumed $5 shortage / $1 excess per unit.')
    image('business_costs',y=115,height=255)
    wrap(f'Baseline ${c.seasonal_naive:,.0f}; boosting ${c.boosting:,.0f}; proxy reduction ${c.proxy_cost_reduction:,.0f}.',16,860,y=96)
    wrap('Sensitivity varies both assumed costs. No causal savings or optimal staffing claim.',14,860,gray,y=64)
    cv.showPage()
    begin(8,'Explanations help; intervals still miss their target','Tree SHAP attributes one prediction; it does not establish causality.')
    image('shap_high',x=40,y=75,width=535,height=300)
    wrap(f'Nominal coverage: 90%. Observed coverage: {interval["empirical_coverage"]:.1%}. Mean width: {interval["average_width"]:.1f} rentals.',18,310,x=610,y=335)
    wrap('Temporal shift and dependence weaken exchangeability assumptions. Coverage requires monitoring.',17,310,gray,x=610,y=185)
    cv.showPage()
    begin(9,'Commute hours deserve closer review','Sliced errors identify stress points; only autumn and winter are represented in the test.')
    image('sliced_errors',y=105,height=265)
    wrap(f'Worst boosting slice: {w["dimension"]}={w["slice"]}; MAE {w.MAE:.1f}, RMSE {w.RMSE:.1f}; n={int(w["n"])}.',16,y=75)
    cv.showPage()
    begin(10,'Pilot, monitor, then decide','Ridge is the current lead candidate; boosting is a documented challenger.')
    bullets(['Validate data latency, calendars, service constraints, and cost assumptions with operations.',
        'Monitor weekly error, missing history, peak-hour misses, interval coverage, and drift.',
        'Run a new prospective comparison before deployment; obtain station-level data for allocation.',
        'Team must review the AI-assisted work and fill in names before submission.'],y=365,size=18)
    cv.showPage();cv.save()
    # Two-page memo, deliberately separated to keep five sections readable.
    styles=getSampleStyleSheet()
    styles['Title'].fontName='BodyBold'
    styles.add(ParagraphStyle(name='MemoText',fontName='BodyFont',fontSize=10.5,leading=15,spaceAfter=10,textColor=navy))
    styles.add(ParagraphStyle(name='MemoHeading',fontName='BodyBold',fontSize=13,leading=17,spaceBefore=10,spaceAfter=7,textColor=teal))
    story=[]
    def para(text,style='MemoText'):story.append(Paragraph(text,styles[style]))
    para('Next-day system-wide bike-sharing capacity planning','Title')
    para('To: Hypothetical Bike-Sharing Operations Manager<br/>From: Anshuman — [NAMES TO FILL IN]<br/>Subject: Historical forecasting evidence and a proposed pilot')
    para('Question','MemoHeading')
    para('Can next-day hourly forecasts improve aggregate rental-service capacity planning over repeating last week? At 11:59 PM, the proposed workflow predicts each of the next calendar day\'s 24 hourly rental counts. The decision is a system-wide hourly throughput plan to inform staffing review. This dataset cannot determine how many bicycles to place at a particular station.')
    para('Data and Model','MemoHeading')
    para(f'The public UCI Bike Sharing dataset contains {audit["source_rows"]:,} observed hourly counts from Capital Bikeshare in 2011-2012. Its {audit["missing_hours"]} missing hourly records remain missing; no rental observations were invented. Inputs are calendar information and demand lags known before issuance. We exclude casual and registered rentals because they compose the target, and exclude future observed weather because no weather-forecast archive exists. A Ridge regression pipeline and a tuned gradient-boosting pipeline are compared with the same-hour-last-week baseline. Training-only preprocessing and time-ordered cross-validation isolate all fitting from testing.')
    para('Honest Evaluation','MemoHeading')
    para(f'Model fitting ends October 7, 2012; October 8-November 4 is reserved for interval calibration. The final eight complete weeks, November 5-December 30, contain {audit["test_rows"]:,} comparable scored hours. Four missing actuals and 38 missing baseline references are excluded. Baseline MAE is {m.loc["seasonal_naive","MAE"]:.2f} rentals per hour, versus {m.loc["ridge","MAE"]:.2f} for Ridge and {m.loc["boosting","MAE"]:.2f} for tuned boosting. RMSE is {m.loc["seasonal_naive","RMSE"]:.2f}, {m.loc["ridge","RMSE"]:.2f}, and {m.loc["boosting","RMSE"]:.2f}, respectively. Ridge has the lowest error; the more complex model does not win. These are daily rolling-origin feature updates with frozen fitted models, not one forecast made eight weeks in advance. Missing-history exclusions and the autumn/winter test window limit generalization.')
    story.append(PageBreak())
    para('Decision and Money','MemoHeading')
    para(f'For a transparent classroom scenario, plan hourly capacity as the rounded-up nonnegative forecast. Assume $5 per rental-unit of underplanned capacity and $1 per excess unit. On identical test hours the seasonal baseline incurs ${c.seasonal_naive:,.0f} of proxy cost, versus ${c.boosting:,.0f} for boosting, a ${c.proxy_cost_reduction:,.0f} difference. The sensitivity grid varies shortage costs from $2 to $10 and excess costs from $0.50 to $2, without changing the policy using test results. These are assumed historical proxy costs, not actual operating expenses, causal savings, or an optimized staffing solution. Completed rentals miss people who could not rent a bike; they are an imperfect measure of true demand. Favor Ridge for the next pilot based on accuracy and keep the baseline as a fallback; validate costs and capacity constraints before selecting an operational policy.')
    para('Trust Story','MemoHeading')
    high=next(x for x in cases if x['case']=='high')
    para(high['explanation'])
    para(f'Three genuine SHAP waterfalls explain low, moderate, and high forecasts. Chronologically calibrated MAPIE intervals target 90% coverage but achieve only {interval["empirical_coverage"]:.2%}, with average width {interval["average_width"]:.1f} rentals. Serial dependence and changing conditions mean ordinary exchangeability-based guarantees do not apply automatically. The worst boosting slice is {w["dimension"]}={w["slice"]}, with MAE {w.MAE:.1f} and RMSE {w.RMSE:.1f} across {int(w["n"])} hours. Weather changes, special events, missing history, reporting delays, and demand shifts remain important failure modes.')
    para('Before deployment, test the frozen decision rule prospectively and confirm that counts are available by issuance. Monitor missing hours, weekly MAE and RMSE against the concurrent baseline, commute-hour errors, interval coverage and width, and assumed-cost outcomes. Review sustained coverage below 90% or errors above baseline for two weeks; recalibrate or retrain using chronological data. Obtain station inventory and unmet-demand evidence before making station allocation claims. This project is a research prototype; it has not been deployed or submitted. OpenAI Codex assisted with coding, analysis and drafting; the team must verify and own the work.')
    para('Source: Fanaee-T &amp; Gama (2013), Event labeling combining ensemble detectors and background knowledge, doi:10.1007/s13748-013-0040-3. UCI Bike Sharing dataset, dataset 275. Full methods, plots, assumptions and checks accompany the executed notebooks.')
    doc=SimpleDocTemplate(str(ROOT/'Anshuman_memo.pdf'),pagesize=letter,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=42)
    def footer(canvas,doc):
        canvas.setFont('BodyFont',9);canvas.setFillColor(gray);canvas.drawString(48,22,f'Anshuman | ISOM 835 | {doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

# ==================== DATA LOADING AND COMMAND-LINE ENTRY ====================
def ensure_source_data(data_file=None):
    """Use real supplied data or verified HTTPS source; never synthesize data."""
    import shutil
    import urllib.request
    import io
    import zipfile
    expected = 'e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f'
    dest = ROOT/'data/hour.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = Path(data_file).expanduser().resolve() if data_file else Path(__file__).resolve().parent/'data/hour.csv'
    if not dest.exists():
        if local.is_file():
            payload = local.read_bytes()
        else:
            urls = [
                'https://raw.githubusercontent.com/agrawalanshuman/ISOM_835_Project/6a75d26e7e408164a8672c6d3d30d76a3efbff32/data/hour.csv',
                'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
            ]
            errors = []
            for url in urls:
                try:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        payload = response.read()
                    if url.endswith('.zip'):
                        payload = zipfile.ZipFile(io.BytesIO(payload)).read('hour.csv')
                    if hashlib.sha256(payload).hexdigest() != expected:
                        raise ValueError('Dataset checksum mismatch')
                    break
                except Exception as exc:
                    errors.append(f'{url}: {type(exc).__name__}: {exc}')
            else:
                raise RuntimeError('Real data could not be loaded. Supply --data-file hour.csv.\n'+'\n'.join(errors))
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Supplied data differ from the audited UCI source')
        dest.write_bytes(payload)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
        raise ValueError('Existing output data differ from the audited UCI source')
    return dest


def main(argv=None):
    """Execute the complete, unchanged analysis and regenerate all business outputs."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'run_outputs')
    parser.add_argument('--data-file', type=Path, help='Optional path to the original hour.csv')
    args = parser.parse_args(argv)
    global ROOT, OUT
    ROOT = args.output_dir.expanduser().resolve()
    OUT = ROOT/'visualizations'
    for folder in [ROOT, OUT, ROOT/'results', ROOT/'data']:
        folder.mkdir(parents=True, exist_ok=True)
    ensure_source_data(args.data_file)
    d = prepare()
    print('DATA AUDIT', json.dumps(d['audit'], indent=2), flush=True)
    print(d['raw'].head().to_string())
    print(d['raw'].dtypes.to_string())
    print(d['raw']['cnt'].describe().to_string())
    eda(d)
    ridge, ridge_pred = fit_first(d)
    first = pd.DataFrame({'actual': d['y'].loc[d['test']],
        'seasonal_naive': d['X'].loc[d['test'], 'lag_168'], 'ridge': ridge_pred})
    first_metrics = pd.DataFrame({n:score(first.actual,first[n]) for n in ['seasonal_naive','ridge']}).T
    first_metrics.to_csv(ROOT/'results/milestone2_metrics.csv')
    evaluation_plots(first)
    print('Tuning boosting on proper-training time folds only...', flush=True)
    a = run_advanced(d, ridge_pred)
    save_results(d, a)
    evaluation_plots(a['predictions'], a['slices'])
    cases = explain_cases(a['model'], d, a['predictions'])
    (ROOT/'results/shap_cases.json').write_text(json.dumps(cases, indent=2))
    for case in cases:
        print(case['explanation'])
    business_plot(a['costs'])
    (ROOT/'model_card.md').write_text(model_card(d,a))
    day = d['test'][0].normalize()
    forecast_day(ridge,d['grid'],day,int(d['grid'].loc[day,'holiday'])).to_csv(ROOT/'results/example_24_hour_forecast.csv')
    generate_pdfs()
    print('\nVERIFIED TEST METRICS\n'+a['metrics'].to_string())
    print('\nINTERVALS', json.dumps(a['intervals'],indent=2))
    print('\nCOST SENSITIVITY\n'+a['costs'].to_string(index=False))
    print('\nOutputs saved to:', ROOT)
    print('AI assistance: OpenAI Codex assisted with code and drafting. Team review required.')
    print('No GitHub push, deployment, or course submission is performed by this script.')
    return a


if __name__ == '__main__':
    main()
