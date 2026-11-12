"""Forecast-only Gurobi decisions and separate retrospective evaluation.

Importing this module and evaluating saved plans require no Gurobi package.
Only optimize_staffing/run_optimization import the optional solver.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import STAFFING, PROJECT_ROOT
from .validation import require, validate_staffing_assumptions, validate_hourly_values

def _gurobi_backend():
    try:
        import gurobipy
    except ImportError as exc:
        raise RuntimeError('The optional solver is not installed. Run python -m pip install '
                           '-r requirements-gurobi.txt and configure a usable Gurobi license.') from exc
    return gurobipy


def staffing_cost(actual, workers, assumptions=None):
    """Evaluate a frozen plan; actuals never enter optimization."""
    p = validate_staffing_assumptions(assumptions)
    actual = validate_hourly_values(actual, 'Actuals')
    workers = np.asarray(workers, dtype=float)
    require(workers.shape == (6,) and np.isfinite(workers).all()
            and np.equal(workers,np.floor(workers)).all() and (workers >= 0).all()
            and (workers <= p['max_per_shift']).all() and workers.sum() <= p['assignment_budget'],
            'Supply six feasible nonnegative integer worker-shift assignments')
    capacity=np.repeat(workers*p['throughput'],int(p['shift_hours']))
    return {'wages':float(np.sum(workers)*p['shift_hours']*p['hourly_wage']),
            'shortage_units':float(np.maximum(actual-capacity,0).sum()),
            'excess_units':float(np.maximum(capacity-actual,0).sum()),
            'total_cost':float(np.sum(workers)*p['shift_hours']*p['hourly_wage']+
                p['under_cost']*np.maximum(actual-capacity,0).sum()+
                p['excess_cost']*np.maximum(capacity-actual,0).sum())}


def optimize_staffing(forecast, assumptions=None):
    """Six integer 4-hour shift assignments; only 24 issued forecasts are inputs."""
    p = validate_staffing_assumptions(assumptions)
    f = validate_hourly_values(forecast, 'Forecast')
    gp = _gurobi_backend()
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            with gp.Model('next_day_capacity',env=env) as model:
                model.Params.Seed=835;model.Params.Threads=2;model.Params.MIPGap=0
                model.Params.TimeLimit=30
                x=model.addVars(6,vtype=gp.GRB.INTEGER,lb=0,ub=p['max_per_shift'],name='assignments')
                short=model.addVars(24,lb=0,name='shortage')
                excess=model.addVars(24,lb=0,name='excess')
                model.addConstr(gp.quicksum(x.values())<=p['assignment_budget'])
                for h in range(24):
                    model.addConstr(p['throughput']*x[h//4]+short[h]-excess[h]==f[h])
                model.setObjective(p['hourly_wage']*4*gp.quicksum(x.values())+
                    p['under_cost']*gp.quicksum(short.values())+
                    p['excess_cost']*gp.quicksum(excess.values()),gp.GRB.MINIMIZE)
                model.optimize()
                if model.Status!=gp.GRB.OPTIMAL:
                    raise RuntimeError(f'Gurobi did not certify optimality: status {model.Status}')
                workers=np.array([round(x[s].X) for s in range(6)],dtype=int)
                return workers,{'status':'OPTIMAL','objective':float(model.ObjVal),
                                'bound':float(model.ObjBound),'mip_gap':float(model.MIPGap),
                                'variables':model.NumVars,'constraints':model.NumConstrs}
    except gp.GurobiError as error:
        if error.errno == gp.GRB.Error.NO_LICENSE:
            raise RuntimeError('Gurobi license unavailable. Configure a valid license securely; '
                               'no substitute solver or invented result is used.') from error
        raise RuntimeError(f'Gurobi solver error {error.errno}; no certified solution returned') from error


def run_optimization(predictions,root=PROJECT_ROOT):
    """Retrospective extension on complete common days, with fixed assumptions."""
    root = Path(root)
    from .visualization import finish
    import matplotlib.pyplot as plt
    gp = _gurobi_backend()
    from plotly import graph_objects as go
    daily=[];hourly=[];plans=[]
    for date,frame in predictions.groupby(predictions.index.normalize()):
        if len(frame)!=24 or list(frame.index.hour)!=list(range(24)):
            continue
        for method in ['flat_staffing','seasonal_naive','ridge','boosting']:
            if method=='flat_staffing':
                workers=np.full(6,8);certificate={'status':'FIXED','objective':None,'bound':None,'mip_gap':None,'variables':0,'constraints':0}
            else:
                workers,certificate=optimize_staffing(frame[method].to_numpy())
            evaluated=staffing_cost(frame.actual,workers)
            daily.append({'date':str(date.date()),'method':method,'assignments':int(workers.sum()),
                          **evaluated,**certificate})
            for shift,count in enumerate(workers):
                plans.append({'date':str(date.date()),'method':method,'start_hour':4*shift,'assignments':int(count)})
            for h,t in enumerate(frame.index):
                hourly.append({'timestamp':str(t),'method':method,'actual':float(frame.actual.iloc[h]),
                               'capacity':float(workers[h//4]*STAFFING['throughput'])})
    daily=pd.DataFrame(daily);hourly=pd.DataFrame(hourly);plans=pd.DataFrame(plans)
    if daily.empty:raise ValueError('No complete common days available for staffing evaluation')
    totals=daily.groupby('method')[['wages','shortage_units','excess_units','total_cost']].sum()
    sensitivity=[]
    for under in [2.,5.,10.]:
        for over in [.5,1.,2.]:
            for method,row in totals.iterrows():
                sensitivity.append({'under_cost':under,'excess_cost':over,'method':method,
                    'fixed_plan_cost':row.wages+under*row.shortage_units+over*row.excess_units})
    sensitivity=pd.DataFrame(sensitivity)
    out=root/'results';out.mkdir(exist_ok=True)
    for name,frame in [('staffing_daily',daily),('staffing_hourly',hourly),('staffing_plans',plans),('staffing_sensitivity',sensitivity)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    totals.to_csv(out/'staffing_metrics.csv')
    summary={'assumptions':STAFFING,'complete_days':int(daily.date.nunique()),
        'evaluated_hours':int(daily.date.nunique()*24),'excluded_partial_days':int(predictions.index.normalize().nunique()-daily.date.nunique()),
        'optimal_solves':int((daily.status=='OPTIMAL').sum()),'gurobi_version':'.'.join(map(str,gp.gurobi.version())),
        'interpretation':'Exploratory retrospective scenario; assumptions were added after forecast test results were known. Not a fresh holdout, causal savings, individual roster or station allocation.'}
    (out/'staffing_summary.json').write_text(json.dumps(summary,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    totals.total_cost.plot.bar(ax=axs[0],color='#127d92',rot=15)
    axs[0].set(title='Hypothetical staffing cost',ylabel='USD across complete common days')
    first=hourly[hourly.timestamp.str.startswith(daily.date.iloc[0])]
    for method,rows in first.groupby('method'):
        axs[1].step(range(24),rows.capacity,label=method,where='post')
    axs[1].plot(range(24),first[first.method=='ridge'].actual,color='black',label='observed rentals')
    axs[1].set(title='First eligible day: planned throughput',xlabel='Hour',ylabel='Rental-service units/hour');axs[1].legend(fontsize=7)
    finish(fig,'staffing_optimization',root/'visualizations')
    interactive=go.Figure()
    for method,rows in daily.groupby('method'):
        interactive.add_trace(go.Scatter(x=rows.date,y=rows.total_cost,name=method,mode='lines+markers'))
    interactive.update_layout(title='Anshuman | Forecast-to-staffing scenarios',xaxis_title='Historical target day',
                              yaxis_title='Hypothetical daily cost (USD)',template='plotly_white')
    interactive.write_html(str(root/'visualizations/staffing_interactive.html'),include_plotlyjs=True)
    return {'summary':summary,'daily':daily,'totals':totals,'sensitivity':sensitivity}
