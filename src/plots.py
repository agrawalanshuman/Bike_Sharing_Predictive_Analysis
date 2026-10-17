"""Figures and faithful local explanations from fitted model outputs."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .forecasting import ROOT

plt.rcParams.update({'figure.dpi':110,'savefig.dpi':150,'axes.spines.top':False,
                     'axes.spines.right':False,'font.size':10})
OUT=ROOT/'visualizations'
OUT.mkdir(exist_ok=True)

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
