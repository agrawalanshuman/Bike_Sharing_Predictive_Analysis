"""Historical Streamlit viewer; no model fitting or solver calls."""
import pandas as pd
from .artifacts import load_saved_analysis

def render_dashboard():
    """Explore saved historical outputs; no retraining, live data, or solver calls."""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go

    st.set_page_config(page_title='Bike Sharing | Anshuman', page_icon='🚲', layout='wide')
    st.title('Bike Sharing Predictive Analysis')
    st.caption('Anshuman · Demand forecasting and staffing scenario analysis')
    st.write('Can next-day hourly forecasts improve staffing plans compared with repeating last week?')
    st.info('Historical demo: explore saved 2012 forecasts and hypothetical staffing plans. '
            'This app does not predict current demand or report realized savings.')
    try:
        saved = load_saved_analysis()
        p, m, plans, costs, daily = (saved[k] for k in ['predictions','metrics','plans','totals','daily'])
    except (OSError, ValueError, KeyError, TypeError):
        st.error('Saved results are missing or inconsistent. Restore the complete audited repository '
                 'and run python bike_sharing.py --verify before viewing the demo.')
        st.stop()

    labels = {'seasonal_naive':'Same hour last week', 'ridge':'Ridge regression',
              'boosting':'Tuned gradient boosting', 'flat_staffing':'Fixed staffing'}
    improvement = 100*(1-m.loc['ridge','MAE']/m.loc['seasonal_naive','MAE'])
    a,b,c = st.columns(3)
    a.metric('Ridge MAE · rentals/hour', f"{m.loc['ridge','MAE']:.2f}")
    b.metric('MAE reduction vs. last week', f'{improvement:.2f}%')
    c.metric('Comparable held-out hours', f'{len(p):,}')
    st.success('Decision insight: pilot Ridge first. It has the lowest held-out MAE and '
               'the lowest hypothetical staffing cost among the tested policies. '
               'Validate prospectively before changing operations.')
    forecasts, staffing, methods = st.tabs(['Explore forecasts', 'Explore staffing', 'How it was evaluated'])
    with forecasts:
        model = st.selectbox('Forecast model', ['ridge','boosting','seasonal_naive'],
                             format_func=labels.get, key='forecast_model')
        dates = sorted(set(p.index.date))
        first_complete = next((i for i,d in enumerate(dates) if len(p.loc[str(d)])==24), 0)
        date = st.selectbox('Forecast day', dates, index=first_complete, key='forecast_day')
        day = p.loc[str(date)]
        st.caption(f'{len(day)} of 24 hourly slots have comparable observations on this day. '
                   'Missing targets or weekly references are excluded from scoring; gaps remain visible.')
        # Reindex for display only: never bridge gaps or invent missing observations.
        grid = day.reindex(pd.date_range(str(date), periods=24, freq='h'))
        fig = go.Figure()
        for column in dict.fromkeys(['actual','seasonal_naive',model]):
            fig.add_trace(go.Scatter(x=grid.index, y=grid[column], mode='lines+markers',
                                    name=labels.get(column,'Observed rentals'), connectgaps=False))
        if model=='boosting' and st.checkbox('Show boosting prediction interval', value=True):
            fig.add_trace(go.Scatter(x=grid.index,y=grid.upper_90,mode='lines',line={'width':0},
                                    name='Upper bound',showlegend=False,connectgaps=False))
            fig.add_trace(go.Scatter(x=grid.index,y=grid.lower_90,mode='lines',line={'width':0},
                                    fill='tonexty',name='90% nominal interval',connectgaps=False))
            st.caption(f"Nominal coverage: 90%; measured test coverage: {saved['summary']['intervals']['empirical_coverage']:.2%}. "
                       'Temporal dependence limits the standard conformal guarantee.')
        fig.update_layout(title=f'Historical forecasts · {date}', xaxis_title='Target hour', yaxis_title='Completed rentals',
                          legend={'orientation':'h'}, template='plotly_white')
        st.plotly_chart(fig, width='stretch')
        st.markdown('**Same-sample model comparison**')
        comparison=m[['MAE','RMSE','MAE_improvement_pct']].rename(index=labels,
            columns={'MAE_improvement_pct':'MAE reduction vs. baseline (%)'})
        st.dataframe(comparison.round(2), width='stretch')
        st.download_button('Download this day’s scored forecasts', day.to_csv().encode(),
                           file_name=f'bike-forecasts-{date}.csv', mime='text/csv', on_click='ignore')
    with staffing:
        st.write('Forecasts feed a Gurobi model with six four-hour shifts and at most 48 '
                 'worker-shift assignments per day. Actual rentals evaluate the plan afterward.')
        st.caption('Assumptions: 40 rental-service units per worker-hour; USD 20 hourly wage; '
                   'USD 5 per shortage unit and USD 1 per excess unit. Assignments are not unique '
                   'employees or numbers of bicycles. The viewer reads saved solver outputs.')
        totals=costs.reset_index()
        totals['Planning method']=totals.method.map(labels)
        st.plotly_chart(px.bar(totals,x='Planning method',y='total_cost',
                              labels={'total_cost':'Hypothetical total cost (USD)'},
                              title='48 complete days · 1,152 comparable hours',
                              color='Planning method'),width='stretch')
        plan_model=st.selectbox('Planning method',list(labels),format_func=labels.get,key='plan_model')
        plan_date=st.selectbox('Planning day',sorted(plans.date.unique()),key='plan_date')
        selected=plans[(plans.method==plan_model)&(plans.date==plan_date)].sort_values('start_hour')
        st.dataframe(selected[['start_hour','assignments']].rename(
            columns={'start_hour':'Shift start (hour)','assignments':'Worker-shift assignments'}),
            hide_index=True,width='stretch')
        row=daily[(daily.method==plan_model)&(daily.date==plan_date)].iloc[0]
        st.metric('Selected day · hypothetical evaluated cost',f"${row.total_cost:,.0f}")
        st.caption('144 model-based solves were certified optimal. Eight partial days are excluded equally. '
                   'This exploratory policy analysis was designed after forecast scores were known; '
                   'it needs prospective validation and does not establish causal savings.')
    with methods:
        st.markdown('''**Time-based evaluation, with no random train/test split**

- Fit: January 8, 2011, 23:00 through October 7, 2012, 23:00.
- Interval calibration: October 8–November 4, 2012.
- Held-out testing: November 5–December 30, 2012.
- Tune boosting only on expanding chronological folds within training.
- Issue each next-day forecast at 11:59 PM the preceding evening. Update historical lags only after observation; model parameters stay frozen.

**Leakage controls:** exclude `casual` and `registered` (target components), future observed weather, and current-hour counts. Fit imputers, scaling, and encoding only on training data. Preserve missing targets.

**Limits:** completed rentals do not measure unmet demand, station inventories, or unique bicycles required. The test covers autumn and winter. SHAP is descriptive; interval coverage falls below its nominal target.

**AI assistance:** OpenAI Codex helped debug and resolve issues, and assisted with implementation, tests, analysis and documentation. Anshuman is responsible for reviewing and explaining the work.

[Source code and full analysis](https://github.com/agrawalanshuman/Bike_Sharing_Predictive_Analysis) · [UCI dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
''')
