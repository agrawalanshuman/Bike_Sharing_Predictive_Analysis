"""Business-facing documents use persisted, executed results only."""
from pathlib import Path
import json
import pandas as pd
from .forecasting import ROOT


def model_card(d,a):
    m=a['metrics']; it=a['intervals']; s=a['slices'].query("model == 'boosting'")
    w=s.loc[s.MAE.idxmax()]
    return f'''# Model card — next-day hourly system-wide rentals
**Owner:** TeamName, [NAMES TO FILL IN]; hypothetical operations manager is the decision owner.
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
    cv=canvas.Canvas(str(ROOT/'TeamName_dryrun.pdf'),pagesize=(W,H))
    cv.setTitle('Next-Day Hourly Bike-Sharing Demand — TeamName')
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
        cv.setFont('BodyBold',12);cv.drawString(50,H-44,'ISOM 835  /  TEAMNAME')
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
    para('To: Hypothetical Bike-Sharing Operations Manager<br/>From: TeamName — [NAMES TO FILL IN]<br/>Subject: Historical forecasting evidence and a proposed pilot')
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
    doc=SimpleDocTemplate(str(ROOT/'TeamName_memo.pdf'),pagesize=letter,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=42)
    def footer(canvas,doc):
        canvas.setFont('BodyFont',9);canvas.setFillColor(gray);canvas.drawString(48,22,f'TeamName | ISOM 835 | {doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
