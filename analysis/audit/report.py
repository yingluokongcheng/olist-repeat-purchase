from pathlib import Path
import pandas as pd
import numpy as np
import json, zipfile, html, math

out=Path('outputs/olist-sample-audit-2026-10-16')
x=pd.read_pickle('work/dataset-scout/sample-audit-cohort.pkl')
r=json.loads((out/'audit_results.json').read_text(encoding='utf-8'))
with zipfile.ZipFile('work/dataset-scout/olist.zip') as z:
    geo=pd.read_csv(z.open('olist_geolocation_dataset.csv'))
    sellers=pd.read_csv(z.open('olist_sellers_dataset.csv'))
    items=pd.read_csv(z.open('olist_order_items_dataset.csv'))
    orders=pd.read_csv(z.open('olist_orders_dataset.csv'),parse_dates=['order_purchase_timestamp','order_approved_at'])
    customers=pd.read_csv(z.open('olist_customers_dataset.csv'))
valid_geo=geo[geo.geolocation_lat.between(-90,90)&geo.geolocation_lng.between(-180,180)]
zip_codes=set(valid_geo.geolocation_zip_code_prefix)
seller_coverage=sellers.set_index('seller_id').seller_zip_code_prefix.isin(zip_codes)
items['seller_geo_available']=items.seller_id.map(seller_coverage).fillna(False)
order_seller_geo=items.groupby('order_id').seller_geo_available.all()
r['quality']['customer_zip_without_geolocation']=int((~x.customer_zip_code_prefix.isin(zip_codes)).sum())
r['quality']['orders_with_any_seller_without_geolocation']=int((~x.order_id.map(order_seller_geo)).sum())
r['quality']['missing_customer_state']=int(x.customer_state.isna().sum())

# A second implementation spot-checks point-in-time state history support.
o=orders.merge(customers,on='customer_id',validate='one_to_one')
weights=items.groupby('order_id').size()
for cid,row in x.sample(n=40,random_state=20261016).iterrows():
    t=row.order_purchase_timestamp
    h=o[(o.customer_state==row.customer_state)&o.order_purchase_timestamp.ge(t-pd.Timedelta(days=180))&o.order_purchase_timestamp.lt(t)&o.order_approved_at.ge(o.order_purchase_timestamp)&o.order_approved_at.lt(t)&o.order_id.isin(weights.index)]
    assert cid not in set(h.customer_unique_id)
    assert abs(len(h)-row.regional_history_order_equivalents)<1e-6
r['checks']['regional_history_independent_check']='40 deterministic random rows agree with direct filtering; own-customer exclusion checked'

refit=x[x.t0<pd.Timestamp('2018-01-01')]
single=refit[refit.single_category]
a=single.groupby('category').agg(customers=('y90','size'),repeat90=('y90','sum'))
a['states_ge100']=single.groupby(['category','customer_state']).size().ge(100).groupby('category').sum()
a['passes_predeclared']=(a.customers.ge(2000)&a.repeat90.ge(50)&a.states_ge100.ge(5))
a=a.sort_values('customers',ascending=False)
a.to_csv(out/'categories_pretest_refit.csv',encoding='utf-8-sig')
r['pretest_refit']={'cutoff_t0_exclusive':'2018-01-01','customers':len(refit),'repeat90':int(refit.y90.sum()),'category_passes':a[a.passes_predeclared].index.tolist(),'note':'Optional final refit before Apr 2018 test, after settings are fixed. Not an additional validation set.'}
cases=['bed_bath_table','sports_leisure']
cells=x[x.single_category & x.category.isin(cases)].groupby(['category','late']).agg(customers=('y90','size'),repeat90=('y90','sum')).reset_index()
cells.to_csv(out/'candidate_case_cells.csv',index=False,encoding='utf-8-sig')
n=len(x); e=int(x.y90.sum()); p=e/n; z=1.959963984540054
den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
r['main']['wilson95_pct']=[100*(center-half),100*(center+half)]
r['case_recommendation']={'descriptive_candidates':cases,'selection_basis':'Top two single-category customer counts in the pretest refit pool; not selected for observed effect or significance. Neither passes the original training-only event threshold.','standalone_category_models_recommended':False}
r['checks']['models_fitted']=0
(out/'audit_results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')

def table(df): return df.to_html(index=False,border=0,escape=True,classes='audit',float_format=lambda v:f'{v:,.2f}')
flow=pd.DataFrame(r['flow']).rename(columns={'stage':'筛选步骤','remaining':'剩余客户','removed':'本步排除'})
allcats=pd.read_csv(out/'categories_full_cohort.csv')
traincats=pd.read_csv(out/'categories_training_only.csv')
selected=allcats.head(8)[['category','customers','repeat90','states_ge100']].merge(a.reset_index()[['category','customers','repeat90']],on='category',suffixes=('_all','_pretest'))
selected=selected.rename(columns={'category':'品类','customers_all':'全队列客户','repeat90_all':'全队列复购','states_ge100':'全队列≥100人州数','customers_pretest':'测试前训练池客户','repeat90_pretest':'测试前训练池复购'})
monthly=pd.read_csv(out/'cohort_monthly.csv').rename(columns={'t0':'首单送达月','customers':'客户','repeat90':'90天复购','rate_pct':'复购率（%）'})
states=pd.read_csv(out/'states.csv')
sens=pd.read_csv(out/'sensitivity.csv')
sens=sens[(sens.history=='180d_since_2017') & (sens.horizon_days==90)].drop(columns=['history','horizon_days'])
split=pd.DataFrame([
    ['初始训练','2017-07-04 至 08-31',6459,71],
    ['验证','2017-12-01 至 12-31',6862,60],
    ['最终测试','2018-04-01 至 05-02',7893,87],
    ['可选：测试前重新训练','2017-07-04 至 12-31',25780,303],
],columns=['用途','首单送达区间','客户','复购事件'])
quality=pd.DataFrame(list(r['quality'].items()),columns=['检查项','客户/订单数'])
source=html.escape(r['source'])
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Olist 样本审计 · 2026-10-16</title>
<style>body{{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#172b42;background:#f3f6fa;margin:0;line-height:1.75}}main{{max-width:1120px;margin:30px auto;background:white;padding:36px;border-radius:12px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:34px}}.lead{{background:#eaf1fb;padding:18px;border-left:5px solid #3267a8}}.warn{{background:#fff4df;padding:16px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}}th,td{{padding:9px 12px;border-bottom:1px solid #dde4ec;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eff3f8}}.scroll{{overflow-x:auto}}code{{word-break:break-all}}a{{color:#245ea4}}small{{color:#526277}}@media(max-width:700px){{main{{margin:0;padding:18px}}}}@media print{{main{{margin:0;padding:0}}body{{background:white}}}}</style><main>
<h1>Olist 样本审计</h1><p>2026-10-16 · 已完成样本与可行性核验，未训练模型</p>
<div class="lead"><b>结论：</b>保守主队列有 <b>51,981</b> 位客户、<b>567</b> 位90天内复购客户，复购率 <b>1.09%</b>。可支持汇总分析和克制的整体模型；原定多个品类独立建模及复杂交互证据不足。</div>
<h2>1. 审计口径与边界</h2>
<p>每个 customer_unique_id 取首次观察订单。首单必须已送达，审批时间不早于下单、送达不早于审批；排除首购时间并列及首单送达前已有其他订单的客户。送达时间为预测起点。90天内出现另一笔新下单且完成审批的订单记为复购；主标签不根据后续最终状态改写。</p>
<p>观察截止采用 <b>2018-08-01 00:00（不含）</b>。9—10月仅20笔订单，8月下旬也明显缩减，因此用7月底作为保守完整月份边界。这不是官方确认的数据截止日，亦不能证明此前记录完全无遗漏。主队列要求首购前180天全部位于2017年1月起连续有订单的月份，首购最早为2017-06-30；此条件比仅满足日历跨度更严格。</p>
<p>首单送达前的其他下单采用保守排除，即使该订单尚未审批；原始客户中这种排除与“当时已审批”口径相差16人。首购时间并列客户不任意指定首单。复购标签包含4位出现最终取消订单、2位出现最终 unavailable 订单的客户，两组可能重叠，后续可作敏感性分析。</p>
<h2>2. 样本流失</h2>{table(flow)}
<p>不要求历史窗口时，完整90天样本为65,493人/716次复购；只要求180天日历跨度为62,733人/693次复购。主队列采用更保守的连续月份窗口，为51,981人/567次复购。主队列复购率的客户级 Wilson 95%区间为 {r['main']['wilson95_pct'][0]:.2f}%—{r['main']['wilson95_pct'][1]:.2f}%；该区间不涵盖渠道覆盖与记录缺失等系统误差。</p>
<h2>3. 品类：原门槛未通过，建议只保留两个描述案例</h2>
<p>原门槛为训练期至少2,000客户、50复购事件、至少5州各100人。严格三段时间切分的初始训练期没有品类通过；即使扩大到测试前最终训练池，仍没有品类通过。全队列中的床浴桌用品和运动休闲虽然通过数量门槛，不能因此冒充训练期合格品类。</p>
<div class="scroll">{table(selected)}</div>
<p>建议的描述案例为 <b>bed_bath_table（床上、浴室及餐桌用品）</b>和<b>sports_leisure（运动休闲）</b>，依据是测试前训练池的客户数量前两名，而非效应显著程度。它们都不是单一商品或高频消耗品的同义词。</p>
<div class="warn">全队列中，这两个品类的延误客户分别为478和341人，但复购事件仅5和3个。继续拆成地区偏好高低、城市或品类模型后，事件会更少。只能展示样本和区间，不能承诺得到可靠的细分机制结论。</div>
<h2>4. 时间切分的代价</h2>{table(split)}
<p>初始训练的90天结果在验证开始前已知，验证的90天结果在测试开始前已知。严格三段划分下，30,767人处于间隔月份，不能直接全部并入初始训练。设定固定后可将2017年年底前已成熟客户用于测试前重新训练，共25,780人/303个复购事件；这是重新训练池，不是额外独立验证集。</p>
<p>以上划分是一个满足时间顺序的可行性方案，并非穷尽所有可能划分。建议后续优先简化逻辑回归、减少参数与调参次数；LightGBM仅作为探索性对照。不得用已查看的分组测试结果选择交互项或阈值。本次仅检查样本/事件数量，未比较模型或检验特征效应。</p>
<h2>5. 地区历史信息是否足够</h2>
<p>州级历史订单等价数中位数为4,148，但589位客户所在州的过去180天记录不足100单。单一已知品类客户50,739人中，7,074人对应的州×品类历史订单少于20笔，其中492人没有相应历史订单。历史信息仅使用首购前已知审批结果，排除自身；多品类订单分摊总权重1。</p>
<p>建议保留州级分析，小样本向全国结构收缩并标记低支持度；不开展城市级独立模型。SP、RJ、MG三州占较大比例，小州的复购率容易受少数事件影响。</p>
<div class="scroll">{table(states.rename(columns={'customer_state':'州','customers':'客户','repeat90':'复购','rate_pct':'复购率（%）'}))}</div>
<h2>6. 数据质量与时间分布</h2>{table(quality)}
<p>没有首单商品明细缺失、预计送达日期缺失或商品总额非正值。商品重量和尺寸各有3笔订单至少一项缺失；894笔订单含未知品类，383笔含多个品类，两者可能重叠。品类案例使用单一已知品类订单，主队列不因此整行删除。邮编坐标覆盖检查仅证明存在合法经纬度记录，不代表坐标准确或等同运输距离。</p>
{table(monthly)}
<h2>7. 截止日敏感性（均为90天、相同历史条件）</h2>{table(sens)}
<p>延长到8月20日或29日，事件数升至643或649，仍不足以支持原计划的多个独立品类模型。60/90/120天敏感性表的可纳入客户不同，不能把其率差直接解释为同一批客户的累计增长。</p>
<h2>8. 核验与来源</h2>
<p>订单、客户、商品和卖家主键及关联粒度已核对；主队列全部复购标签由原始订单组另一套逐客户逻辑复核一致；月度、州级汇总与总数对齐；40条固定随机样本的州级历史支持度经直接过滤复核一致。此次未训练模型、未上传GitHub。</p>
<p><a href="{source}">Olist 原始数据发布页</a> · ZIP SHA-256：<code>{r['source_archive_sha256']}</code></p>
<p><a href="audit_results.json">机器可读审计结果</a> · <a href="categories_training_only.csv">初始训练期品类</a> · <a href="categories_pretest_refit.csv">测试前训练池品类</a> · <a href="categories_full_cohort.csv">全队列品类</a> · <a href="sensitivity.csv">完整敏感性表</a></p>
</main></html>'''
(out/'Olist样本审计.html').write_text(body,encoding='utf-8')
assert len(x)==sum(r['splits'][s]['customers'] for s in r['splits'])
assert sum(r['splits'][s]['repeat90'] for s in r['splits'])==567
print(json.dumps({'pretest_refit':r['pretest_refit'],'quality':r['quality'],'ci':r['main']['wilson95_pct'],'report':str(out/'Olist样本审计.html')},ensure_ascii=True,indent=2))
