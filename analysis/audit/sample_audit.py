from pathlib import Path
import zipfile, json, hashlib
import pandas as pd
import numpy as np

ROOT = Path('work/dataset-scout')
OUT = Path('outputs/olist-sample-audit-2026-10-16')
OUT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ROOT/'olist.zip') as z:
    def read(name): return pd.read_csv(z.open(name))
    orders = read('olist_orders_dataset.csv')
    customers = read('olist_customers_dataset.csv')
    items = read('olist_order_items_dataset.csv')
    products = read('olist_products_dataset.csv')
    translation = read('product_category_name_translation.csv')
    sellers = read('olist_sellers_dataset.csv')
for col in [c for c in orders if c.endswith('_date') or c.endswith('_timestamp') or c=='order_approved_at']:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')
assert orders.order_id.is_unique and customers.customer_id.is_unique
assert products.product_id.is_unique and sellers.seller_id.is_unique
o = orders.merge(customers, on='customer_id', validate='one_to_one', indicator=True)
assert o['_merge'].eq('both').all()
o = o.drop(columns='_merge').sort_values(['customer_unique_id','order_purchase_timestamp','order_id'])
assert o.customer_unique_id.notna().all()
monthly_orders = o.groupby(o.order_purchase_timestamp.dt.to_period('M')).agg(orders=('order_id','size'),approved=('order_approved_at','count'))
monthly_orders.index = monthly_orders.index.astype(str)
monthly_orders.to_csv(OUT/'source_monthly_orders.csv',encoding='utf-8-sig')
first = o.drop_duplicates('customer_unique_id',keep='first').copy().set_index('customer_unique_id')
min_dates = o.groupby('customer_unique_id').order_purchase_timestamp.transform('min')
ties = o[o.order_purchase_timestamp.eq(min_dates)].groupby('customer_unique_id').size()
first['first_timestamp_tied'] = ties.gt(1)
first['t0'] = first.order_delivered_customer_date
others = o.merge(first[['order_id','t0']].rename(columns={'order_id':'first_id'}),left_on='customer_unique_id',right_index=True,validate='many_to_one')
others = others[others.order_id.ne(others.first_id)].copy()
# Conservative landmark: exclude any other order placed before/at first delivery,
# including pending/canceled orders. Approval-only sensitivity is also counted.
prior = others[others.order_purchase_timestamp.le(others.t0)]
first['prior_order'] = first.index.isin(prior.customer_unique_id)
paid_prior = prior[prior.order_approved_at.notna() & prior.order_approved_at.le(prior.t0) & prior.order_approved_at.ge(prior.order_purchase_timestamp)]
first['prior_approved_order'] = first.index.isin(paid_prior.customer_unique_id)
following = others[others.order_purchase_timestamp.gt(others.t0) & others.order_approved_at.ge(others.order_purchase_timestamp)].copy()
# Both placement and approval must occur within the horizon; eventual status does
# not redefine an observed approval. Canceled-after-approval is reported separately.
following['event_time'] = following.order_approved_at
first['event_time'] = following.groupby('customer_unique_id').event_time.min()
for h in (60,90,120):
    first[f'y{h}'] = first.event_time.le(first.t0 + pd.Timedelta(days=h)).astype(int)
checks = {
    'raw_orders':len(o), 'raw_customers':len(first),
    'first_timestamp_ties':int(first.first_timestamp_tied.sum()),
    'orders_approved_before_purchase':int((o.order_approved_at < o.order_purchase_timestamp).sum()),
    'orders_delivered_before_purchase':int((o.order_delivered_customer_date < o.order_purchase_timestamp).sum()),
    'delivered_orders_missing_delivery':int((o.order_status.eq('delivered') & o.order_delivered_customer_date.isna()).sum()),
    'prior_order_without_known_approval_by_t0':int((first.prior_order & ~first.prior_approved_order).sum()),
}
flow=[]
cur=first.copy()
def stage(label, mask):
    global cur
    before=len(cur); cur=cur.loc[mask].copy()
    flow.append({'stage':label,'remaining':len(cur),'removed':before-len(cur)})
flow.append({'stage':'唯一客户 / 首次观察订单','remaining':len(cur),'removed':0})
stage('首单状态 delivered 且有送达时间',cur.order_status.eq('delivered') & cur.t0.notna())
stage('首购时间不并列且审批、送达时间有效',~cur.first_timestamp_tied & cur.order_approved_at.ge(cur.order_purchase_timestamp) & cur.t0.ge(cur.order_approved_at))
stage('首单送达前没有其他下单',~cur.prior_order)
base=cur.copy()
# July is the last complete calendar month before the observable August taper.
# This is a conservative analytical boundary, not a claimed official export date.
CUTOFF=pd.Timestamp('2018-08-01')
stage('90天结果在2018-08-01之前成熟',cur.t0+pd.Timedelta(days=90)<CUTOFF)
mature=cur.copy()
calendar_start=o.order_purchase_timestamp.min()+pd.Timedelta(days=180)
stage('首购前有180天日历跨度（含早期稀疏数据）',cur.order_purchase_timestamp.ge(calendar_start))
literal=cur.copy()
stable_start=pd.Timestamp('2017-01-01')+pd.Timedelta(days=180)
stage('180天窗口全部位于2017年起连续月份',cur.order_purchase_timestamp.ge(stable_start))
main=cur.copy()

pi=items.merge(products,on='product_id',validate='many_to_one').merge(translation,on='product_category_name',how='left',validate='many_to_one')
assert len(pi)==len(items)
pi['category']=pi.product_category_name_english.fillna(pi.product_category_name).fillna('UNKNOWN')
pi['volume_cm3']=pi.product_length_cm*pi.product_height_cm*pi.product_width_cm
agg=pi.groupby('order_id').agg(item_count=('order_item_id','size'),amount=('price','sum'),freight=('freight_value','sum'),seller_count=('seller_id','nunique'),category_count=('category','nunique'),category=('category','first'),weight_missing=('product_weight_g',lambda s:bool(s.isna().any())),dimensions_missing=('volume_cm3',lambda s:bool(s.isna().any())),unknown_category=('category',lambda s:bool(s.eq('UNKNOWN').any())),weight_g=('product_weight_g','sum'),volume_cm3=('volume_cm3','sum'))
main=main.join(agg,on='order_id',validate='one_to_one')
main['single_category']=main.category_count.eq(1)&~main.unknown_category.fillna(True)
main['late']=(main.t0.dt.normalize()>main.order_estimated_delivery_date.dt.normalize())
main['freight_ratio']=main.freight/main.amount
quality={
    'missing_items':int(main.item_count.isna().sum()),
    'missing_estimated_delivery':int(main.order_estimated_delivery_date.isna().sum()),
    'missing_weight_any_item':int(main.weight_missing.fillna(True).sum()),
    'missing_dimensions_any_item':int(main.dimensions_missing.fillna(True).sum()),
    'unknown_category_any_item':int(main.unknown_category.fillna(True).sum()),
    'multi_category_orders':int(main.category_count.gt(1).sum()),
    'nonpositive_amount':int(main.amount.le(0).sum()),
    'multi_seller_orders':int(main.seller_count.gt(1).sum()),
}
def stats(df):
    n=len(df); e=int(df.y90.sum())
    return {'customers':n,'repeat90':e,'rate_pct':100*e/n if n else None}
def categories(df):
    s=df[df.single_category]
    a=s.groupby('category').agg(customers=('y90','size'),repeat90=('y90','sum'),late_customers=('late','sum'),median_weight_g=('weight_g','median'),median_freight_ratio=('freight_ratio','median'))
    coverage=s.groupby(['category','customer_state']).size().ge(100).groupby('category').sum()
    a['states_ge100']=coverage
    a['passes_predeclared']=(a.customers.ge(2000)&a.repeat90.ge(50)&a.states_ge100.ge(5))
    a['rate_pct']=100*a.repeat90/a.customers
    return a.sort_values('customers',ascending=False)

# Calendar-based feasibility schedules; no fitted model, no feature/outcome effect search.
main['split']='unused_gap'
main.loc[main.t0.lt('2017-09-01'),'split']='train'
main.loc[main.t0.ge('2017-12-01') & main.t0.lt('2018-01-01'),'split']='validation'
main.loc[main.t0.ge('2018-04-01'),'split']='test'
train=main[main.split.eq('train')]
validation=main[main.split.eq('validation')]
test=main[main.split.eq('test')]
assert train.t0.max()+pd.Timedelta(days=90)<validation.t0.min()
assert validation.t0.max()+pd.Timedelta(days=90)<test.t0.min()
cats_all=categories(main); cats_train=categories(train)
cats_all.to_csv(OUT/'categories_full_cohort.csv',encoding='utf-8-sig')
cats_train.to_csv(OUT/'categories_training_only.csv',encoding='utf-8-sig')
monthly=main.groupby(main.t0.dt.to_period('M')).agg(customers=('y90','size'),repeat90=('y90','sum'))
monthly['rate_pct']=monthly.repeat90/monthly.customers*100
monthly.index=monthly.index.astype(str)
monthly.to_csv(OUT/'cohort_monthly.csv',encoding='utf-8-sig')
states=main.groupby('customer_state').agg(customers=('y90','size'),repeat90=('y90','sum'))
states['rate_pct']=100*states.repeat90/states.customers
states.sort_values('customers',ascending=False).to_csv(OUT/'states.csv',encoding='utf-8-sig')

# Point-in-time support for regional category history, not feature modeling.
# Use only paid orders whose approval is known before the focal purchase.
history=o[o.order_approved_at.ge(o.order_purchase_timestamp)].copy()
history['known_time']=history[['order_purchase_timestamp','order_approved_at']].max(axis=1)
histcat=pi[['order_id','category']].drop_duplicates().merge(history[['order_id','customer_unique_id','customer_state','order_purchase_timestamp','known_time']],on='order_id',validate='many_to_one')
histcat['weight']=1/histcat.groupby('order_id').category.transform('size')
histcat['expires']=histcat.order_purchase_timestamp+pd.Timedelta(days=180)
histcat=histcat[histcat.known_time<histcat.expires]
def hist_lookup(group_cols,target,weight_col=None):
    groups={}
    for key,g in histcat.groupby(group_cols):
        if not isinstance(key,tuple): key=(key,)
        weights=g[weight_col].to_numpy() if weight_col else np.ones(len(g))
        # Starts and ends form cumulative weighted events; start strictly before
        # focal purchase, expire only when purchase falls outside [t-180d,t).
        starts=g.known_time.to_numpy(dtype='datetime64[ns]'); ends=g.expires.to_numpy(dtype='datetime64[ns]')
        a=np.argsort(starts); b=np.argsort(ends)
        groups[key]=(starts[a],np.r_[0,np.cumsum(weights[a])],ends[b],np.r_[0,np.cumsum(weights[b])])
    result=[]
    for row in target.itertuples():
        key=tuple(getattr(row,c) for c in group_cols); t=row.order_purchase_timestamp.to_datetime64()
        if key not in groups: result.append(0.0); continue
        a,ca,b,cb=groups[key]
        result.append(float(ca[np.searchsorted(a,t,side='left')]-cb[np.searchsorted(b,t,side='left')]))
    return result
# Every focal row is its customer's first observed order, so earlier history
# cannot contain that same customer's order; checked by cohort construction.
main['regional_history_order_equivalents']=hist_lookup(['customer_state'],main,'weight')
main['regional_category_history_orders']=hist_lookup(['customer_state','category'],main)
hist_support={
    'state_history_equiv_quantiles':main.regional_history_order_equivalents.quantile([0,.1,.5,.9,1]).to_dict(),
    'state_history_below100':int(main.regional_history_order_equivalents.lt(100).sum()),
    'single_category_history_below20':int((main.single_category & main.regional_category_history_orders.lt(20)).sum()),
    'single_category_history_zero':int((main.single_category & main.regional_category_history_orders.eq(0)).sum()),
    'single_category_customers':int(main.single_category.sum()),
}

sens=[]
for cutoff in ['2018-08-01','2018-08-21','2018-08-30']:
    for history_start,label in [(None,'no_history_burnin'),(calendar_start,'180d_calendar'),(stable_start,'180d_since_2017')]:
        for horizon in [60,90,120]:
            d=base[(base.t0+pd.Timedelta(days=horizon)<pd.Timestamp(cutoff))]
            if history_start is not None: d=d[d.order_purchase_timestamp.ge(history_start)]
            sens.append({'cutoff_exclusive':cutoff,'history':label,'horizon_days':horizon,'customers':len(d),'repeat':int(d[f'y{horizon}'].sum()),'rate_pct':100*d[f'y{horizon}'].mean()})
pd.DataFrame(sens).to_csv(OUT/'sensitivity.csv',index=False,encoding='utf-8-sig')

# Independent full-cohort label verification with raw customer order groups.
order_groups={cid:g for cid,g in o.groupby('customer_unique_id') if len(g)>1}
independent=0
for cid,row in main.iterrows():
    if cid not in order_groups: continue
    g=order_groups[cid]; end=row.t0+pd.Timedelta(days=90)
    hit=(g.order_id.ne(row.order_id)&g.order_purchase_timestamp.gt(row.t0)&g.order_purchase_timestamp.le(end)&g.order_approved_at.ge(g.order_purchase_timestamp)&g.order_approved_at.le(end)).any()
    assert int(hit)==int(row.y90)
    independent+=int(hit)
assert independent==int(main.y90.sum())
assert main.index.is_unique and monthly.customers.sum()==len(main) and states.repeat90.sum()==main.y90.sum()
events90=following[following.customer_unique_id.isin(main.index)&following.event_time.le(following.t0+pd.Timedelta(days=90))]
checks['repeat_customers_with_any_final_canceled_order']=int(events90[events90.order_status.eq('canceled')].customer_unique_id.nunique())
checks['repeat_customers_with_any_final_unavailable_order']=int(events90[events90.order_status.eq('unavailable')].customer_unique_id.nunique())
checks['independent_raw_order_label_verification']='all main-cohort labels agree'
main.to_pickle(ROOT/'sample-audit-cohort.pkl')
summary={
    'source':'https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce',
    'source_archive_sha256':hashlib.sha256((ROOT/'olist.zip').read_bytes()).hexdigest(),
    'cutoff_exclusive':str(CUTOFF),'history_earliest_purchase':str(stable_start),
    'cutoff_reason':'Conservative complete-month boundary before late-August order taper; not a confirmed official observation endpoint.',
    'flow':flow,'checks':checks,'quality':quality,
    'mature_without_history_burnin':stats(mature),'literal_180d_history':stats(literal),'main':stats(main),
    'splits':{s:stats(main[main.split.eq(s)]) for s in ['train','validation','test','unused_gap']},
    'split_dates':{s:{'t0_min':str(main.loc[main.split.eq(s),'t0'].min()),'t0_max':str(main.loc[main.split.eq(s),'t0'].max())} for s in ['train','validation','test']},
    'train_category_passes':cats_train[cats_train.passes_predeclared].index.tolist(),
    'full_cohort_category_passes_not_selection':cats_all[cats_all.passes_predeclared].index.tolist(),
    'history_support':hist_support,
}
(OUT/'audit_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True,indent=2))
print('FULL COHORT CATEGORIES\n'+cats_all.head(15).to_string())
print('TRAIN CATEGORIES\n'+cats_train.head(12).to_string())
print('MONTHLY\n'+monthly.to_string())
