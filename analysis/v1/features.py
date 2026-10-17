from pathlib import Path
import zipfile, json, hashlib
import numpy as np
import pandas as pd

ROOT=Path('work/olist-logistic')
ROOT.mkdir(parents=True,exist_ok=True)
x=pd.read_pickle('work/dataset-scout/sample-audit-cohort.pkl').copy()
with zipfile.ZipFile('work/dataset-scout/olist.zip') as z:
    def read(n): return pd.read_csv(z.open(n))
    o=read('olist_orders_dataset.csv'); c=read('olist_customers_dataset.csv')
    items=read('olist_order_items_dataset.csv'); products=read('olist_products_dataset.csv')
    tr=read('product_category_name_translation.csv'); sellers=read('olist_sellers_dataset.csv')
    geo=read('olist_geolocation_dataset.csv')
for col in ['order_purchase_timestamp','order_approved_at']:
    o[col]=pd.to_datetime(o[col])
o=o.merge(c,on='customer_id',validate='one_to_one')
pi=items.merge(products,on='product_id',validate='many_to_one').merge(tr,on='product_category_name',how='left',validate='many_to_one')
pi['cat']=pi.product_category_name_english.fillna(pi.product_category_name).fillna('UNKNOWN')
cats=pi[['order_id','cat']].drop_duplicates()
cats['weight']=1/cats.groupby('order_id').cat.transform('size')
assert np.allclose(cats.groupby('order_id').weight.sum(),1)
hist=cats.merge(o[['order_id','customer_unique_id','customer_state','order_purchase_timestamp','order_approved_at']],on='order_id',validate='many_to_one')
hist=hist[hist.order_approved_at.ge(hist.order_purchase_timestamp)].copy()
hist['start']=hist.order_approved_at
hist['end']=hist.order_purchase_timestamp+pd.Timedelta(days=180)
hist=hist[hist.start<hist.end].copy()
focal=x.reset_index()[['customer_unique_id','order_id','customer_state','order_purchase_timestamp']].merge(cats,on='order_id',validate='one_to_many')
K=len(set(tr.product_category_name_english)|{'UNKNOWN'}|set(cats.cat))

def lookup(keys):
    out=np.zeros(len(focal))
    if not keys:
        hgroups={():hist}; qgroups={():focal}
    else:
        hgroups={k if isinstance(k,tuple) else (k,):v for k,v in hist.groupby(keys)}
        qgroups={k if isinstance(k,tuple) else (k,):v for k,v in focal.groupby(keys)}
    for key,q in qgroups.items():
        if key not in hgroups: continue
        h=hgroups[key]
        s=h.start.to_numpy(dtype='datetime64[ns]'); e=h.end.to_numpy(dtype='datetime64[ns]'); w=h.weight.to_numpy()
        si=np.argsort(s); ei=np.argsort(e)
        cs=np.r_[0,np.cumsum(w[si])]; ce=np.r_[0,np.cumsum(w[ei])]
        times=q.order_purchase_timestamp.to_numpy(dtype='datetime64[ns]')
        out[q.index]=cs[np.searchsorted(s[si],times,side='left')]-ce[np.searchsorted(e[ei],times,side='left')]
    assert out.min()>-1e-7
    return np.maximum(out,0)
for label,keys in [('local_total',['customer_state']),('local_category',['customer_state','cat']),('national_total',[]),('national_category',['cat'])]:
    focal[label]=lookup(keys)
focal['national_share']=(focal.national_category+.5)/(focal.national_total+.5*K)
focal['local_share']=(focal.local_category+100*focal.national_share)/(focal.local_total+100)
focal['log_relative']=np.log(focal.local_share/focal.national_share)
regional=focal.groupby('customer_unique_id').agg(local_category_share=('local_share','mean'),log_relative_share=('log_relative','mean'))
x=x.join(regional,validate='one_to_one')
# Direct verification uses no rolling lookup code and checks category weighting.
for _,row in focal.sample(30,random_state=20261016).iterrows():
    t=row.order_purchase_timestamp
    h=hist[hist.order_purchase_timestamp.ge(t-pd.Timedelta(days=180))&hist.order_purchase_timestamp.lt(t)&hist.order_approved_at.lt(t)]
    assert not h.customer_unique_id.eq(row.customer_unique_id).any()
    for name,mask in [('national_total',np.ones(len(h),dtype=bool)),('national_category',h.cat.eq(row['cat'])),('local_total',h.customer_state.eq(row.customer_state)),('local_category',h.customer_state.eq(row.customer_state)&h.cat.eq(row['cat']))]:
        assert abs(h.loc[mask,'weight'].sum()-row[name])<1e-6
assert x.local_category_share.between(0,1).all() and np.isfinite(x.log_relative_share).all()

# Postal-code medians avoid duplicating orders when a prefix has many coordinates.
g=geo[geo.geolocation_lat.between(-90,90)&geo.geolocation_lng.between(-180,180)].groupby('geolocation_zip_code_prefix')[['geolocation_lat','geolocation_lng']].median()
locations=x[['order_id','customer_zip_code_prefix']].merge(items[['order_id','seller_id']],on='order_id',validate='one_to_many').merge(sellers[['seller_id','seller_zip_code_prefix']],on='seller_id',validate='many_to_one')
for prefix,col in [('customer','customer_zip_code_prefix'),('seller','seller_zip_code_prefix')]:
    locations[prefix+'_lat']=locations[col].map(g.geolocation_lat)
    locations[prefix+'_lon']=locations[col].map(g.geolocation_lng)
lat1,lon1,lat2,lon2=[np.deg2rad(locations[k].to_numpy()) for k in ['customer_lat','customer_lon','seller_lat','seller_lon']]
a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
locations['distance_km']=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
dist=locations.groupby('order_id').agg(distance_km=('distance_km','mean'),distance_missing=('distance_km',lambda s:bool(s.isna().any())))
x=x.join(dist,on='order_id',validate='one_to_one')
x.loc[x.distance_missing,'distance_km']=np.nan
x.loc[x.weight_missing,'weight_g']=np.nan
x.loc[x.dimensions_missing,'volume_cm3']=np.nan
for name in ['amount','item_count','weight_g','volume_cm3','distance_km']:
    x['log_'+name]=np.log1p(x[name])
x['multi_seller']=(x.seller_count>1).astype(int)
x['month_sin']=np.sin(2*np.pi*(x.order_purchase_timestamp.dt.month-1)/12)
x['month_cos']=np.cos(2*np.pi*(x.order_purchase_timestamp.dt.month-1)/12)
x['category_model']=np.where(x.category_count.gt(1),'MULTI_CATEGORY',x.category)
x['delivery_days']=(x.t0-x.order_purchase_timestamp).dt.total_seconds()/86400
x['late_days']=(x.t0.dt.normalize()-x.order_estimated_delivery_date.dt.normalize()).dt.days.clip(lower=0)
x['log_delivery_days']=np.log1p(x.delivery_days)
x['log_late_days']=np.log1p(x.late_days)
x['log_freight_ratio']=np.log1p(x.freight_ratio)
for name in ['weight_missing','dimensions_missing','distance_missing','late']:
    x[name]=x[name].astype(int)
assert len(x)==51981 and x.y90.sum()==567
assert x.index.is_unique
assert (x.t0+pd.Timedelta(days=90)<pd.Timestamp('2018-08-01')).all()
assert x.event_time.isna().any()  # label-only field retained in cache, never allowed in X
x.to_pickle(ROOT/'model_frame.pkl')
focal.to_pickle(ROOT/'regional_feature_components.pkl')
checks={'rows':len(x),'events90':int(x.y90.sum()),'regional_direct_checks':30,'history_days':180,'shrinkage_orders':100,'national_prior_per_category':.5,'taxonomy_categories':K,'distance_missing':int(x.distance_missing.sum()),'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'source_zip_sha256':hashlib.sha256(Path('work/dataset-scout/olist.zip').read_bytes()).hexdigest()}
(ROOT/'feature_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks))
