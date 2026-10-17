from pathlib import Path
import json, hashlib, warnings, math, shutil
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

W=Path('work/olist-calibration');O=Path('outputs/olist-calibration-v2-2026-10-16')
registration=json.loads((W/'registration.json').read_text())
assert hashlib.sha256((W/'protocol.json').read_bytes()).hexdigest()==registration['protocol_sha256']
assert (O/'protocol.json').read_bytes()==(W/'protocol.json').read_bytes()
before=json.loads((W/'v1_before.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
x=pd.read_pickle('work/olist-logistic/model_frame.pkl').sort_index()
train=x[x.split=='train'];cal=x[x.split=='validation'];ev=x[x.split=='test']
assert [(len(d),int(d.y90.sum())) for d in [train,cal,ev]]==[(6459,71),(6862,60),(7893,87)]
assert train.t0.max()+pd.Timedelta(days=90)<cal.t0.min()
assert cal.t0.max()+pd.Timedelta(days=90)<ev.t0.min()
assert set(train.index).isdisjoint(cal.index) and set(train.index).isdisjoint(ev.index) and set(cal.index).isdisjoint(ev.index)
colsA=['log_amount','log_item_count','log_weight_g','log_volume_cm3','log_distance_km','multi_seller','month_sin','month_cos','weight_missing','dimensions_missing','distance_missing','log_delivery_days','late','log_late_days','log_freight_ratio']
colsB=[v for v in colsA if v not in ['month_sin','month_cos']]
cats=['category_model','customer_state']
assert not set(colsA+cats)&{'y90','event_time','t0','customer_unique_id','split'}
def pipeline(cols):
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),cols),('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=200,sparse_output=False),cats)],sparse_threshold=0)
    return Pipeline([('prep',pre),('model',LogisticRegression(C=.1,solver='lbfgs',max_iter=3000,tol=1e-8,class_weight=None))])
models={};logits={};checks=[]
for name,cols in [('A',colsA),('B',colsB)]:
    model=pipeline(cols)
    with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):
        warnings.simplefilter('always');model.fit(train[cols+cats],train.y90)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in ws)
    for label,d in [('calibration',cal),('evaluation',ev)]: logits[(label,name)]=model.decision_function(d[cols+cats])
    models[name]=model
    joblib.dump(model,W/(name+'.joblib'))
    checks.append({'model':name,'converged':True,'iterations':int(model.named_steps['model'].n_iter_[0]),'features':len(model.named_steps['prep'].get_feature_names_out()),'warnings':[str(w.message) for w in ws]})

def solve_delta(z,y):
    rate=float(np.mean(y))
    if not 0<rate<1: return None
    return float(brentq(lambda delta:expit(z+delta).mean()-rate,-40,40,xtol=1e-12))
delta=solve_delta(logits[('calibration','B')],cal.y90.to_numpy())
assert delta is not None
pred={}
for label,d in [('calibration',cal),('evaluation',ev)]:
    for name in ['A','B']:pred[(label,name)]=expit(logits[(label,name)])
    pred[(label,'C')]=expit(logits[(label,'B')]+delta)
    pred[(label,'P_train')]=np.full(len(d),train.y90.mean())
    pred[(label,'P_cal')]=np.full(len(d),cal.y90.mean())
assert abs(pred[('calibration','C')].mean()-cal.y90.mean())<1e-12
old=pd.read_pickle('work/olist-logistic/validation_predictions.pkl').set_index('customer_unique_id').loc[cal.index]
assert np.allclose(pred[('calibration','A')],old.M2.to_numpy(),atol=1e-12,rtol=1e-10)
ranking={}
for label,d in [('calibration',cal),('evaluation',ev)]:
    b=pred[(label,'B')];c=pred[(label,'C')];k=math.ceil(.1*len(d))
    ib=np.argsort(-b,kind='stable');ic=np.argsort(-c,kind='stable')
    assert np.array_equal(ib,ic)
    assert average_precision_score(d.y90,b)==average_precision_score(d.y90,c)
    assert roc_auc_score(d.y90,b)==roc_auc_score(d.y90,c)
    ranking[label]={'full_rank_identical':True,'top10_identical':True,'rank_positions_changed':0,'top10_size':k,'top10_overlap':k,'AP_difference':0.0,'ROC_AUC_difference':0.0}

def metric(y,p,constant=False):
    y=np.asarray(y);k=math.ceil(.1*len(y));hits=int(y[np.argsort(-p,kind='stable')[:k]].sum())
    return {'n':len(y),'events':int(y.sum()),'observed_rate':float(y.mean()),'mean_prediction':float(p.mean()),'mean_error':float(p.mean()-y.mean()),'log_loss':float(log_loss(y,p,labels=[0,1])),'Brier':float(brier_score_loss(y,p)),'AP':float(average_precision_score(y,p)),'ROC_AUC':float(roc_auc_score(y,p)),'top10_hits':None if constant else hits,'top10_precision':None if constant else hits/k,'top10_recall':None if constant else hits/y.sum(),'top10_lift':None if constant else hits/k/y.mean()}
rows=[]
for label,d in [('calibration',cal),('evaluation',ev)]:
    for name in ['P_train','P_cal','A','B','C']:
        rows.append({'stage':label,'model':name,**metric(d.y90.to_numpy(),pred[(label,name)],name.startswith('P_'))})
metrics=pd.DataFrame(rows);metrics.to_csv(O/'metrics.csv',index=False,encoding='utf-8-sig')
for label,d in [('calibration',cal),('evaluation',ev)]:
    pd.DataFrame({'customer_unique_id':d.index,'y90':d.y90.to_numpy(),'state':d.customer_state.to_numpy(),**{n:pred[(label,n)] for n in ['A','B','C']}}).to_pickle(W/(label+'_predictions.pkl'))

def direct_ap(y,p):
    idx=np.argsort(-p,kind='stable');yy=y[idx];pp=p[idx];end=np.r_[np.flatnonzero(np.diff(pp)!=0),len(pp)-1];tp=np.cumsum(yy)[end]
    return float(np.sum(np.diff(np.r_[0,tp])*(tp/(end+1)))/y.sum())
for name in ['A','B','C']:
    p=pred[('evaluation',name)];y=ev.y90.to_numpy()
    assert abs(direct_ap(y,p)-average_precision_score(y,p))<1e-12
    assert abs(np.mean((y-p)**2)-brier_score_loss(y,p))<1e-12

def samples(d,kind,seed):
    rng=np.random.default_rng(seed);groups={s:np.flatnonzero(d.customer_state.to_numpy()==s) for s in d.customer_state.unique()};keys=list(groups)
    for _ in range(1000):
        yield rng.integers(0,len(d),len(d)) if kind=='customer' else np.concatenate([groups[s] for s in rng.choice(keys,len(keys),replace=True)])
def intervals(rows):
    df=pd.DataFrame(rows)
    return {col:{'low':float(df[col].quantile(.025)),'high':float(df[col].quantile(.975))} for col in df}
pairs=[('B','A'),('C','B'),('B','P_train'),('C','P_cal')]
evalboots=[]
for kind,seed in [('customer',20261018),('state_cluster',20261019)]:
    bs=[];yfull=ev.y90.to_numpy()
    for idx in samples(ev,kind,seed):
        y=yfull[idx]
        if not 0<y.mean()<1:continue
        stats={}
        for n in ['A','B','C','P_train','P_cal']:
            p=pred[('evaluation',n)][idx]
            stats[n]={'log_loss':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),'Brier':float(np.mean((y-p)**2)),'AP':direct_ap(y,p)}
        bs.append({f'{s}_{n}-{o}':stats[n][s]-stats[o][s] for n,o in pairs for s in ['log_loss','Brier','AP']})
    evalboots.append({'unit':kind,'replicates':len(bs),'intervals':intervals(bs),'scope':'fixed models and fixed calibrator; evaluation sampling uncertainty only'})
calboots=[]
for kind,seed in [('customer',20261020),('state_cluster',20261021)]:
    bs=[];yfull=cal.y90.to_numpy();z=logits[('calibration','B')]
    for idx in samples(cal,kind,seed):
        d=solve_delta(z[idx],yfull[idx])
        if d is None:continue
        p=expit(logits[('evaluation','B')]+d);y=ev.y90.to_numpy()
        bs.append({'delta':d,'odds_multiplier':float(np.exp(d)),'evaluation_mean_prediction':float(p.mean()),'evaluation_Brier':float(np.mean((y-p)**2)),'evaluation_log_loss':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))})
    calboots.append({'unit':kind,'replicates':len(bs),'intervals':intervals(bs),'scope':'calibration sampling uncertainty only, fixed bottom model and evaluation sample; not joint total uncertainty'})

def wilson(e,n):
    z=1.959963984540054;p=e/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return c-h,c+h
bins=[]
for label,d in [('calibration',cal),('evaluation',ev)]:
    y=d.y90.to_numpy()
    for name in ['A','B','C']:
        p=pred[(label,name)]
        for j,idx in enumerate(np.array_split(np.argsort(p,kind='stable'),5),1):
            n=len(idx);e=int(y[idx].sum());lo,hi=wilson(e,n)
            bins.append({'stage':label,'model':name,'quintile':j,'n':n,'events':e,'predicted':float(p[idx].mean()),'observed':e/n,'low':lo,'high':hi})
pd.DataFrame(bins).to_csv(O/'calibration_bins.csv',index=False,encoding='utf-8-sig')
unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()}
assert all(unchanged.values())
out={'protocol':json.loads((W/'protocol.json').read_text()),'registration':registration,'features':{'A':colsA+cats,'B':colsB+cats},'delta':delta,'odds_multiplier':float(np.exp(delta)),'metrics':rows,'ranking':ranking,'fit_checks':checks,'evaluation_bootstrap':evalboots,'calibrator_bootstrap':calboots,'checks':{'V1_preserved':True,'V1_files_checked':len(unchanged),'A_calibration_predictions_reproduce_V1_M2':True,'calibration_mean_matches_by_construction':True,'disjoint_time_splits':True,'labels_mature_before_next_stage':True,'bottom_models_fitted':2,'hyperparameter_candidates':1,'calibration_methods':1,'independent_AP_Brier_check':True,'no_refit_after_calibration':True,'no_LightGBM':True,'evaluation_not_fresh':True}}
(O/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'v1_preservation.json').write_text(json.dumps({'before_sha256':before,'unchanged_after':unchanged},ensure_ascii=False,indent=2),encoding='utf-8')
for f in ['registration.json','run_fixed.py']:shutil.copy2(W/f,O/f)
print(metrics.to_string(index=False))
print('DELTA',delta,'ODDS_MULTIPLIER',np.exp(delta))
print('EVALUATION_INTERVALS',json.dumps(evalboots))
print('CALIBRATOR_INTERVALS',json.dumps(calboots))
print('CHECKS',json.dumps(out['checks']))
