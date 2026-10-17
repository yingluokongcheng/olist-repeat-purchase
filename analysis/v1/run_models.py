from pathlib import Path
import json, math, warnings, hashlib, shutil, sys
import numpy as np
import pandas as pd
import sklearn, scipy, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path('work/olist-logistic')
OUT=Path('outputs/olist-logistic-2026-10-16')
OUT.mkdir(parents=True,exist_ok=True)
protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
x=pd.read_pickle(ROOT/'model_frame.pkl').sort_index()
base=['log_amount','log_item_count','log_weight_g','log_volume_cm3','log_distance_km','multi_seller','month_sin','month_cos','weight_missing','dimensions_missing','distance_missing']
fulfill=['log_delivery_days','late','log_late_days','log_freight_ratio']
regional=['local_category_share','log_relative_share']
cats=['category_model','customer_state']
features={'M1':base,'M2':base+fulfill,'M3':base+fulfill+regional}
for cols in features.values():
    assert not (set(cols+cats)&{'y90','y60','y120','event_time','order_id','customer_unique_id','split','order_status','t0'})
train=x[x.split.eq('train')]; val=x[x.split.eq('validation')]
refit=x[x.t0.lt('2018-01-01')]; test=x[x.split.eq('test')]
assert (len(train),int(train.y90.sum()),len(val),int(val.y90.sum()),len(refit),int(refit.y90.sum()),len(test),int(test.y90.sum()))==(6459,71,6862,60,25780,303,7893,87)
assert train.t0.max()+pd.Timedelta(days=90)<val.t0.min()
assert refit.t0.max()+pd.Timedelta(days=90)<test.t0.min()
assert set(refit.index).isdisjoint(test.index)

def make_model(cols):
    prep=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),cols),
        ('cat',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=200,sparse_output=False),cats)
    ],sparse_threshold=0)
    return Pipeline([('prep',prep),('model',LogisticRegression(C=.1,solver='lbfgs',max_iter=3000,tol=1e-8,class_weight=None))])

def metrics(y,p,constant=False):
    y=np.asarray(y); p=np.asarray(p); k=math.ceil(.1*len(y))
    ids=np.argsort(-p,kind='stable')[:k]; hits=int(y[ids].sum())
    return {'n':len(y),'events':int(y.sum()),'prevalence':float(y.mean()),'AP':float(average_precision_score(y,p)),
        'Brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
        'ROC_AUC':float(roc_auc_score(y,p)),'mean_prediction':float(p.mean()),'top10_n':k,
        'top10_hits':None if constant else hits,'top10_precision':None if constant else hits/k,
        'top10_recall':None if constant else hits/y.sum(),'top10_lift':None if constant else hits/k/y.mean()}

results=[]; predictions={}; coef=[]; fit_checks=[]
for stage,fit,ev in [('validation',train,val),('test',refit,test)]:
    y=ev.y90.to_numpy()
    prior=float(fit.y90.mean()); pred=np.repeat(prior,len(ev))
    results.append({'stage':stage,'model':'M0',**metrics(y,pred,True)})
    predictions[(stage,'M0')]=pred
    for label,cols in features.items():
        model=make_model(cols)
        with warnings.catch_warnings(record=True) as ws, threadpool_limits(limits=2):
            warnings.simplefilter('always')
            model.fit(fit[cols+cats],fit.y90)
        bad=[str(w.message) for w in ws if issubclass(w.category,ConvergenceWarning)]
        assert not bad, bad
        p=model.predict_proba(ev[cols+cats])[:,1]
        assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        predictions[(stage,label)]=p
        results.append({'stage':stage,'model':label,**metrics(y,p)})
        names=model.named_steps['prep'].get_feature_names_out()
        fit_checks.append({'stage':stage,'model':label,'n_features':len(names),'iterations':int(model.named_steps['model'].n_iter_[0]),'converged':True,'warnings':[str(w.message) for w in ws]})
        if stage=='test':
            joblib.dump(model,ROOT/(label+'.joblib'))
            if label=='M3':
                fit_matrix=model.named_steps['prep'].transform(fit[cols+cats])
                test_matrix=model.named_steps['prep'].transform(ev[cols+cats])
                shift=(test_matrix.mean(axis=0)-fit_matrix.mean(axis=0))*model.named_steps['model'].coef_[0]
                pd.DataFrame({'feature':names,'test_minus_refit_logit_contribution':shift}).to_csv(OUT/'prediction_shift_decomposition.csv',index=False)
            for name,value in zip(names,model.named_steps['model'].coef_[0]):
                coef.append({'model':label,'feature':name,'coefficient':float(value),'odds_ratio':float(np.exp(value)),'note':'numeric coefficients per training SD after log transforms; categorical relative to regularized coding; not causal'})
        print(stage,label,json.dumps(results[-1]),flush=True)

results_df=pd.DataFrame(results)
results_df.to_csv(OUT/'model_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(coef).to_csv(OUT/'coefficients.csv',index=False,encoding='utf-8-sig')
yp=test.y90.to_numpy()
predframe=pd.DataFrame({'customer_unique_id':test.index,'y90':yp,'state':test.customer_state.to_numpy(),'t0':test.t0.to_numpy(),**{m:predictions[('test',m)] for m in ['M0','M1','M2','M3']}})
predframe.to_pickle(ROOT/'test_predictions.pkl')
vframe=pd.DataFrame({'customer_unique_id':val.index,'y90':val.y90.to_numpy(),**{m:predictions[('validation',m)] for m in ['M0','M1','M2','M3']}})
vframe.to_pickle(ROOT/'validation_predictions.pkl')

# Independent AP calculation checks probability ranks without sklearn's metric.
def direct_ap(y,p):
    order=np.argsort(-p,kind='stable'); yy=y[order]; pp=p[order]
    end=np.r_[np.flatnonzero(np.diff(pp)!=0),len(pp)-1]
    tp=np.cumsum(yy)[end]; prec=tp/(end+1)
    return float(np.sum(np.diff(np.r_[0,tp])*prec)/y.sum())
for m in ['M0','M1','M2','M3']:
    p=predictions[('test',m)]
    assert abs(direct_ap(yp,p)-average_precision_score(yp,p))<1e-12
    assert abs(np.mean((yp-p)**2)-brier_score_loss(yp,p))<1e-12

def bootstrap(kind,reps=1000):
    rng=np.random.default_rng(20261016 if kind=='customer' else 20261017)
    states=test.customer_state.to_numpy(); unique=np.unique(states)
    groups={s:np.flatnonzero(states==s) for s in unique}
    records=[]
    for i in range(reps):
        if kind=='customer': idx=rng.integers(0,len(test),len(test))
        else: idx=np.concatenate([groups[s] for s in rng.choice(unique,size=len(unique),replace=True)])
        y=yp[idx]
        if y.sum()==0 or y.sum()==len(y): continue
        ap={}; bs={}
        for m in ['M0','M1','M2','M3']:
            p=predictions[('test',m)][idx]
            ap[m]=direct_ap(y,p); bs[m]=float(np.mean((y-p)**2))
        row={f'AP_{m}':ap[m] for m in ap}
        for newer,older in [('M1','M0'),('M2','M1'),('M3','M2'),('M3','M1')]:
            row[f'dAP_{newer}-{older}']=ap[newer]-ap[older]
            row[f'dBrier_{newer}-{older}']=bs[newer]-bs[older]
        records.append(row)
    b=pd.DataFrame(records)
    return {'unit':kind,'replicates':len(b),'intervals':{c:{'low':float(b[c].quantile(.025)),'high':float(b[c].quantile(.975))} for c in b}}

boot=[bootstrap('customer'),bootstrap('state_cluster')]
cal=[]
def wilson(e,n):
    z=1.959963984540054;p=e/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
for m in ['M1','M2','M3']:
    p=predictions[('test',m)]; indices=np.argsort(p,kind='stable')
    for k,idx in enumerate(np.array_split(indices,5),1):
        n=len(idx);e=int(yp[idx].sum());lo,hi=wilson(e,n)
        cal.append({'model':m,'quintile':k,'n':n,'events':e,'predicted':float(p[idx].mean()),'observed':e/n,'low':lo,'high':hi})
pd.DataFrame(cal).to_csv(OUT/'calibration.csv',index=False,encoding='utf-8-sig')

# Cases are descriptive on the whole audited cohort, never used to revise models.
case_rows=[]
for category in ['bed_bath_table','sports_leisure']:
    d=x[x.single_category & x.category.eq(category)].copy()
    d['match_group']=pd.qcut(d.log_relative_share,4,labels=['Q1','Q2','Q3','Q4'],duplicates='raise')
    for dimension in ['overall','late','match_group']:
        groups=[('all',d)] if dimension=='overall' else d.groupby(dimension,observed=True)
        for name,g in groups:
            n=len(g);e=int(g.y90.sum());lo,hi=wilson(e,n)
            case_rows.append({'category':category,'dimension':dimension,'group':str(name),'customers':n,'repeat90':e,'rate':e/n,'low':lo,'high':hi,'median_delivery_days':float(g.delivery_days.median()),'median_freight_ratio':float(g.freight_ratio.median())})
pd.DataFrame(case_rows).to_csv(OUT/'descriptive_cases.csv',index=False,encoding='utf-8-sig')
subgroups=[]
for state,g in predframe.groupby('state'):
    row={'state':state,'customers':len(g),'events':int(g.y90.sum())}
    for m in ['M1','M2','M3']:
        row[m+'_mean_prediction']=float(g[m].mean());row[m+'_Brier']=float(np.mean((g.y90-g[m])**2))
        row[m+'_AP']=float(average_precision_score(g.y90,g[m])) if g.y90.sum()>=10 else None
    subgroups.append(row)
pd.DataFrame(subgroups).to_csv(OUT/'state_diagnostics.csv',index=False,encoding='utf-8-sig')

drift=[]
for col in base+fulfill+regional:
    drift.append({'feature':col,'refit_mean':float(refit[col].mean()),'test_mean':float(test[col].mean()),'refit_sd':float(refit[col].std()),'standardized_mean_difference':float((test[col].mean()-refit[col].mean())/refit[col].std()) if refit[col].std()>0 else None})
pd.DataFrame(drift).to_csv(OUT/'feature_shift.csv',index=False,encoding='utf-8-sig')
summary={'protocol':protocol,'feature_sets':{m:cols+cats for m,cols in features.items()},'metrics':results,'bootstrap':boot,'fit_checks':fit_checks,'feature_checks':json.loads((ROOT/'feature_checks.json').read_text()),'versions':{'python':sys.version,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'numpy':np.__version__,'pandas':pd.__version__},'checks':{'AP_independent_recalculation':True,'Brier_independent_recalculation':True,'disjoint_refit_test':True,'mature_labels_before_evaluation':True,'no_model_changes_after_test':True,'no_LightGBM':True,'test_events':87,'models_fitted':6,'hyperparameter_candidates':1},'case_scope':'full audited cohort, exploratory description, not held-out causal validation'}
(OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
for file in ['protocol.json','features.py','run_models.py']:
    shutil.copy2(ROOT/file,OUT/file)
print('BOOTSTRAP',json.dumps(boot),flush=True)
print('COMPLETE',str(OUT),flush=True)
