from pathlib import Path
import json, hashlib, datetime, math, shutil, io, contextlib
import numpy as np
import pandas as pd
import joblib, lightgbm as lgb, sklearn, scipy
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss

W=Path('work/olist-lightgbm');O=Path('outputs/olist-lightgbm-v3-2026-10-16')
protocol=json.loads((W/'protocol.json').read_text())
registration=json.loads((W/'registration.json').read_text())
assert lgb.__version__=='4.7.0'
assert hashlib.sha256((W/'protocol.json').read_bytes()).hexdigest()==registration['protocol_sha256']
before=json.loads((W/'previous_before.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
x=pd.read_pickle('work/olist-logistic/model_frame.pkl').sort_index()
train=x[x.split=='train'];dev=x[x.split=='validation'];ev=x[x.split=='test']
assert [(len(d),int(d.y90.sum())) for d in [train,dev,ev]]==[(6459,71),(6862,60),(7893,87)]
assert train.t0.max()+pd.Timedelta(days=90)<dev.t0.min()
assert dev.t0.max()+pd.Timedelta(days=90)<ev.t0.min()
assert set(train.index).isdisjoint(ev.index) and set(dev.index).isdisjoint(ev.index)
lr=joblib.load('work/olist-calibration/B.joblib');prep=lr.named_steps['prep']
feature_names=list(prep.get_feature_names_out())
def base_matrix(df):return pd.DataFrame(prep.transform(df),columns=feature_names,index=df.index)
Xt=base_matrix(train);Xd=base_matrix(dev)
regional=['local_category_share','log_relative_share']
extra=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
rt=extra.fit_transform(train[regional]);rd=extra.transform(dev[regional])
Xt2=Xt.copy();Xd2=Xd.copy()
for j,name in enumerate(regional):Xt2['region__'+name]=rt[:,j];Xd2['region__'+name]=rd[:,j]
assert not any('month' in name for name in Xt2.columns)
assert len(Xt2.columns)==len(Xt.columns)+2
models={};model_records={};curves=[]
for name,tx,dx in [('G1',Xt,Xd),('G2',Xt2,Xd2)]:
    model=lgb.LGBMClassifier(**protocol['parameters'])
    history={};log=io.StringIO()
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        model.fit(tx,train.y90,eval_set=[(dx,dev.y90)],eval_names=['development'],eval_metric='binary_logloss',callbacks=[lgb.early_stopping(stopping_rounds=50,first_metric_only=True,min_delta=.00001,verbose=False),lgb.record_evaluation(history)])
    (O/(name+'_training.log')).write_text(log.getvalue(),encoding='utf-8')
    assert set(history)=={'development'} and set(history['development'])=={'binary_logloss'}
    scores=history['development']['binary_logloss'];best=int(model.best_iteration_)
    assert 1<=best<=len(scores)<=500
    selected=float(scores[best-1])
    # Independently reconstruct the callback's min_delta-aware best selection.
    chosen=0;best_score=scores[0]
    for i,s in enumerate(scores[1:],1):
        if s<best_score-.00001:chosen=i;best_score=s
    assert chosen+1==best
    model.booster_.save_model(str(O/(name+'_model.txt')),num_iteration=best)
    dump=model.booster_.dump_model(num_iteration=best)
    model_records[name]={'best_iteration':best,'evaluated_iterations':len(scores),'selected_dev_logloss':selected,'numerical_min_dev_logloss':float(min(scores)),'numerical_min_iteration':int(np.argmin(scores)+1),'selected_tree_count':len(dump['tree_info']),'max_leaves_in_selected_trees':max(t['num_leaves'] for t in dump['tree_info']),'actual_parameters':model.get_params(),'model_sha256':hashlib.sha256((O/(name+'_model.txt')).read_bytes()).hexdigest()}
    for i,s in enumerate(scores,1):curves.append({'model':name,'iteration':i,'development_log_loss':s,'selected':i==best})
    models[name]=model
    print('SAVED',name,json.dumps({k:model_records[name][k] for k in ['best_iteration','evaluated_iterations','selected_dev_logloss','selected_tree_count']}),flush=True)
pd.DataFrame(curves).to_csv(O/'learning_curves.csv',index=False,encoding='utf-8-sig')
joblib.dump(extra,W/'regional_preprocessor.joblib')
locked={'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':registration['protocol_sha256'],'models':model_records,'evaluation_predictions_generated':False}
(O/'locked_models_before_evaluation.json').write_text(json.dumps(locked,indent=2),encoding='utf-8')

# Only after both models/iterations are saved and locked: generate evaluation predictions.
Xe=base_matrix(ev);Xe2=Xe.copy();re=extra.transform(ev[regional])
for j,name in enumerate(regional):Xe2['region__'+name]=re[:,j]
pred={};rows=[]
def metrics(y,p,constant=False):
    y=np.asarray(y);k=math.ceil(.1*len(y));order=np.argsort(-p,kind='stable');hits=int(y[order[:k]].sum());threshold=p[order[k-1]]
    return {'n':len(y),'events':int(y.sum()),'observed_rate':float(y.mean()),'mean_prediction':float(p.mean()),'AP':float(average_precision_score(y,p)),'ROC_AUC':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'Brier':float(brier_score_loss(y,p)),'top_n':k,'top_hits':None if constant else hits,'top_precision':None if constant else hits/k,'top_recall':None if constant else hits/y.sum(),'top_lift':None if constant else hits/k/y.mean(),'boundary_tie_count':None if constant else int((p==threshold).sum()),'boundary_ties_selected':None if constant else int((p[order[:k]]==threshold).sum())}
for stage,d,mat1,mat2 in [('development',dev,Xd,Xd2),('evaluation',ev,Xe,Xe2)]:
    pred[(stage,'LR')]=lr.predict_proba(d)[:,1]
    for name,mat in [('G1',mat1),('G2',mat2)]:
        pred[(stage,name)]=models[name].predict_proba(mat,num_iteration=model_records[name]['best_iteration'])[:,1]
        # Saved text model must reproduce the selected model exactly.
        saved=lgb.Booster(model_file=str(O/(name+'_model.txt')))
        assert np.allclose(saved.predict(mat),pred[(stage,name)],atol=1e-14,rtol=0)
    pred[(stage,'P_train')]=np.repeat(train.y90.mean(),len(d));pred[(stage,'P_dev')]=np.repeat(dev.y90.mean(),len(d))
    for name in ['LR','G1','G2','P_train','P_dev']:
        p=pred[(stage,name)];assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
        rows.append({'stage':stage,'model':name,**metrics(d.y90.to_numpy(),p,name.startswith('P_'))})
old=pd.read_pickle('work/olist-calibration/evaluation_predictions.pkl').set_index('customer_unique_id').loc[ev.index]
assert np.allclose(old.B.to_numpy(),pred[('evaluation','LR')],atol=1e-14,rtol=0)
pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'customer_unique_id':ev.index,'y90':ev.y90.to_numpy(),'state':ev.customer_state.to_numpy(),**{m:pred[('evaluation',m)] for m in ['LR','G1','G2']}}).to_pickle(W/'evaluation_predictions.pkl')

def direct_ap(y,p):
    idx=np.argsort(-p,kind='stable');yy=y[idx];pp=p[idx];end=np.r_[np.flatnonzero(np.diff(pp)!=0),len(pp)-1];tp=np.cumsum(yy)[end]
    return float(np.sum(np.diff(np.r_[0,tp])*tp/(end+1))/y.sum())
y=ev.y90.to_numpy()
for name in ['LR','G1','G2','P_train','P_dev']:
    p=pred[('evaluation',name)]
    assert abs(direct_ap(y,p)-average_precision_score(y,p))<1e-12
    assert abs(np.mean((y-p)**2)-brier_score_loss(y,p))<1e-12
bootstrap=[]
for kind,seed in [('customer',20261023),('state_cluster',20261024)]:
    rng=np.random.default_rng(seed);states=ev.customer_state.to_numpy();keys=np.unique(states);groups={s:np.flatnonzero(states==s) for s in keys};bs=[]
    for _ in range(1000):
        idx=rng.integers(0,len(ev),len(ev)) if kind=='customer' else np.concatenate([groups[s] for s in rng.choice(keys,len(keys),replace=True)])
        yy=y[idx]
        if not 0<yy.mean()<1:continue
        stats={}
        for name in ['LR','G1','G2','P_train','P_dev']:
            p=pred[('evaluation',name)][idx]
            stats[name]={'AP':direct_ap(yy,p),'log_loss':float(np.mean(-yy*np.log(p)-(1-yy)*np.log1p(-p))),'Brier':float(np.mean((yy-p)**2))}
        pairs=[('G1','LR'),('G2','G1'),('G1','P_train'),('G2','P_train')]
        bs.append({f'{metric}_{new}-{old}':stats[new][metric]-stats[old][metric] for new,old in pairs for metric in ['AP','log_loss','Brier']})
    frame=pd.DataFrame(bs)
    bootstrap.append({'unit':kind,'replicates':len(frame),'intervals':{c:{'low':float(frame[c].quantile(.025)),'high':float(frame[c].quantile(.975))} for c in frame}})
bins=[]
for name in ['LR','G1','G2']:
    p=pred[('evaluation',name)]
    for i,idx in enumerate(np.array_split(np.argsort(p,kind='stable'),5),1):
        n=len(idx);events=int(y[idx].sum());rate=events/n;z=1.959963984540054;den=1+z*z/n;c=(rate+z*z/(2*n))/den;h=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
        bins.append({'model':name,'quintile':i,'n':n,'events':events,'predicted':float(p[idx].mean()),'observed':rate,'low':c-h,'high':c+h})
pd.DataFrame(bins).to_csv(O/'calibration_bins.csv',index=False,encoding='utf-8-sig')
importance=[]
for name in ['G1','G2']:
    booster=models[name].booster_;iteration=model_records[name]['best_iteration']
    for feature,gain,splits in zip(booster.feature_name(),booster.feature_importance('gain',iteration=iteration),booster.feature_importance('split',iteration=iteration)):
        importance.append({'model':name,'feature':feature,'gain':float(gain),'splits':int(splits)})
pd.DataFrame(importance).to_csv(O/'training_feature_importance.csv',index=False,encoding='utf-8-sig')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
summary={'protocol':protocol,'registration':registration,'locked_models':locked,'metrics':rows,'bootstrap':bootstrap,'features':{'G1':list(Xt.columns),'G2':list(Xt2.columns)},'versions':{'lightgbm':lgb.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'numpy':np.__version__},'checks':{'previous_files_unchanged':len(before),'LightGBM_fits':2,'parameter_sets':1,'seed_values':1,'calibration_or_refit':False,'independent_AP_Brier':True,'saved_models_reproduce_predictions':True,'LR_reproduces_V2_B':True,'labels_mature_before_next_stage':True,'models_locked_before_evaluation':True,'evaluations_are_exploratory':True,'implementation_fixes':[]}}
(O/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'preservation.json').write_text(json.dumps({'before_sha256':before,'all_unchanged':True},ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['run_fixed.py','registration.json']:shutil.copy2(W/name,O/name)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
print('BOOTSTRAP',json.dumps(bootstrap),flush=True)
print('REGION_IMPORTANCE',[i for i in importance if i['feature'].startswith('region__')],flush=True)
