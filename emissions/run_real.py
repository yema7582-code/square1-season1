import argparse,os
_ap=argparse.ArgumentParser()
_ap.add_argument('--train',default=os.environ.get('TRAIN_CSV','./data/train.csv'))
_ap.add_argument('--test',default=os.environ.get('TEST_CSV','./data/test.csv'))
_ap.add_argument('--out',default=os.environ.get('OUT_DIR','./submissions'))
ARGS=_ap.parse_args();os.makedirs(ARGS.out,exist_ok=True)
import json,time,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import GroupKFold
import model as M
t=time.time();log=open('real_progress.log','w')
def P(*a): print(*a,round(time.time()-t),file=log,flush=True)
M.model=lambda: lgb.LGBMRegressor(objective='regression',n_estimators=800,learning_rate=0.04,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.7,reg_lambda=2.0,random_state=7,verbose=-1,n_jobs=3)
tr=pd.read_csv(ARGS.train);te=pd.read_csv(ARGS.test)
al=M.build(tr,te);A=al[al._tr==1].reset_index(drop=True);B=al[al._tr==0].reset_index(drop=True)
rm=lambda p,y:float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2)))
raw=[c for c in ['year','fuel_count','capacity_mw','lat_r','lon_r'] if c in tr]
sc,sb,sd=[],[],[];oof=np.zeros(len(A))
for k,(i,j) in enumerate(GroupKFold(5).split(A,groups=A.cell)):
    a,b=A.iloc[i].reset_index(drop=True),A.iloc[j].reset_index(drop=True)
    p,_,_=M.fit_predict(a,b);oof[j]=p;sc.append(rm(p,b.annual_co2e_t))
    med=(a.annual_co2e_t/a.capacity_mw).groupby(a.primary_fuel).median();sb.append(rm(b.capacity_mw*b.primary_fuel.map(med).astype(float).fillna(med.median()),b.annual_co2e_t))
    X=lambda d:tr.set_index('id').loc[d.id,raw].assign(pf=pd.Categorical(tr.set_index('id').loc[d.id,'primary_fuel'],categories=sorted(tr.primary_fuel.unique())))
    dg=lgb.LGBMRegressor(verbose=-1,n_jobs=3).fit(X(a),np.log1p(a.annual_co2e_t));sd.append(rm(np.expm1(dg.predict(X(b))),b.annual_co2e_t))
    P(k,sc[-1],sb[-1],sd[-1])
e=pd.Series((np.log1p(oof)-np.log1p(A.annual_co2e_t))**2)
rep={'cv_rmsle_model':float(np.mean(sc)),'cv_rmsle_baseline_cap_x_median':float(np.mean(sb)),'cv_rmsle_default_gbt_log':float(np.mean(sd)),
 'rmsle_by_fuel':e.groupby(A.primary_fuel.values).mean().pow(.5).round(3).to_dict(),'rmsle_by_year':e.groupby(A.year.values).mean().pow(.5).round(3).to_dict()}
P(json.dumps(rep));json.dump(rep,open('report_real.json','w'),indent=1)
S=os.path.join(ARGS.out,'')
p,_,_=M.fit_predict(A,B);pd.DataFrame({'id':B.id,'annual_co2e_t':np.round(p,2)}).to_csv(S+'sub1_residual_gbm.csv',index=False);P('wrote1')
M.model=lambda: lgb.LGBMRegressor(objective='regression',n_estimators=800,learning_rate=0.04,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.7,reg_lambda=2.0,random_state=11,verbose=-1,n_jobs=3)
p2,_,_=M.fit_predict(A,B);pb=np.expm1((np.log1p(p)+np.log1p(p2))/2)
pd.DataFrame({'id':B.id,'annual_co2e_t':np.round(pb,2)}).to_csv(S+'sub2_residual_gbm_2seed.csv',index=False);P('wrote2')
